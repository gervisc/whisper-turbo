"""
whisper-turbo — Fast STT proxy for whisper.cpp with OpenVINO acceleration.

Routes audio to a running whisper.cpp server (with model kept warm in memory)
for significantly faster speech-to-text on Intel hardware.

Endpoints:
    POST /v1/transcribe   — generic STT (multipart file upload)
    POST /v1/subtitles    — upload media and return/save SRT subtitles
    POST /api/willow       — Willow/WIS compatible (raw PCM body + ESP32 headers)
    GET  /health           — health check (proxy + whisper-server status)
"""

import os
import struct
import time
import logging
import argparse
import asyncio
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

VERSION = "1.0.0"

# ──────────────────────────────────────────────
# CONFIG — all values from .env with sensible defaults
# ──────────────────────────────────────────────

load_dotenv()

WHISPER_SERVER_HOST = os.getenv("WHISPER_SERVER_HOST", "127.0.0.1")
WHISPER_SERVER_PORT = os.getenv("WHISPER_SERVER_PORT", "19003")
WHISPER_SERVER_URL = f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}/inference"
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "15"))
WHISPER_SUBTITLE_TIMEOUT = int(os.getenv("WHISPER_SUBTITLE_TIMEOUT", "1800"))
MEDIA_CONVERT_TIMEOUT = int(os.getenv("MEDIA_CONVERT_TIMEOUT", "3600"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "ggml-small.en.bin")
SUBTITLE_OUTPUT_DIR = Path(os.getenv("SUBTITLE_OUTPUT_DIR", "subtitles")).expanduser().resolve()
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

SUPPORTED_MEDIA_EXTENSIONS = {
    ".3gp",
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".flv",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".opus",
    ".ts",
    ".wav",
    ".webm",
    ".wmv",
}

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("whisper-turbo")

# ──────────────────────────────────────────────
# AUDIO HELPERS
# ──────────────────────────────────────────────

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000,
               bits: int = 16, channels: int = 1) -> bytes:
    """
    Wrap raw PCM bytes in a WAV container.
    Devices like the ESP32 send raw PCM — whisper.cpp expects WAV format.
    """
    bytes_per_sample = bits // 8
    data_size = len(pcm_bytes)
    # Standard 44-byte WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,                             # file size minus 8
        b'WAVE',
        b'fmt ',
        16,                                         # fmt chunk size
        1,                                          # PCM format
        channels,
        sample_rate,
        sample_rate * channels * bytes_per_sample,  # byte rate
        channels * bytes_per_sample,                # block align
        bits,
        b'data',
        data_size,
    )
    return header + pcm_bytes


def clean_text(text: str) -> str:
    """Clean up whisper.cpp output — collapse whitespace and strip."""
    return " ".join(text.split()).strip()


def audio_duration_ms(body_len: int, sample_rate: int = 16000,
                      bits: int = 16, channels: int = 1) -> float:
    """Calculate audio duration in milliseconds from raw PCM byte length."""
    bytes_per_sample = bits // 8
    denominator = sample_rate * channels * bytes_per_sample
    if denominator == 0:
        return 0
    return (body_len / denominator) * 1000

# ──────────────────────────────────────────────
# CORE TRANSCRIPTION
# ──────────────────────────────────────────────

async def transcribe(wav_bytes: bytes, duration_ms: float,
                     language: str = None) -> dict:
    """
    Send audio to whisper.cpp server and return transcription result.
    The whisper.cpp server keeps the model loaded in memory for fast inference.
    """
    lang = language or WHISPER_LANGUAGE
    t_start = time.time()

    async with httpx.AsyncClient(timeout=WHISPER_TIMEOUT) as client:
        response = await client.post(
            WHISPER_SERVER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json", "language": lang},
        )
        response.raise_for_status()
        result = response.json()

    text = clean_text(result.get("text", ""))
    elapsed_ms = round((time.time() - t_start) * 1000, 1)
    speedup = round(duration_ms / elapsed_ms, 1) if elapsed_ms > 0 else 0

    log.info(
        f"{elapsed_ms}ms | {duration_ms:.0f}ms audio | "
        f"{speedup}x realtime | \"{text[:80]}\""
    )

    return {
        "text": text,
        "language": lang,
        "duration_ms": round(duration_ms),
        "infer_time_ms": elapsed_ms,
        "speedup": speedup,
    }


def normalize_language(language: Optional[str]) -> str:
    """Normalize language codes enough for whisper.cpp request parameters."""
    lang = (language or WHISPER_LANGUAGE or "en").strip().lower()
    return lang or "en"


def is_english_only_model() -> bool:
    return ".en." in WHISPER_MODEL or WHISPER_MODEL.endswith(".en.bin")


def validate_media_path(media_path: str) -> Path:
    """Validate a local media path before handing it to ffmpeg."""
    path = Path(media_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Media file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Media path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValueError(f"Unsupported media extension '{path.suffix}'. Supported: {supported}")
    return path


def validate_media_extension(filename: str) -> str:
    """Validate media extension from a client-supplied filename."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise ValueError(f"Unsupported media extension '{suffix}'. Supported: {supported}")
    return suffix


def subtitle_filename(media_filename: str, srt_language: str) -> str:
    """Build a stable SRT filename from an uploaded media filename."""
    safe_name = Path(media_filename or "subtitle").name
    stem = Path(safe_name).stem or "subtitle"
    lang = normalize_language(srt_language)
    return f"{stem}.{lang}.srt"


async def save_upload_to_temp(file: UploadFile) -> Path:
    """Stream an uploaded media file to a temporary file on the server."""
    filename = Path(file.filename or "upload").name
    suffix = validate_media_extension(filename)
    temp_file = tempfile.NamedTemporaryFile(
        prefix="whisper-turbo-upload-",
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Empty media file")

    return temp_path


def extract_wav(media_path: Path) -> Path:
    """
    Extract audio from video/audio into the 16-bit mono WAV format whisper.cpp expects.
    The caller is responsible for deleting the returned temp file.
    """
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for movie subtitle generation")

    temp_file = tempfile.NamedTemporaryFile(prefix="whisper-turbo-", suffix=".wav", delete=False)
    wav_path = Path(temp_file.name)
    temp_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=MEDIA_CONVERT_TIMEOUT,
        )
    except Exception:
        wav_path.unlink(missing_ok=True)
        raise

    return wav_path


async def transcribe_srt(wav_path: Path, movie_language: str, srt_language: str) -> dict:
    """
    Ask whisper.cpp for SRT output.
    Whisper can transcribe in the source language, or translate supported source languages to English.
    """
    source_lang = normalize_language(movie_language)
    target_lang = normalize_language(srt_language)
    translate = source_lang != target_lang

    if translate and target_lang != "en":
        raise ValueError(
            "Whisper translation only outputs English. Use srt_language=en or keep "
            "srt_language the same as movie_language."
        )
    if translate and is_english_only_model():
        raise ValueError(
            f"{WHISPER_MODEL} is an English-only model and cannot translate "
            "non-English movies. Configure a multilingual model such as ggml-small.bin."
        )

    data = {
        "response_format": "srt",
        "language": source_lang,
    }
    if translate:
        data["translate"] = "true"

    t_start = time.time()
    async with httpx.AsyncClient(timeout=WHISPER_SUBTITLE_TIMEOUT) as client:
        with wav_path.open("rb") as wav_file:
            response = await client.post(
                WHISPER_SERVER_URL,
                files={"file": ("audio.wav", wav_file, "audio/wav")},
                data=data,
            )
        response.raise_for_status()

    elapsed_ms = round((time.time() - t_start) * 1000, 1)
    srt_text = response.text.strip()
    log.info(
        f"{elapsed_ms}ms | subtitles | {source_lang}->{target_lang} | "
        f"{len(srt_text)} chars"
    )

    return {
        "srt": srt_text + ("\n" if srt_text else ""),
        "movie_language": source_lang,
        "srt_language": target_lang,
        "translated": translate,
        "infer_time_ms": elapsed_ms,
    }


async def create_subtitles(
    media_path: str,
    movie_language: Optional[str] = None,
    srt_language: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Create an SRT file for a local media file."""
    source_path = validate_media_path(media_path)
    target_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else source_path.with_suffix(".srt")
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    wav_path = await asyncio.to_thread(extract_wav, source_path)
    try:
        result = await transcribe_srt(
            wav_path,
            movie_language or WHISPER_LANGUAGE,
            srt_language or "en",
        )
    finally:
        wav_path.unlink(missing_ok=True)

    target_path.write_text(result["srt"], encoding="utf-8")
    return {
        "media_path": str(source_path),
        "srt_path": str(target_path),
        "movie_language": result["movie_language"],
        "srt_language": result["srt_language"],
        "translated": result["translated"],
        "infer_time_ms": result["infer_time_ms"],
    }


async def create_subtitles_from_upload(
    file: UploadFile,
    movie_language: Optional[str] = None,
    srt_language: Optional[str] = None,
) -> dict:
    """Create an SRT file from an uploaded media file and save it in the output directory."""
    SUBTITLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "upload").name
    target_path = SUBTITLE_OUTPUT_DIR / subtitle_filename(filename, srt_language or "en")

    media_path = await save_upload_to_temp(file)
    try:
        result = await create_subtitles(
            media_path=str(media_path),
            movie_language=movie_language or WHISPER_LANGUAGE,
            srt_language=srt_language or "en",
            output_path=str(target_path),
        )
    finally:
        media_path.unlink(missing_ok=True)

    result["uploaded_filename"] = filename
    return result

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

app = FastAPI(
    title="whisper-turbo",
    description="Fast STT proxy for whisper.cpp with OpenVINO acceleration",
    version=VERSION,
)


@app.get("/")
async def root():
    """Basic info about the service."""
    return {
        "name": "whisper-turbo",
        "version": VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check — verifies both the proxy and whisper.cpp server are up."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}/health"
            )
            whisper_status = resp.json().get("status", "unknown")
    except Exception:
        whisper_status = "unreachable"

    status = "ok" if whisper_status == "ok" else "degraded"
    return JSONResponse(
        content={
            "status": status,
            "whisper_server": whisper_status,
            "language": WHISPER_LANGUAGE,
            "device": os.getenv("OPENVINO_DEVICE", "CPU"),
        },
        status_code=200 if status == "ok" else 503,
    )

# ──────────────────────────────────────────────
# GENERIC ENDPOINT — works with any client
# ──────────────────────────────────────────────

@app.post("/v1/transcribe")
async def transcribe_generic(
    file: UploadFile = File(...),
    language: str = Form(default=None),
):
    """
    Generic speech-to-text endpoint.

    Accepts a WAV or PCM audio file via multipart upload.
    Returns transcription as JSON.

    Example:
        curl -X POST http://localhost:19004/v1/transcribe \
            -F "file=@audio.wav" \
            -F "language=en"
    """
    body = await file.read()
    if not body:
        return JSONResponse({"text": "", "error": "Empty audio file"}, status_code=400)

    # Duration estimate — whisper.cpp handles format detection internally.
    # For WAV files the byte count includes headers, so this is approximate.
    dur = audio_duration_ms(len(body))

    try:
        result = await transcribe(body, dur, language)
        return JSONResponse(result)
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return JSONResponse(
            {"text": "", "error": str(e)},
            status_code=502,
        )


@app.post("/v1/subtitles")
async def subtitles_from_upload(
    file: UploadFile = File(...),
    movie_language: str = Form(default="en"),
    srt_language: str = Form(default="en"),
):
    """
    Create an SRT file from an uploaded movie/audio file.

    The model is selected by service config, not by this request. ffmpeg extracts
    audio from common video formats before sending WAV to whisper.cpp. The SRT is
    saved in SUBTITLE_OUTPUT_DIR by default and also returned as the response.

    Example:
        curl -X POST http://localhost:19004/v1/subtitles \
            -F "file=@/home/me/example.mkv" \
            -F "movie_language=en" \
            -F "srt_language=en" \
            -o example.en.srt
    """
    try:
        result = await create_subtitles_from_upload(
            file=file,
            movie_language=movie_language,
            srt_language=srt_language,
        )
        return FileResponse(
            result["srt_path"],
            media_type="application/x-subrip",
            filename=Path(result["srt_path"]).name,
            headers={
                "X-SRT-Path": result["srt_path"],
                "X-Movie-Language": result["movie_language"],
                "X-SRT-Language": result["srt_language"],
            },
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except subprocess.TimeoutExpired:
        log.error("Subtitle generation timed out")
        return JSONResponse({"error": "Subtitle generation timed out"}, status_code=504)
    except subprocess.CalledProcessError as e:
        log.error(f"ffmpeg failed: {e.stderr}")
        return JSONResponse(
            {"error": "ffmpeg failed while extracting audio", "details": e.stderr[-1000:]},
            status_code=400,
        )
    except Exception as e:
        log.error(f"Subtitle generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)

# ──────────────────────────────────────────────
# WILLOW ENDPOINT — compatible with WIS /api/willow
# ──────────────────────────────────────────────

@app.post("/api/willow")
async def willow_stt(request: Request):
    """
    Willow-compatible STT endpoint.

    Accepts the same request format as WIS: raw PCM/WAV audio in the body
    with audio format described in x-audio-* headers (set by ESP32 firmware).

    Returns WIS-compatible JSON: {"language": "en", "text": "..."}
    """
    # Read the audio body (chunked transfer from ESP32, typically ~64KB)
    body = await request.body()
    if not body:
        return JSONResponse({"language": "en", "text": ""}, status_code=400)

    # Parse audio format from ESP32 headers
    try:
        sample_rate = int(request.headers.get("x-audio-sample-rate", "16000"))
        bits = int(request.headers.get("x-audio-bits", "16"))
        channels = int(request.headers.get("x-audio-channel", "1"))
    except ValueError:
        log.warning("Malformed audio headers — using defaults (16kHz, 16-bit, mono)")
        sample_rate, bits, channels = 16000, 16, 1

    codec = request.headers.get("x-audio-codec", "pcm").lower()

    # Convert PCM to WAV if needed (whisper.cpp expects WAV)
    if codec == "pcm":
        wav_bytes = pcm_to_wav(body, sample_rate, bits, channels)
    elif codec == "wav":
        wav_bytes = body
    else:
        # Unknown codec — can't handle it
        log.warning(f"Unsupported codec '{codec}'")
        return JSONResponse(
            {"language": "en", "text": "", "error": f"Unsupported codec: {codec}"},
            status_code=400,
        )

    dur = audio_duration_ms(len(body), sample_rate, bits, channels)

    try:
        result = await transcribe(wav_bytes, dur)
        # Return WIS-compatible response format
        return JSONResponse({
            "language": result["language"],
            "text": result["text"],
            "infer_time": result["infer_time_ms"],
            "infer_speedup": result["speedup"],
            "audio_duration": result["duration_ms"],
        })
    except Exception as e:
        log.error(f"Transcription failed: {e}")
        return JSONResponse(
            {"language": "en", "text": ""},
            status_code=502,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SRT subtitles with the configured whisper.cpp server."
    )
    parser.add_argument("path", help="Path to the movie/audio file")
    parser.add_argument(
        "--movie-language",
        default="en",
        help="Language spoken in the movie (default: en)",
    )
    parser.add_argument(
        "--srt-language",
        default="en",
        help="Language for the SRT output (default: en; Whisper can only translate to English)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output SRT path (default: same path as movie with .srt extension)",
    )
    return parser.parse_args()


async def cli_main() -> None:
    args = parse_args()
    result = await create_subtitles(
        media_path=args.path,
        movie_language=args.movie_language,
        srt_language=args.srt_language,
        output_path=args.output,
    )
    print(result["srt_path"])


if __name__ == "__main__":
    asyncio.run(cli_main())
