"""
whisper-turbo — Fast STT proxy for whisper.cpp with OpenVINO acceleration.

Routes audio to a running whisper.cpp server (with model kept warm in memory)
for significantly faster speech-to-text on Intel hardware.

Endpoints:
    POST /v1/transcribe   — generic STT (multipart file upload)
    POST /api/willow       — Willow/WIS compatible (raw PCM body + ESP32 headers)
    GET  /health           — health check (proxy + whisper-server status)
"""

import os
import struct
import time
import logging

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

VERSION = "1.0.0"

# ──────────────────────────────────────────────
# CONFIG — all values from .env with sensible defaults
# ──────────────────────────────────────────────

load_dotenv()

WHISPER_SERVER_HOST = os.getenv("WHISPER_SERVER_HOST", "127.0.0.1")
WHISPER_SERVER_PORT = os.getenv("WHISPER_SERVER_PORT", "19003")
WHISPER_SERVER_URL = f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}/inference"
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "15"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

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
