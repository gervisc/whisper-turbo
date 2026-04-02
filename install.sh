#!/usr/bin/env bash
#
# whisper-turbo install script
# Builds whisper.cpp with OpenVINO, downloads a model, sets up the proxy.
# Idempotent — safe to run multiple times.
#
# Usage:
#   ./install.sh                         # defaults: small.en model, /opt/whisper.cpp
#   ./install.sh --model base.en         # use a smaller/faster model
#   ./install.sh --whisper-dir ~/whisper.cpp --install-dir ~/whisper-turbo

set -euo pipefail

# ──────────────────────────────────────────────
# DEFAULTS (override with flags)
# ──────────────────────────────────────────────

WHISPER_CPP_DIR="/opt/whisper.cpp"
INSTALL_DIR="/opt/whisper-turbo"
MODEL="small.en"
OPENVINO_DEVICE="CPU"
THREADS=4

# ──────────────────────────────────────────────
# PARSE ARGS
# ──────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --whisper-dir)  WHISPER_CPP_DIR="$2"; shift 2 ;;
        --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --device)       OPENVINO_DEVICE="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --whisper-dir DIR   Where to clone/build whisper.cpp (default: /opt/whisper.cpp)"
            echo "  --install-dir DIR   Where whisper-turbo lives (default: /opt/whisper-turbo)"
            echo "  --model MODEL       Whisper model to download (default: small.en)"
            echo "                      Options: tiny.en, base.en, small.en, medium.en, small, medium"
            echo "  --device DEVICE     OpenVINO device: CPU or GPU (default: CPU)"
            echo "  --threads N         CPU threads for whisper.cpp (default: 4)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

info()  { echo -e "\033[1;32m[+]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }
error() { echo -e "\033[1;31m[x]\033[0m $*"; exit 1; }

check_cmd() {
    command -v "$1" &>/dev/null || return 1
}

# ──────────────────────────────────────────────
# STEP 1: CHECK PREREQUISITES
# ──────────────────────────────────────────────

# Verify sudo access — needed for /opt, systemd, etc.
if ! sudo -v 2>/dev/null; then
    error "This script requires sudo access. Run with a user that has sudo privileges."
fi

info "Checking prerequisites..."

MISSING=()
for cmd in cmake gcc g++ git python3; do
    check_cmd "$cmd" || MISSING+=("$cmd")
done

# Check for python3-venv
python3 -m venv --help &>/dev/null 2>&1 || MISSING+=("python3-venv")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    error "Missing: ${MISSING[*]}
    Install with: sudo apt install -y cmake build-essential git python3 python3-venv"
fi

info "All prerequisites found."

# ──────────────────────────────────────────────
# STEP 2: CREATE PYTHON VENV (needed early for OpenVINO)
# ──────────────────────────────────────────────

VENV_PIP="$INSTALL_DIR/venv/bin/pip"
VENV_PYTHON="$INSTALL_DIR/venv/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
    info "Python venv already exists — skipping."
else
    info "Creating Python virtual environment..."
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown -R "$USER:$USER" "$INSTALL_DIR"
    python3 -m venv "$INSTALL_DIR/venv"
    "$VENV_PIP" install -q -r "$INSTALL_DIR/requirements.txt"
    info "Python dependencies installed."
fi

# ──────────────────────────────────────────────
# STEP 3: INSTALL OPENVINO (into the venv)
# ──────────────────────────────────────────────

info "Checking OpenVINO..."

if "$VENV_PYTHON" -c "import openvino" &>/dev/null 2>&1; then
    OPENVINO_LIBS=$("$VENV_PYTHON" -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))")
    info "OpenVINO found: $OPENVINO_LIBS"
else
    info "Installing OpenVINO into venv..."
    "$VENV_PIP" install -q openvino || error "Failed to install OpenVINO"
    OPENVINO_LIBS=$("$VENV_PYTHON" -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))")
fi

# Verify OpenVINO cmake is available
OPENVINO_CMAKE=$("$VENV_PYTHON" -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'cmake'))")
if [[ ! -f "$OPENVINO_CMAKE/OpenVINOConfig.cmake" ]]; then
    error "OpenVINO cmake config not found at $OPENVINO_CMAKE"
fi

# ──────────────────────────────────────────────
# STEP 3: CLONE AND BUILD WHISPER.CPP
# ──────────────────────────────────────────────

if [[ -x "$WHISPER_CPP_DIR/build/bin/whisper-server" ]]; then
    info "whisper.cpp already built at $WHISPER_CPP_DIR — skipping."
else
    info "Building whisper.cpp with OpenVINO support..."

    if [[ ! -d "$WHISPER_CPP_DIR" ]]; then
        info "Cloning whisper.cpp..."
        sudo mkdir -p "$(dirname "$WHISPER_CPP_DIR")"
        sudo git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_CPP_DIR"
        sudo chown -R "$USER:$USER" "$WHISPER_CPP_DIR"
    fi

    cd "$WHISPER_CPP_DIR"
    cmake -B build -DWHISPER_OPENVINO=ON -DOpenVINO_DIR="$OPENVINO_CMAKE"
    cmake --build build --config Release -j"$(nproc)"

    if [[ ! -x "$WHISPER_CPP_DIR/build/bin/whisper-server" ]]; then
        error "Build failed — whisper-server binary not found"
    fi

    info "whisper.cpp built successfully."
fi

# ──────────────────────────────────────────────
# STEP 4: DOWNLOAD MODEL
# ──────────────────────────────────────────────

MODEL_FILE="ggml-${MODEL}.bin"
MODEL_PATH="$WHISPER_CPP_DIR/models/$MODEL_FILE"

if [[ -f "$MODEL_PATH" ]]; then
    info "Model $MODEL_FILE already exists — skipping download."
else
    info "Downloading model: $MODEL..."
    cd "$WHISPER_CPP_DIR"
    bash models/download-ggml-model.sh "$MODEL"

    if [[ ! -f "$MODEL_PATH" ]]; then
        error "Model download failed — $MODEL_FILE not found"
    fi
    info "Model downloaded: $MODEL_PATH"
fi

# ──────────────────────────────────────────────
# STEP 5: GENERATE OPENVINO ENCODER MODEL
# ──────────────────────────────────────────────

ENCODER_XML="$WHISPER_CPP_DIR/models/ggml-${MODEL}-encoder-openvino.xml"

if [[ -f "$ENCODER_XML" ]]; then
    info "OpenVINO encoder model already exists — skipping."
else
    info "Generating OpenVINO encoder model..."
    info "This requires optimum[openvino] and may take a few minutes on first run."

    "$VENV_PIP" install -q "optimum[openvino]" 2>/dev/null

    # Map ggml model names to HuggingFace model names
    case "$MODEL" in
        tiny.en)    HF_MODEL="openai/whisper-tiny.en" ;;
        base.en)    HF_MODEL="openai/whisper-base.en" ;;
        small.en)   HF_MODEL="openai/whisper-small.en" ;;
        medium.en)  HF_MODEL="openai/whisper-medium.en" ;;
        tiny)       HF_MODEL="openai/whisper-tiny" ;;
        base)       HF_MODEL="openai/whisper-base" ;;
        small)      HF_MODEL="openai/whisper-small" ;;
        medium)     HF_MODEL="openai/whisper-medium" ;;
        *)          warn "Unknown model '$MODEL' — skipping OpenVINO encoder generation."; ;;
    esac

    if [[ -n "${HF_MODEL:-}" ]]; then
        EXPORT_DIR="$WHISPER_CPP_DIR/models/whisper-${MODEL}-openvino"
        "$VENV_PYTHON" -c "
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
model = OVModelForSpeechSeq2Seq.from_pretrained('$HF_MODEL', export=True)
model.save_pretrained('$EXPORT_DIR')
print('Export complete')
" 2>&1 | tail -3

        # Copy encoder files to where whisper.cpp expects them
        if [[ -f "$EXPORT_DIR/openvino_encoder_model.xml" ]]; then
            cp "$EXPORT_DIR/openvino_encoder_model.xml" "$ENCODER_XML"
            cp "$EXPORT_DIR/openvino_encoder_model.bin" \
               "$WHISPER_CPP_DIR/models/ggml-${MODEL}-encoder-openvino.bin"
            info "OpenVINO encoder model ready."
        else
            warn "OpenVINO encoder export failed — whisper.cpp will run without it (slower)."
        fi
    fi
fi

# ──────────────────────────────────────────────
# STEP 6: GENERATE .env
# ──────────────────────────────────────────────

ENV_FILE="$INSTALL_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    warn ".env already exists — not overwriting. Compare with .env.example for new options."
else
    info "Generating .env..."
    cat > "$ENV_FILE" << ENVEOF
# Generated by install.sh on $(date -Iseconds)
WHISPER_CPP_DIR="$WHISPER_CPP_DIR"
WHISPER_MODEL="$MODEL_FILE"
WHISPER_SERVER_HOST="127.0.0.1"
WHISPER_SERVER_PORT="19003"
PROXY_HOST="127.0.0.1"
PROXY_PORT="19004"
WHISPER_LANGUAGE="en"
WHISPER_THREADS="$THREADS"
OPENVINO_DEVICE="$OPENVINO_DEVICE"
WHISPER_TIMEOUT="15"
LOG_LEVEL="info"
ENVEOF
    info ".env generated at $ENV_FILE"
fi

# ──────────────────────────────────────────────
# STEP 7: INSTALL SYSTEMD SERVICES
# ──────────────────────────────────────────────

info "Installing systemd services..."

# whisper-server reads its config from .env via EnvironmentFile.
# Paths are baked in at install time (binary location, model location, OpenVINO libs)
# but port/threads/language/device come from .env and can be changed without reinstalling.
sudo tee /etc/systemd/system/whisper-server.service > /dev/null << SVCEOF
[Unit]
Description=whisper.cpp STT Server (OpenVINO)
After=network.target

[Service]
EnvironmentFile=$ENV_FILE
Environment="LD_LIBRARY_PATH=$OPENVINO_LIBS"
ExecStart=$WHISPER_CPP_DIR/build/bin/whisper-server \\
    -m $WHISPER_CPP_DIR/models/\${WHISPER_MODEL} \\
    --host \${WHISPER_SERVER_HOST} \\
    --port \${WHISPER_SERVER_PORT} \\
    --language \${WHISPER_LANGUAGE} \\
    -t \${WHISPER_THREADS} \\
    -oved \${OPENVINO_DEVICE}
User=$USER
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

sudo tee /etc/systemd/system/whisper-turbo.service > /dev/null << SVCEOF
[Unit]
Description=whisper-turbo STT Proxy
After=network.target whisper-server.service

[Service]
EnvironmentFile=$ENV_FILE
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/uvicorn whisper_turbo:app \\
    --host \${PROXY_HOST} \\
    --port \${PROXY_PORT} \\
    --log-level \${LOG_LEVEL}
User=$USER
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable whisper-server whisper-turbo

info "Services installed and enabled (not started)."

# ──────────────────────────────────────────────
# STEP 9: SMOKE TEST
# ──────────────────────────────────────────────

info "Running smoke test..."

# Start whisper-server temporarily
sudo systemctl start whisper-server
info "Waiting for whisper-server to load model..."

# Wait for health endpoint
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:19003/health &>/dev/null; then
        break
    fi
    sleep 1
done

if curl -sf http://127.0.0.1:19003/health &>/dev/null; then
    info "whisper-server is healthy."
else
    warn "whisper-server did not respond within 30s — check logs: sudo journalctl -u whisper-server"
fi

# Stop after smoke test — let user start manually
sudo systemctl stop whisper-server

# ──────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────

echo ""
info "============================================"
info "  whisper-turbo installed successfully!"
info "============================================"
echo ""
echo "  Config:    $ENV_FILE"
echo "  Proxy:     $INSTALL_DIR/whisper_turbo.py"
echo "  Models:    $WHISPER_CPP_DIR/models/"
echo ""
echo "  Start services:"
echo "    sudo systemctl start whisper-server"
echo "    sudo systemctl start whisper-turbo"
echo ""
echo "  Test:"
echo "    curl http://127.0.0.1:19004/health"
echo "    curl -X POST http://127.0.0.1:19004/v1/transcribe -F 'file=@audio.wav'"
echo ""
echo "  For Willow/WIS integration, see: willow/WILLOW.md"
echo ""
