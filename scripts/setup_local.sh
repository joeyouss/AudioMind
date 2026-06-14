#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/app"
VENV_DIR="$APP_DIR/.venv"
MODEL="${OLLAMA_MODEL:-gemma3:1b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Python 3.10 or 3.11 is required. Install Python first, then rerun this script."
  exit 1
fi

echo "Setting up Python environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "Created app/.env from app/.env.example."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Note: ffmpeg is not installed. WAV uploads still work, but MP3/M4A conversion may need ffmpeg."
  if command -v brew >/dev/null 2>&1; then
    echo "On macOS, install it with: brew install ffmpeg"
  fi
fi

if ! command -v ollama >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Ollama with Homebrew..."
    brew install ollama
  else
    echo "Ollama is not installed. Install it from https://ollama.com/download, then run:"
    echo "  ollama pull $MODEL"
    exit 0
  fi
fi

if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  mkdir -p "$APP_DIR/.cache"
  nohup ollama serve > "$APP_DIR/.cache/ollama.log" 2>&1 &

  for _ in {1..30}; do
    if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama did not start. Check app/.cache/ollama.log, then run: ollama serve"
  exit 1
fi

if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
  echo "Downloading local Gemma model: $MODEL"
  ollama pull "$MODEL"
else
  echo "Gemma model already installed: $MODEL"
fi

cat <<EOF

Local setup is ready.

Next:
1. Add your pyannoteAI key to app/.env for the default Precision-2 mode:
   PYANNOTE_API_KEY=your_pyannoteai_key_here
2. Optional: add your Hugging Face token for local pyannote + Whisper mode:
   HF_TOKEN=hf_your_read_token_here
3. Start the app:
   make run

EOF
