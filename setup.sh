#!/bin/bash
set -e

echo "=== CosyVoice Hokkien TTS Setup ==="

# ── 1. CosyVoice repo ──────────────────────────────────────────────────────────
echo "[1/4] Cloning CosyVoice..."
mkdir -p apps/synthesis/repositories
git clone https://github.com/FunAudioLLM/CosyVoice apps/synthesis/repositories/CosyVoice
cd apps/synthesis/repositories/CosyVoice
git submodule update --init --recursive
cd ../../../..

# ── 2. 安裝 Python 依賴 ────────────────────────────────────────────────────────
echo "[2/4] Installing CosyVoice requirements..."
# setuptools<71 still bundles pkg_resources; needed by openai-whisper==20231117
python -m pip install "setuptools<71" wheel
python -m pip install --no-build-isolation --no-deps openai-whisper==20231117
python -m pip install --timeout 300 -r apps/synthesis/repositories/CosyVoice/requirements.txt

echo "Installing synthesis script requirements..."
python -m pip install --timeout 300 soundfile datasets huggingface_hub pandas pyarrow

# ── 3. 下載模型權重 ────────────────────────────────────────────────────────────
echo "[3/4] Downloading Fun-CosyVoice3-0.5B model weights..."
mkdir -p apps/synthesis/pretrained_models
git clone https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B \
    apps/synthesis/pretrained_models/Fun-CosyVoice3-0.5B

# ── 4. 安裝 vLLM ───────────────────────────────────────────────────────────────
echo "[4/4] Installing vLLM..."
python -m pip install --timeout 300 vllm==0.9.0 transformers==4.51.3 numpy==1.26.4

echo ""
echo "=== Setup complete! ==="
echo "Run: python synthesize_audio.py --src-dir ./tw-hokkien-seed-text"
