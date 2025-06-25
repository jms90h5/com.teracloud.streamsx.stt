#!/bin/bash
# Download pre-exported sherpa-onnx Paraformer model for inference
# No Python or PyTorch dependencies required!

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}/models"
SHERPA_DIR="${MODELS_DIR}/sherpa_onnx_paraformer"

echo "=== Downloading Sherpa-ONNX Zipformer Model ==="
echo "This is a pre-exported ONNX model - no Python/PyTorch needed!"
echo ""

# Create models directory
mkdir -p "${SHERPA_DIR}"
cd "${SHERPA_DIR}"

# Sherpa-ONNX streaming Zipformer model URL
# Using the bilingual Chinese-English model (also supports English)
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"

echo "Downloading model from GitHub releases..."
echo "URL: ${MODEL_URL}"
wget -c "${MODEL_URL}" -O model.tar.bz2

echo ""
echo "Extracting model files..."
tar -xjf model.tar.bz2

# The extracted directory name
MODEL_DIR="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# Create symlinks for easier access
if [ -d "${MODEL_DIR}" ]; then
    echo "Creating symlinks..."
    ln -sf "${MODEL_DIR}/model.onnx" encoder.onnx
    ln -sf "${MODEL_DIR}/tokens.txt" vocab.txt
    
    # Check what files are available
    echo ""
    echo "Model files:"
    ls -la "${MODEL_DIR}/"
    
    # Create a README
    cat > README.txt << EOF
Sherpa-ONNX Paraformer Model
============================

Model: Streaming Zipformer Bilingual (Chinese-English)
Parameters: 42M
WER: 2.4% (clean) / 5.0% (other) on LibriSpeech
Training data: 4,500 hours

Files:
- model.onnx: The main Paraformer model
- tokens.txt: Vocabulary file
- configuration.yaml: Model configuration

Cache tensor: 'cache' (32 frames)

This is a streaming-capable model optimized for real-time ASR.
No Python or PyTorch dependencies required for inference!

Usage in the toolkit:
- Set encoder_onnx_path to: model.onnx
- Set vocab_path to: tokens.txt
EOF
    
    echo ""
    echo "✅ Model downloaded successfully!"
    echo "Location: ${SHERPA_DIR}/${MODEL_DIR}"
else
    echo "❌ Error: Model directory not found after extraction"
    exit 1
fi

echo ""
echo "=== Download Complete ==="
echo ""
echo "To use this model with the toolkit:"
echo "1. Update your config to point to:"
echo "   - encoder_onnx_path: ${SHERPA_DIR}/encoder.onnx"
echo "   - vocab_path: ${SHERPA_DIR}/vocab.txt"
echo "2. This model uses 'cache' tensor with 32 frame chunks"
echo "3. No CTC/decoder needed - Paraformer is an all-in-one model"