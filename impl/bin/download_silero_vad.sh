#!/bin/bash

# Download Silero VAD model for voice activity detection
# Model: https://github.com/snakers4/silero-vad

set -e

MODEL_DIR="models/silero_vad"
MODEL_URL="https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
MODEL_FILE="silero_vad.onnx"

echo "Downloading Silero VAD model..."

# Create model directory
mkdir -p "$MODEL_DIR"

# Download model if it doesn't exist
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Downloading $MODEL_FILE from $MODEL_URL"
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    echo "Downloaded $MODEL_FILE"
else
    echo "$MODEL_FILE already exists, skipping download"
fi

# Verify the model file
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Silero VAD model ready: $MODEL_DIR/$MODEL_FILE"
    echo "Model size: $(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)"
else
    echo "Error: Failed to download Silero VAD model"
    exit 1
fi

echo "Silero VAD setup complete!"