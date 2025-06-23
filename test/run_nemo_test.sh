#!/bin/bash
# Run NeMo test with proper library paths

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set library paths
export LD_LIBRARY_PATH="${SCRIPT_DIR}/deps/onnxruntime/lib:${SCRIPT_DIR}/impl/lib:${LD_LIBRARY_PATH}"

# Default values
NEMO_MODEL="${SCRIPT_DIR}/models/nemo_fastconformer_streaming/fastconformer_streaming.onnx"
AUDIO_FILE="${SCRIPT_DIR}/test_data/audio/librispeech-1995-1837-0001.raw"
TEST_BINARY="${SCRIPT_DIR}/samples/CppONNX_OnnxSTT/test_nemo_standalone"

# Check if test binary exists
if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: Test binary not found at $TEST_BINARY"
    echo "Build it with: cd samples/CppONNX_OnnxSTT && make -f Makefile.nemo"
    exit 1
fi

# If no arguments, show usage
if [ $# -eq 0 ]; then
    echo "=== NeMo Test Runner ==="
    echo "This script runs the NeMo test with proper library paths."
    echo
    echo "Usage: $0 [--nemo-model PATH] [--audio-file PATH] [OPTIONS]"
    echo
    echo "Quick test (if model and audio exist):"
    echo "  $0 --verbose"
    echo
    echo "Full options:"
    "$TEST_BINARY" --help
    echo
    
    # Check if default files exist
    if [ -f "$NEMO_MODEL" ] && [ -f "$AUDIO_FILE" ]; then
        echo "Default files found. Running test..."
        echo "Model: $NEMO_MODEL"
        echo "Audio: $AUDIO_FILE"
        echo
        exec "$TEST_BINARY" --nemo-model "$NEMO_MODEL" --audio-file "$AUDIO_FILE" "$@"
    else
        echo "Note: Default model or audio file not found."
        [ ! -f "$NEMO_MODEL" ] && echo "  Model missing: $NEMO_MODEL"
        [ ! -f "$AUDIO_FILE" ] && echo "  Audio missing: $AUDIO_FILE"
    fi
else
    # Pass all arguments to the test binary
    exec "$TEST_BINARY" "$@"
fi