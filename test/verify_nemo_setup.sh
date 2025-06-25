#!/bin/bash
# Verify NeMo setup and dependencies for TeraCloud Streams STT

echo "=== NeMo Setup Verification Script ==="
echo "Checking all dependencies and components..."
echo

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        echo "  ${3}"
        return 1
    fi
}

# Track overall status
OVERALL_STATUS=0

# 1. Check ONNX Runtime
echo "1. Checking ONNX Runtime..."
if [ -f "lib/onnxruntime/lib/libonnxruntime.so" ]; then
    check_status 0 "ONNX Runtime found at lib/onnxruntime/lib/"
else
    check_status 1 "ONNX Runtime not found" "Run: ./setup_onnx_runtime.sh"
    OVERALL_STATUS=1
fi
echo

# 2. Check C++ implementation library
echo "2. Checking C++ implementation..."
if [ -f "impl/lib/libs2t_impl.so" ]; then
    check_status 0 "STT implementation library built"
    # Check if NeMo support is included
    if nm impl/lib/libs2t_impl.so | grep -q "NeMoCacheAwareConformer"; then
        check_status 0 "NeMo support included in library"
    else
        check_status 1 "NeMo support not found in library" "Rebuild with: cd impl && make clean && make"
        OVERALL_STATUS=1
    fi
else
    check_status 1 "STT implementation library not built" "Run: cd impl && make"
    OVERALL_STATUS=1
fi
echo

# 3. Check sample applications
echo "3. Checking sample applications..."
if [ -f "samples/CppONNX_OnnxSTT/test_nemo_standalone" ]; then
    check_status 0 "NeMo standalone test built"
    # Test if it runs
    if LD_LIBRARY_PATH=lib/onnxruntime/lib:impl/lib:$LD_LIBRARY_PATH ./samples/CppONNX_OnnxSTT/test_nemo_standalone --help >/dev/null 2>&1; then
        check_status 0 "NeMo standalone test executable runs"
    else
        check_status 1 "NeMo standalone test fails to run" "Check library paths"
        OVERALL_STATUS=1
    fi
else
    check_status 1 "NeMo standalone test not built" "Run: cd samples/CppONNX_OnnxSTT && make -f Makefile.nemo"
    OVERALL_STATUS=1
fi
echo

# 4. Check Python dependencies for model export
echo "4. Checking Python dependencies..."
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null
check_status $? "Python 3.8+ available"

# Check for NeMo toolkit
python3 -c "import nemo" 2>/dev/null
if [ $? -eq 0 ]; then
    check_status 0 "NeMo toolkit installed"
else
    check_status 1 "NeMo toolkit not installed" "Install with: pip install -r requirements_nemo.txt"
    echo -e "  ${YELLOW}Note: NeMo installation is optional if you have pre-exported ONNX models${NC}"
fi

# Check for ONNX
python3 -c "import onnx" 2>/dev/null
if [ $? -eq 0 ]; then
    check_status 0 "ONNX Python package installed"
else
    check_status 1 "ONNX Python package not installed" "Install with: pip install onnx"
fi
echo

# 5. Check model files
echo "5. Checking model files..."
MODEL_DIR="opt/models/fastconformer_ctc_export"
if [ -d "$MODEL_DIR" ]; then
    check_status 0 "Model directory exists"
    if [ -f "$MODEL_DIR/model.onnx" ]; then
        check_status 0 "NeMo CTC ONNX model found"
        if [ -f "$MODEL_DIR/tokens.txt" ]; then
            check_status 0 "Vocabulary file found"
        else
            check_status 1 "Vocabulary file missing" "Should be at $MODEL_DIR/tokens.txt"
        fi
    else
        check_status 1 "NeMo ONNX model not found" "Export a model using: python export_model_ctc_patched.py"
        echo -e "  ${YELLOW}Or provide your own ONNX model exported from NeMo${NC}"
        OVERALL_STATUS=1
    fi
else
    check_status 1 "Model directory not created" "Run: python export_model_ctc_patched.py"
    OVERALL_STATUS=1
fi
echo

# 6. Check test data
echo "6. Checking test data..."
if [ -f "opt/test_data/audio/librispeech-1995-1837-0001.raw" ]; then
    check_status 0 "Test audio file available"
else
    check_status 1 "Test audio file not found" "Download test data or provide your own 16kHz raw audio"
    echo -e "  ${YELLOW}You can convert any audio file to raw format with:${NC}"
    echo "  ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 output.raw"
fi
echo

# 7. Check kaldifeat (optional)
echo "7. Checking kaldifeat (optional)..."
if [ -f "impl/lib/libkaldifeat.a" ]; then
    check_status 0 "Kaldifeat static library found"
else
    echo -e "${YELLOW}ℹ${NC} Kaldifeat not built (using fallback implementation)"
    echo "  To build kaldifeat: ./impl/setup_kaldifeat.sh"
fi
echo

# Summary
echo "=== Summary ==="
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All required components are ready!${NC}"
    echo
    echo "Next steps:"
    echo "1. Export a NeMo model (if not done):"
    echo "   python3 export_nemo_to_onnx.py"
    echo
    echo "2. Test with standalone application:"
    echo "   ./samples/CppONNX_OnnxSTT/test_nemo_standalone \\"
    echo "     --nemo-model opt/models/nemo_fastconformer_streaming/fastconformer_streaming.onnx \\"
    echo "     --audio-file opt/test_data/audio/librispeech-1995-1837-0001.raw"
    echo
    echo "3. Use in SPL applications"
else
    echo -e "${RED}Some components are missing or need attention.${NC}"
    echo "Please address the issues marked with ✗ above."
fi
echo

# Optional: Show quick start commands
if [ $OVERALL_STATUS -ne 0 ]; then
    echo "=== Quick Setup Commands ==="
    echo "# Install ONNX Runtime"
    echo "./setup_onnx_runtime.sh"
    echo
    echo "# Build C++ implementation"
    echo "cd impl && make"
    echo
    echo "# Build sample application"
    echo "cd samples/CppONNX_OnnxSTT && make -f Makefile.nemo"
    echo
    echo "# (Optional) Install Python dependencies for model export"
    echo "pip install -r requirements_nemo.txt"
    echo
fi

exit $OVERALL_STATUS