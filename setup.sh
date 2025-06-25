#!/bin/bash

# Automatic setup script for TeraCloud Streams STT Toolkit
# Run this after cloning the repository to set up all dependencies

set -e

echo "=== TeraCloud Streams STT Toolkit Setup ==="
echo "Setting up toolkit for first-time use..."
echo ""

# Check if we're in the right directory
if [ ! -f "info.xml" ] || [ ! -d "impl" ]; then
    echo "❌ Please run this script from the toolkit root directory"
    echo "   Should contain: info.xml, impl/, lib/, etc."
    exit 1
fi

echo "✓ In toolkit root directory"
echo ""

# 1. Check for required system dependencies
echo "1. Checking system dependencies..."

# Check for Python 3.11
if command -v python3.11 >/dev/null 2>&1; then
    echo "✓ Python 3.11 found: $(python3.11 --version)"
else
    echo "❌ Python 3.11 not found. Please install Python 3.11:"
    echo "   # Rocky Linux / RHEL:"
    echo "   sudo dnf install python3.11 python3.11-venv python3.11-pip"
    echo "   # Ubuntu:"
    echo "   sudo apt install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

# Check for Streams environment
if [ -z "$STREAMS_INSTALL" ]; then
    echo "⚠️  STREAMS_INSTALL not set. You may need to source Streams environment:"
    echo "   source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh"
else
    echo "✓ Streams environment: $STREAMS_INSTALL"
fi

# Check for basic build tools
if command -v g++ >/dev/null 2>&1; then
    echo "✓ C++ compiler found: $(g++ --version | head -1)"
else
    echo "❌ g++ not found. Please install build tools:"
    echo "   sudo dnf groupinstall 'Development Tools'"
    exit 1
fi

echo ""

# 2. Set up Python virtual environment
echo "2. Setting up Python environment..."
if [ -d "impl/venv" ]; then
    echo "✓ Python virtual environment already exists"
else
    echo "Creating Python virtual environment..."
    python3.11 -m venv impl/venv
    echo "✓ Virtual environment created"
fi

# Activate and install basic packages
echo "Installing essential Python packages..."
source impl/venv/bin/activate
pip install --upgrade pip --quiet
pip install onnx onnxruntime numpy scipy --quiet
echo "✓ Essential packages installed"
echo ""

# 3. Build C++ implementation
echo "3. Building C++ implementation..."
if [ -f "impl/lib/libs2t_impl.so" ]; then
    echo "✓ C++ library already built"
else
    echo "Building C++ implementation..."
    cd impl
    make clean
    make
    cd ..
    echo "✓ C++ implementation built successfully"
fi
echo ""

# 4. Check for models
echo "4. Checking for models..."
if [ -f "opt/models/fastconformer_ctc_export/model.onnx" ]; then
    MODEL_SIZE=$(stat -c%s opt/models/fastconformer_ctc_export/model.onnx 2>/dev/null | numfmt --to=iec || echo "unknown")
    echo "✓ NeMo CTC model found (${MODEL_SIZE}B)"
else
    echo "⚠️  No model found. You'll need to export a model:"
    echo "   source ./activate_python.sh"
    echo "   pip install -r doc/requirements_nemo.txt  # (large download)"
    echo "   python impl/bin/export_model_ctc_patched.py"
fi
echo ""

# 5. Create convenience scripts
echo "5. Creating convenience aliases..."
cat > .envrc << 'EOF'
# Environment setup for STT toolkit development
# Source this file: source .envrc

# Activate Python environment
source impl/venv/bin/activate

# Set library paths for testing
export LD_LIBRARY_PATH="$(pwd)/lib/onnxruntime/lib:$(pwd)/impl/lib:$LD_LIBRARY_PATH"

# Convenience aliases
alias stt-build='cd impl && make && cd ..'
alias stt-test='./test/verify_nemo_setup.sh'
alias stt-python='source ./activate_python.sh'
alias stt-samples='cd samples && make status && cd ..'

echo "STT toolkit environment activated"
echo "Aliases: stt-build, stt-test, stt-python, stt-samples"
EOF

echo "✓ Created .envrc with convenience aliases"
echo ""

# 6. Final verification
echo "6. Running verification..."
./test/verify_nemo_setup.sh || true
echo ""

# 7. Summary and next steps
echo "=== Setup Complete! ==="
echo ""
echo "✅ Python environment: impl/venv/ (Python $(impl/venv/bin/python --version 2>&1 | cut -d' ' -f2))"
echo "✅ C++ implementation: impl/lib/libs2t_impl.so"
echo "✅ ONNX Runtime: lib/onnxruntime/"
echo ""
echo "🚀 Quick Start:"
echo "   # Activate development environment"
echo "   source .envrc"
echo ""
echo "   # Export a model (if needed)"
echo "   pip install -r doc/requirements_nemo.txt"
echo "   python impl/bin/export_model_ctc_patched.py"
echo ""
echo "   # Build and test samples"
echo "   stt-samples"
echo "   cd samples && make BasicNeMoDemo"
echo ""
echo "📖 Documentation: README.md, doc/"
echo "🔧 Troubleshooting: ./test/verify_nemo_setup.sh"
echo ""
echo "Happy coding! 🎉"