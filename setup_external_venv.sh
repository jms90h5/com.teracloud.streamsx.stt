#!/bin/bash
# Setup script using Python venv in home directory
# This avoids SPL compiler issues with files containing spaces

set -e

# Configuration
VENV_NAME="stt_toolkit_env"
VENV_PATH="$HOME/.$VENV_NAME"
PYTHON_VERSION="python3.11"

echo "=== STT Toolkit Setup (External Python Environment) ==="
echo ""

# Check toolkit directory
if [ ! -f "info.xml" ] || [ ! -d "impl" ]; then
    echo "❌ Please run from toolkit root directory"
    exit 1
fi

TOOLKIT_ROOT=$(pwd)
echo "✓ Toolkit root: $TOOLKIT_ROOT"
echo ""

# 1. Check Python
echo "1. Checking Python..."
if ! command -v $PYTHON_VERSION >/dev/null 2>&1; then
    echo "❌ $PYTHON_VERSION not found. Please install it first."
    exit 1
fi
echo "✓ Found $(${PYTHON_VERSION} --version)"
echo ""

# 2. Create/activate virtual environment
echo "2. Setting up Python environment..."
if [ -d "$VENV_PATH" ]; then
    echo "✓ Virtual environment exists at $VENV_PATH"
else
    echo "Creating virtual environment at $VENV_PATH..."
    $PYTHON_VERSION -m venv "$VENV_PATH"
    echo "✓ Created virtual environment"
fi

# Activate it
source "$VENV_PATH/bin/activate"
echo "✓ Activated Python environment"
echo ""

# 3. Install dependencies
echo "3. Installing Python dependencies..."
pip install --upgrade pip --quiet

# Install in stages to avoid timeouts
echo "  Installing core dependencies..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --quiet
pip install onnx==1.18.0 onnxruntime==1.22.0 librosa soundfile scipy numpy --quiet

echo "  Installing NeMo (this may take a few minutes)..."
pip install nemo_toolkit[asr]==2.3.1 --quiet || {
    echo "⚠️  NeMo installation failed. Try manual installation:"
    echo "   source ~/.${VENV_NAME}/bin/activate"
    echo "   pip install nemo_toolkit[asr]"
}
echo "✓ Python dependencies installed"
echo ""

# 4. Create convenience scripts
echo "4. Creating convenience scripts..."

# Create activation script
cat > activate_python.sh << EOF
#!/bin/bash
# Activate the STT toolkit Python environment

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated STT Python environment"
    echo "  Python: \$(which python)"
    echo "  Version: \$(python --version)"
else
    echo "❌ Python environment not found at $VENV_PATH"
    echo "   Run ./setup_external_venv.sh first"
    exit 1
fi
EOF
chmod +x activate_python.sh

# Create .envrc for development
cat > .envrc << EOF
# STT Toolkit Development Environment
# Source this: source .envrc

# Python environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Library paths
export LD_LIBRARY_PATH="\$(pwd)/lib/onnxruntime/lib:\$(pwd)/impl/lib:\$LD_LIBRARY_PATH"

# Convenience aliases
alias stt-build='cd impl && make && cd ..'
alias stt-test='./test/verify_nemo_setup.sh'
alias stt-python='source ./activate_python.sh'
alias stt-export='python impl/bin/export_model_ctc_patched.py'

echo "✓ STT toolkit environment ready"
echo "  Python: $VENV_PATH"
echo "  Commands: stt-build, stt-test, stt-python, stt-export"
EOF

# Create model export script
cat > export_model.sh << EOF
#!/bin/bash
# Export the NeMo model

source ./activate_python.sh || exit 1

echo "Exporting NeMo FastConformer model..."
python impl/bin/export_model_ctc_patched.py || {
    echo "❌ Model export failed"
    exit 1
}

# Check tokens file
if [ ! -s "opt/models/fastconformer_ctc_export/tokens.txt" ]; then
    echo "Generating vocabulary file..."
    python fix_tokens.py || echo "⚠️  Token generation failed"
fi

echo "✓ Model export complete"
ls -lh opt/models/fastconformer_ctc_export/
EOF
chmod +x export_model.sh

echo "✓ Created convenience scripts"
echo ""

# 5. Build C++ implementation
echo "5. Building C++ implementation..."
cd impl
make clean >/dev/null 2>&1
make || {
    echo "❌ C++ build failed"
    exit 1
}
# Create symlink for SPL operators
cd lib
ln -sf libs2t_impl.so libnemo_ctc_impl.so
cd ../..
echo "✓ C++ implementation built"
echo ""

# 6. Summary
echo "=== Setup Complete! ==="
echo ""
echo "Python environment location: $VENV_PATH"
echo ""
echo "Quick start commands:"
echo "  source .envrc           # Activate development environment"
echo "  ./export_model.sh       # Export the NeMo model"
echo "  stt-build              # Rebuild C++ implementation"
echo "  stt-test               # Run verification tests"
echo ""
echo "To use Python scripts directly:"
echo "  source ./activate_python.sh"
echo "  python impl/bin/any_script.py"
echo ""
echo "✅ No SPL compilation issues - Python environment is outside toolkit!"
echo ""
echo "Next steps:"
echo "1. Export a model: ./export_model.sh"
echo "2. Build samples: cd samples && make"
echo ""