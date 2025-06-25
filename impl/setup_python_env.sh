#!/bin/bash

# Setup script for Python virtual environment used by STT toolkit helper scripts
# This environment is needed for model export and other Python utilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== STT Toolkit Python Environment Setup ==="
echo "Setting up Python virtual environment for model export scripts..."
echo ""

# Check if venv exists
if [ -d "$VENV_DIR" ]; then
    echo "✓ Virtual environment already exists at: $VENV_DIR"
    echo "To reinstall, delete the directory first: rm -rf $VENV_DIR"
    echo ""
else
    echo "Creating fresh Python 3.11 virtual environment..."
    python3.11 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate and check
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Python version: $(python --version)"
echo "✓ Python location: $(which python)"
echo ""

# Install basic packages
echo "Installing basic packages..."
pip install --upgrade pip > /dev/null
pip install onnx onnxruntime numpy scipy > /dev/null
echo "✓ Basic packages installed (onnx, onnxruntime, numpy, scipy)"
echo ""

# Check if full NeMo is needed
echo "=== NeMo Installation (Optional) ==="
echo "For model export, you need the full NeMo toolkit."
echo "This is a large download (several GB) and takes time to install."
echo ""
echo "To install NeMo toolkit:"
echo "  source impl/venv/bin/activate"
echo "  pip install -r doc/requirements_nemo.txt"
echo ""
echo "Or install just what you need:"
echo "  pip install nemo_toolkit[all] torch librosa soundfile"
echo ""

# Show usage
echo "=== Usage ==="
echo "To use the Python environment:"
echo "  source impl/venv/bin/activate"
echo "  python impl/bin/export_model_ctc_patched.py"
echo "  python impl/bin/check_model_shapes.py"
echo ""
echo "Scripts that need this environment:"
for script in $(ls "$SCRIPT_DIR/bin"/*.py 2>/dev/null | head -5); do
    echo "  - $(basename "$script")"
done
echo "  - (and other Python scripts in impl/bin/)"
echo ""

echo "✓ Python environment setup complete!"
echo "✓ Location: $VENV_DIR"
echo "✓ Activate with: source impl/venv/bin/activate"