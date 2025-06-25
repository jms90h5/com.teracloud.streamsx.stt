#!/bin/bash

# Quick activation script for STT toolkit Python environment
# Usage: source ./activate_python.sh

if [ -f "impl/venv/bin/activate" ]; then
    echo "Activating STT toolkit Python environment..."
    source impl/venv/bin/activate
    echo "✓ Python $(python --version) activated"
    echo "✓ Location: $(which python)"
    echo ""
    echo "Available Python utilities in impl/bin/:"
    ls impl/bin/*.py 2>/dev/null | head -5 | sed 's|impl/bin/|  - |'
    echo "  - (and others...)"
    echo ""
    echo "To install NeMo: pip install -r doc/requirements_nemo.txt"
else
    echo "❌ Python virtual environment not found."
    echo "Run: ./impl/setup_python_env.sh"
fi