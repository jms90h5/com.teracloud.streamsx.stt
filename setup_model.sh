#!/bin/bash
# Automated model setup script for STT Toolkit
# This script handles NeMo installation and model export

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== STT Toolkit Model Setup ==="
echo "This script will install NeMo and export the FastConformer model"
echo

# Check if Python environment exists
if [ ! -d "impl/venv" ]; then
    echo -e "${RED}✗ Python environment not found${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate Python environment
echo "1. Activating Python environment..."
source impl/venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo -e "${RED}✗ Wrong Python version: $PYTHON_VERSION (need 3.11)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION activated${NC}"

# Check if model already exists
if [ -f "opt/models/fastconformer_ctc_export/model.onnx" ] && [ -f "opt/models/fastconformer_ctc_export/tokens.txt" ]; then
    echo -e "${YELLOW}ℹ Model already exported. Delete opt/models/ to re-export.${NC}"
    exit 0
fi

# Check if NeMo is already installed
echo
echo "2. Checking NeMo installation..."
if python -c "import nemo" 2>/dev/null; then
    echo -e "${GREEN}✓ NeMo already installed${NC}"
else
    echo "NeMo not found. Installing dependencies..."
    
    # Install in stages to avoid timeouts
    echo "  Installing PyTorch (this may take a few minutes)..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu || {
        echo -e "${RED}✗ PyTorch installation failed${NC}"
        echo "Try running: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        exit 1
    }
    
    echo "  Installing ONNX and audio libraries..."
    pip install onnx==1.18.0 onnxruntime==1.22.0 librosa soundfile scipy numpy || {
        echo -e "${RED}✗ Dependency installation failed${NC}"
        exit 1
    }
    
    echo "  Installing NeMo ASR (this may take several minutes)..."
    pip install nemo_toolkit[asr]==2.3.1 || {
        echo -e "${RED}✗ NeMo installation failed${NC}"
        echo "Try installing manually: pip install nemo_toolkit[asr]"
        exit 1
    }
    
    echo -e "${GREEN}✓ NeMo installed successfully${NC}"
fi

# Export the model
echo
echo "3. Exporting FastConformer model..."
echo "  This will download ~460MB and create a ~438MB ONNX file"

python impl/bin/export_model_ctc_patched.py || {
    echo -e "${RED}✗ Model export failed${NC}"
    echo "Check the error messages above"
    exit 1
}

# Check if tokens file was generated
if [ ! -s "opt/models/fastconformer_ctc_export/tokens.txt" ]; then
    echo
    echo "4. Generating vocabulary file..."
    echo "  Note: This is a known issue with some NeMo export configurations"
    
    # Use the proven fix_tokens.py approach
    if [ -f "fix_tokens.py" ]; then
        echo "  Using existing fix_tokens.py script..."
        python fix_tokens.py || {
            echo -e "${RED}✗ Token generation failed${NC}"
            exit 1
        }
    else
        # Create tokens generation script inline
        cat > /tmp/generate_tokens.py << 'EOF'
import nemo.collections.asr as nemo_asr
import os

print("Loading model for vocabulary extraction...")
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
)

output_path = "opt/models/fastconformer_ctc_export/tokens.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    tokenizer = model.tokenizer
    for i in range(tokenizer.tokenizer.get_piece_size()):
        piece = tokenizer.tokenizer.id_to_piece(i)
        f.write(f"{piece}\n")

print(f"✓ Generated {tokenizer.tokenizer.get_piece_size()} tokens")
EOF

        python /tmp/generate_tokens.py || {
            echo -e "${RED}✗ Token generation failed${NC}"
            exit 1
        }
        rm /tmp/generate_tokens.py
    fi
fi

# Verify the output
echo
echo "5. Verifying model files..."
if [ -f "opt/models/fastconformer_ctc_export/model.onnx" ]; then
    MODEL_SIZE=$(ls -lh opt/models/fastconformer_ctc_export/model.onnx | awk '{print $5}')
    echo -e "${GREEN}✓ Model exported: opt/models/fastconformer_ctc_export/model.onnx ($MODEL_SIZE)${NC}"
else
    echo -e "${RED}✗ Model file not found${NC}"
    exit 1
fi

if [ -s "opt/models/fastconformer_ctc_export/tokens.txt" ]; then
    TOKEN_COUNT=$(wc -l < opt/models/fastconformer_ctc_export/tokens.txt)
    echo -e "${GREEN}✓ Vocabulary generated: $TOKEN_COUNT tokens${NC}"
else
    echo -e "${RED}✗ Tokens file is empty or missing${NC}"
    exit 1
fi

echo
echo -e "${GREEN}=== Model Setup Complete! ===${NC}"
echo
echo "Model details:"
echo "  - Model: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
echo "  - Architecture: FastConformer-Hybrid CTC"
echo "  - Parameters: 114M"
echo "  - Vocabulary: 1024 SentencePiece tokens"
echo
echo -e "${YELLOW}⚠️  IMPORTANT: Python venv and SPL Compilation${NC}"
echo "  The Python virtual environment in impl/venv may interfere with SPL compilation."
echo "  If you get 'Lorem' file errors when building SPL samples, temporarily move the venv:"
echo "    mv impl/venv /tmp/backup && make && mv /tmp/backup impl/venv"
echo
echo "Next steps:"
echo "  1. Run verification: ./test/verify_nemo_setup.sh"
echo "  2. Test C++ directly: cd test && make test_with_audio && ./test_with_audio"
echo "  3. Build SPL samples: cd samples && make (see venv note above)"
echo "  4. Use model in your applications!"
echo
echo "For troubleshooting, see: doc/KNOWN_ISSUES.md"
echo