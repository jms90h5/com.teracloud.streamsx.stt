# Complete Setup Guide for STT Toolkit

This guide provides comprehensive instructions for setting up the TeraCloud Streams Speech-to-Text toolkit from scratch.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Model Installation](#model-installation)
4. [Verification](#verification)
5. [Common Issues](#common-issues)
6. [Manual Steps](#manual-steps)

## Prerequisites

### System Requirements
- **OS**: Rocky Linux 9, RHEL 9, or Ubuntu 20.04+
- **RAM**: 8GB minimum (16GB recommended for model export)
- **Disk**: 5GB free space
- **Network**: Internet access for downloading models

### Software Requirements

#### 1. Python 3.11+
```bash
# Check if installed
python3.11 --version

# Install if missing:
# Rocky Linux/RHEL:
sudo dnf install python3.11 python3.11-venv python3.11-pip python3.11-devel

# Ubuntu:
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3.11-distutils
```

#### 2. C++ Compiler (GCC 11+)
```bash
# Check if installed
g++ --version

# Install if missing:
# Rocky Linux/RHEL:
sudo dnf groupinstall "Development Tools"
sudo dnf install gcc-c++

# Ubuntu:
sudo apt install build-essential
```

#### 3. Teracloud Streams
```bash
# Verify installation
echo $STREAMS_INSTALL
streamtool --version

# If not set, source the profile:
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh
```

#### 4. FFmpeg (Optional, for audio processing)
```bash
# Rocky Linux/RHEL 9:
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled crb
sudo rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm
sudo dnf install -y ffmpeg

# Ubuntu:
sudo apt update && sudo apt install ffmpeg
```

## Initial Setup

### Step 1: Clone and Basic Setup
```bash
# Clone the repository
git clone <repository-url>
cd com.teracloud.streamsx.stt

# Run automatic setup
./setup.sh

# Activate development environment
source .envrc
```

The setup script will:
- Check all prerequisites
- Create Python virtual environment at `impl/venv/`
- Build C++ implementation library
- Create convenience aliases
- Run basic verification

### Step 2: Verify Initial Setup
```bash
# Check Python environment
which python
# Should show: .../impl/venv/bin/python

# Check C++ library
ls -la impl/lib/libs2t_impl.so
# Should exist and be ~400KB

# Check toolkit indexing
ls -la toolkit.xml
# Should exist
```

## Model Installation

The toolkit requires the NVIDIA NeMo FastConformer model. This is a one-time setup.

### Step 1: Activate Python Environment
```bash
# ALWAYS do this first before any Python commands
source ./activate_python.sh
```

### Step 2: Install Dependencies

#### Option A: Quick Installation (Recommended)
Install dependencies separately to avoid timeouts:

```bash
# 1. Install PyTorch CPU version (largest dependency, ~800MB)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install ONNX and audio libraries
pip install onnx==1.18.0 onnxruntime==1.22.0
pip install librosa soundfile scipy numpy

# 3. Install NeMo ASR component only
pip install nemo_toolkit[asr]==2.3.1
```

#### Option B: Full Installation
For complete NeMo toolkit (includes TTS, NLP, etc.):
```bash
pip install -r requirements_nemo.txt
```

**Note**: This downloads ~2-3GB and may timeout. Use Option A if you experience issues.

### Step 3: Export the Model

```bash
# Ensure you're in the toolkit root with activated environment
python impl/bin/export_model_ctc_patched.py
```

This will:
1. Download the model from NVIDIA NGC (~460MB)
2. Export to ONNX format (~438MB)
3. Save to `opt/models/fastconformer_ctc_export/`

Expected output:
```
=== Exporting NeMo FastConformer Model in CTC Mode ===
Loading NeMo FastConformer model...
Downloading model from HuggingFace...
[NeMo I] Successfully exported to opt/models/fastconformer_ctc_export/model.onnx
```

### Step 4: Generate Vocabulary (if needed)

If the tokens.txt file is empty or missing:

```bash
# Create the fix_tokens.py script
cat > fix_tokens.py << 'EOF'
#!/usr/bin/env python3
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")
tokenizer = model.tokenizer

with open("opt/models/fastconformer_ctc_export/tokens.txt", 'w', encoding='utf-8') as f:
    for i in range(tokenizer.tokenizer.get_piece_size()):
        piece = tokenizer.tokenizer.id_to_piece(i)
        f.write(f"{piece}\n")

print(f"✅ Wrote {tokenizer.tokenizer.get_piece_size()} tokens")
EOF

# Run it
python fix_tokens.py
```

### Step 5: Verify Model Files

```bash
ls -lh opt/models/fastconformer_ctc_export/
```

Should show:
- `model.onnx` (~438MB)
- `tokens.txt` (~7.7KB with 1024 lines)

## Verification

### Step 1: Run Toolkit Tests
```bash
# Use the convenience alias
stt-test

# Or run directly
./test/verify_nemo_setup.sh
```

### Step 2: Build Toolkit
```bash
# Index the toolkit
spl-make-toolkit -i .

# Verify operators are indexed
grep -c "operator name" toolkit.xml
# Should show: 2 (FileAudioSource and OnnxSTT)
```

### Step 3: Test Sample (Optional)
```bash
cd samples
# Update namespace declarations as needed
make SimpleTest
```

## Common Issues

### Installation Timeouts
**Problem**: `pip install` hangs or times out  
**Solution**: Use Option A (quick installation) which installs packages separately

### Empty Tokens File
**Problem**: `tokens.txt` is 0 bytes  
**Solution**: Run the `fix_tokens.py` script as shown above

### ONNX Header Conflicts
**Problem**: C++ compilation fails with `NO_EXCEPTION` errors  
**Solution**: Already fixed in toolkit - ensure using `#include "onnx_wrapper.hpp"`

### Model Download Fails
**Problem**: Can't download model from HuggingFace  
**Solution**: 
- Check internet connection
- Try using a proxy if behind firewall
- Download manually and place in cache directory

### Python Module Not Found
**Problem**: `ModuleNotFoundError: No module named 'nemo'`  
**Solution**: Always activate the Python environment first:
```bash
source ./activate_python.sh
```

## Manual Steps

If automatic setup fails, here are the manual steps:

### 1. Create Python Environment
```bash
python3.11 -m venv impl/venv
source impl/venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2. Install Python Dependencies
```bash
# Core dependencies
pip install numpy scipy onnx onnxruntime

# PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Audio processing
pip install librosa soundfile

# NeMo
pip install nemo_toolkit[asr]
```

### 3. Build C++ Implementation
```bash
cd impl
make clean
make
cd ..
```

### 4. Index Toolkit
```bash
spl-make-toolkit -i .
```

### 5. Export Model Manually
```python
# Python script to export model
import nemo.collections.asr as nemo_asr
import os

# Create output directory
os.makedirs("opt/models/fastconformer_ctc_export", exist_ok=True)

# Load and export model
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
)
model.set_export_config({"decoder_type": "ctc", "encoder_type": "fastconformer"})
model.export("opt/models/fastconformer_ctc_export/model.onnx")
```

## Next Steps

Once setup is complete:
1. Review the [Architecture Guide](ARCHITECTURE.md)
2. Check the [Operator Guide](OPERATOR_GUIDE.md) 
3. Try the sample applications in `samples/`
4. Build your own STT applications!

## Support

For issues:
1. Check the troubleshooting sections
2. Review `test/verify_nemo_setup.sh` output
3. Check toolkit logs in `output/` directories
4. File an issue with complete error messages