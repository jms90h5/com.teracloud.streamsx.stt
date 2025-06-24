# Python 3.11 Upgrade Documentation - June 24, 2025

## Summary

This document records the Python upgrade from 3.9 to 3.11 performed to support the NeMo model export process in the Teracloud Streams STT toolkit.

## Issue Encountered

The `export_model_ctc.py` script failed with the following error on Python 3.9:
```
ERROR: Python 3.10 or newer is required to run this export script.
Please invoke with python3.10 (not python3.9).
```

This requirement exists because:
1. The script checks for Python 3.10+ due to use of PEP 604 union syntax (`str | List[str]`)
2. The NeMo library itself contains Python 3.10+ syntax in its code

## Upgrade Process

### 1. System Information
- OS: Rocky Linux 9.6 (Blue Onyx)
- Original Python: 3.9.21
- Target Python: 3.11.11

### 2. Installation Steps

```bash
# Install Python 3.11 from Rocky Linux repositories
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# Verify installation
python3.11 --version  # Output: Python 3.11.11
```

### 3. Virtual Environment Setup

```bash
# Create new virtual environment with Python 3.11
cd /home/streamsadmin/workspace/teracloud/toolkits/com.teracloud.streamsx.stt
python3.11 -m venv venv_nemo_py311

# Activate the environment
source venv_nemo_py311/bin/activate
```

### 4. Dependencies Installation

```bash
# Upgrade pip and setuptools
pip install --upgrade pip wheel setuptools

# Install huggingface_hub 0.19.4 (required by NeMo)
pip install huggingface_hub==0.19.4

# Install PyTorch CPU version (to avoid memory issues)
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install Cython (required by some dependencies)
pip install Cython

# Install core NeMo toolkit
pip install nemo_toolkit==1.22.0

# Install additional required dependencies
pip install "hydra-core<=1.3.2"
pip install "pytorch-lightning<=2.0.7" "torchmetrics>=0.11.0" "transformers>=4.36.0"
```

### 5. Dependency Conflict

After installing all dependencies, we encountered the circular dependency issue documented in `NEMO_PYTHON_DEPENDENCIES.md`:
- NeMo 1.22.0 requires `huggingface_hub` < 0.20.0 (for ModelFilter class)
- transformers >= 4.36.0 requires `huggingface_hub` >= 0.30.0 (for list_repo_tree function)
- Installing transformers automatically upgrades huggingface_hub, breaking NeMo

## Current Status

1. **Python 3.11 Installed**: Successfully installed and available at `/usr/bin/python3.11`
2. **Virtual Environment Created**: `venv_nemo_py311` with Python 3.11
3. **Dependency Conflict Persists**: The fundamental incompatibility between NeMo and transformers versions remains

## Model Backups

Before attempting further fixes, existing models were backed up:
```bash
mkdir -p models/backup_2025_06_24
cp -r models/fastconformer_ctc_export models/backup_2025_06_24/
cp -r models/fastconformer_nemo_export models/backup_2025_06_24/
```

## Available Pre-Exported Models

The following models are already available and can be used without re-exporting:
1. `models/fastconformer_ctc_export/model.onnx` (45.7MB)
2. `models/fastconformer_nemo_export/ctc_model.onnx` (471MB) - **This is the working model per CRITICAL_FINDINGS_2025_06_23.md**

## Recommendations

1. **Use Existing Models**: The pre-exported models are functional and tested
2. **Alternative Export Methods**: Consider the approaches documented in `NEMO_PYTHON_DEPENDENCIES.md`:
   - Docker containers with pre-configured environments
   - Manual PyTorch model reconstruction
   - Alternative model sources
3. **Future Work**: Monitor NeMo and huggingface_hub releases for compatibility fixes

## Environment Details

### venv_nemo_py311 Package Versions
- Python: 3.11.11
- torch: 2.0.1+cpu
- nemo_toolkit: 1.22.0
- huggingface_hub: 0.33.0 (automatically upgraded by transformers)
- transformers: 4.52.4
- pytorch-lightning: 2.0.7
- hydra-core: 1.3.2
- omegaconf: 2.3.0

## Next Steps

To proceed with model export, consider:
1. Using a Docker container with a known-working NeMo environment
2. Implementing the manual export approach detailed in NEMO_PYTHON_DEPENDENCIES.md
3. Creating a custom patch to make NeMo work with newer huggingface_hub versions
4. Using the existing pre-exported models for immediate needs