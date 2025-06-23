# NeMo Model Integration: Python Dependency Challenge and Solutions

## Overview

This document explains why integrating NVIDIA NeMo models into the TeraCloud Streams STT toolkit is challenging from a Python dependency perspective and documents the solution approaches.

## The Dependency Conflict Problem

### What We're Trying to Do
- Export a real NVIDIA NeMo model (`stt_en_conformer_ctc_small.nemo`) from PyTorch format to ONNX format
- Use the resulting ONNX model in our C++ STT pipeline

### The Circular Dependency Issue

```
NVIDIA NeMo Toolkit
    ↓ requires
huggingface_hub with ModelFilter class
    ↓ but
ModelFilter was removed in huggingface_hub >= 0.20.0
    ↓ so we need
huggingface_hub < 0.20.0 (e.g., 0.19.4)
    ↓ but
transformers >= 4.50.0 requires huggingface_hub >= 0.30.0 for list_repo_tree
    ↓ and
pytorch-lightning → torchmetrics → transformers (automatic dependency)
    ↓ result:
IMPOSSIBLE TO SATISFY ALL CONSTRAINTS
```

### Detailed Technical Analysis

1. **NeMo 1.23.0** and **NeMo 1.22.0** both expect:
   ```python
   from huggingface_hub import ModelFilter  # Removed in 0.20.0+
   ```

2. **huggingface_hub 0.19.4** (last version with ModelFilter):
   - ✅ Has `ModelFilter` class (needed by NeMo)
   - ❌ Missing `list_repo_tree` function (needed by transformers)

3. **huggingface_hub 0.32.2** (current version):
   - ❌ Missing `ModelFilter` class (breaks NeMo import)
   - ✅ Has `list_repo_tree` function (needed by transformers)

4. **transformers 4.52.3** requires:
   ```python
   from huggingface_hub import list_repo_tree  # Added in 0.30.0+
   ```

5. **Dependencies install transformers automatically**:
   ```
   pytorch-lightning → torchmetrics → transformers → huggingface_hub>=0.30.0
   ```

### Error Manifestations

When attempting export with incompatible versions:

**With huggingface_hub 0.19.4:**
```
ImportError: cannot import name 'list_repo_tree' from 'huggingface_hub'
```

**With huggingface_hub 0.32.2:**
```
ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'
```

## Why This Isn't a Simple Version Pinning Problem

### Attempted Solutions That Failed

1. **Pin huggingface_hub to 0.19.4**: Breaks transformers dependency chain
2. **Downgrade NeMo to 1.18.0**: Still has ModelFilter dependency  
3. **Downgrade transformers**: Breaks pytorch-lightning compatibility
4. **Remove pytorch-lightning**: Breaks NeMo core functionality

### The Real Issue: Architecture Changes

- `ModelFilter` was an architectural component in huggingface_hub's model discovery system
- It was completely removed and replaced with new filtering mechanisms
- This isn't a "bug fix" - it's an intentional API breaking change
- NeMo hasn't updated to the new huggingface_hub API yet

## Solution Approaches

### Approach 1: Complete Environment Isolation ⚠️ 
**Status: High complexity, uncertain success**

```bash
# Create completely isolated environment with ancient versions
conda create -n nemo_ancient python=3.8
conda activate nemo_ancient
pip install torch==1.13.1 
pip install "transformers==4.21.0"      # 1+ years old
pip install "huggingface_hub==0.10.0"   # 2+ years old  
pip install "pytorch-lightning==1.8.0"  # Very old
pip install nemo_toolkit==1.18.0        # Much older NeMo
```

**Risks:**
- May have security vulnerabilities in old packages
- Other compatibility issues may emerge
- Unpredictable behavior with old PyTorch/CUDA

### Approach 2: Docker with Known Working Environment ⚠️
**Status: Container availability uncertain**

```bash
# Try various NVIDIA containers until one works
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:23.11 python /workspace/export_real_nemo.py
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:23.08 python /workspace/export_real_nemo.py
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:23.06 python /workspace/export_real_nemo.py
```

**Risks:**
- May not have access to required NVIDIA containers
- Container may not have required models/tools
- Security/networking restrictions

### Approach 3: Manual PyTorch Model Export ✅ 
**Status: More work but reliable**

**Why this is NOT a lazy shortcut:**
- Requires deep understanding of NeMo model architecture
- Must manually parse .nemo archive format  
- Must reconstruct Conformer encoder from config
- Must handle CTC decoder properly
- Must map NeMo parameter names to standard PyTorch
- Must create proper ONNX export with correct dynamic axes

**Implementation steps:**
1. Extract .nemo file (it's a gzipped tar archive)
2. Parse `model_config.yaml` for architecture details
3. Load `model_weights.ckpt` PyTorch checkpoint
4. Reconstruct model architecture matching the config:
   - Conformer encoder with 16 layers, 176 hidden units, 4 attention heads
   - CTC decoder with 128 output classes
   - Proper subsampling (factor 4) and convolution layers
5. Map checkpoint parameter names to reconstructed model
6. Export to ONNX with proper input/output specifications

### Approach 4: Alternative Model Sources 🎯
**Status: Potentially faster**

Find already-exported ONNX models:
- Search Hugging Face for "conformer onnx" or "nemo onnx"
- Check NVIDIA Model Zoo for ONNX versions
- Use different streaming models (e.g., Whisper ONNX, Wav2Vec2 ONNX)

## Recommended Implementation Strategy

### Phase 1: Try Dependency Resolution (15 minutes max)
Try the nuclear option for environment cleanup:

```bash
# Complete cleanup
pip3 uninstall -y $(pip3 list | grep -E "(nemo|torch|transformers|huggingface)" | awk '{print $1}')

# Install in very specific order with exact versions
pip3 install --user torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install --user "transformers==4.21.0" 
pip3 install --user "huggingface_hub==0.10.0"
pip3 install --user "pytorch-lightning==1.8.0"
pip3 install --user "nemo_toolkit==1.18.0"
```

### Phase 2: If Phase 1 Fails, Manual Export
Implement the manual PyTorch checkpoint loading and ONNX export.

### Phase 3: Validation and Documentation
Create reproducible scripts and validate the exported model works with our C++ implementation.

## Expected Deliverables

1. **Working ONNX Model**: `models/nemo_fastconformer_streaming/conformer_ctc.onnx`
2. **Export Script**: `export_nemo_model.py` (whatever approach works)
3. **Dependencies Documentation**: This file + requirements.txt 
4. **Validation Script**: Test the exported model with real audio
5. **Model Sharing**: Either include ONNX model in repo or provide separate download

## Why This Documentation Matters

1. **Future Maintainers**: Won't waste time on the same dependency issues
2. **Users**: Can understand why we provide pre-exported models vs asking them to export
3. **Alternative Approaches**: Documents what was tried and why it failed
4. **Technical Learning**: Shows real-world Python dependency management challenges

## Model Architecture Details (for manual export)

Based on `model_config.yaml` analysis:

```yaml
Conformer Encoder:
  - Input: 80-dim mel features
  - Layers: 16
  - Model dim: 176  
  - Attention heads: 4
  - Subsampling: Striding (factor 4)
  - Conv kernel: 31

CTC Decoder:
  - Input: 176-dim (from encoder)
  - Output: 128 classes
  - Vocabulary: WordPiece with 128 tokens

Audio Preprocessing:
  - Sample rate: 16kHz
  - Window: 25ms (Hann window)
  - Stride: 10ms  
  - Features: 80 mel bins
  - FFT: 512 points
```

This provides everything needed for manual model reconstruction if the dependency approach fails.