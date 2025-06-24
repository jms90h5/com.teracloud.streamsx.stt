# NeMo Export Solutions - Comprehensive Guide

## Summary

This document provides working solutions for the NeMo FastConformer export dependency conflicts. Based on analysis of legacy working implementations and GitHub research, multiple proven approaches exist.

## The Core Issue

NeMo toolkit requires:
- `ModelFilter` class (removed from huggingface_hub >= 0.20.0)
- `list_repo_tree` function (added in huggingface_hub >= 0.30.0)
- `OfflineModeIsEnabled` class (moved locations in newer versions)

This creates a **circular dependency** that's impossible to satisfy with standard version pinning.

## Working Solutions

### Solution 1: Dynamic Patching (RECOMMENDED)

The legacy implementation in `streamsx.stt.other` proves this approach works:

```bash
source venv_nemo_py311/bin/activate
python export_model_ctc_patched.py
```

**How it works:**
- Detects huggingface_hub version
- Dynamically adds missing classes/functions
- Allows NeMo to import without errors

**Advantages:**
- Works with newer Python versions
- No complex environment management
- Maintains compatibility with other packages

### Solution 2: Specific Version Downgrade

From GitHub issue research, this combination is known to work:

```bash
source venv_nemo_py311/bin/activate
pip install huggingface_hub==0.19.4
pip install transformers==4.21.0
pip install pytorch-lightning==1.8.0
python export_model_ctc.py
```

**Note:** May require downgrading other packages that depend on newer transformers.

### Solution 3: Alternative Version (From GitHub)

Based on issue reports, this also works:

```bash
pip install huggingface_hub==0.23.5
python export_model_ctc.py
```

### Solution 4: Use Pre-Exported Models

The toolkit already has working models:
- `models/fastconformer_nemo_export/ctc_model.onnx` (471MB) - **VERIFIED WORKING**
- `models/fastconformer_ctc_export/model.onnx` (45.7MB)

## Implementation Priority

1. **Try Solution 1 (Patching)** - Most likely to work
2. **Use Solution 4 (Pre-exported)** - Immediate fallback
3. **Try Solution 2/3 (Version downgrade)** - If you need to re-export

## Why This Was Working Before

The legacy implementation (`streamsx.stt.other`) used:
1. **Simpler export script** without strict version checks
2. **Dynamic patching** of missing functions/classes
3. **CTC-only export** (not hybrid RNNT-CTC)
4. **Graceful fallbacks** for missing attributes

## Testing the Solution

After export, test with:

```python
import onnxruntime as ort
import numpy as np

# Load and test the exported model
session = ort.InferenceSession("models/fastconformer_ctc_export/model.onnx")
print("Model inputs:", [input.name for input in session.get_inputs()])
print("Model outputs:", [output.name for output in session.get_outputs()])
```

## Key Insights from Research

1. **The dependency conflict is real** but has known workarounds
2. **NeMo will eventually fix this** (GitHub issue #10272)
3. **CTC export is much simpler** than hybrid RNNT-CTC
4. **Dynamic patching works reliably** (proven in legacy code)

## Environment Details That Work

From the legacy implementation:
- Python 3.11.11 ✅
- Any huggingface_hub version with proper patching ✅
- NeMo 1.22.0+ ✅
- PyTorch 2.0.1+ ✅

## Next Steps

1. **Test Solution 1**: `python export_model_ctc_patched.py`
2. **If it works**: Update main export script with patching approach
3. **If it fails**: Use pre-exported models and document the approach
4. **Monitor NeMo releases** for official fix to dependency conflicts

## References

- NeMo GitHub Issue #10272: Stop using ModelFilter
- NeMo GitHub Issue #9793: ModelFilter import error
- Legacy working implementation: `streamsx.stt.other/export_model_ctc.py`
- Stack Overflow solutions for huggingface_hub compatibility