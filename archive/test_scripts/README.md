# Archived Test Scripts

This directory contains test scripts that are no longer actively used but preserved for historical reference.

## Archived Files

### `test_real_nemo.cpp`
- **Created**: June 23, 2025
- **Reason for archival**: Uses outdated model paths (`nemo_fastconformer_streaming/`)
- **Superseded by**: `../../test/test_real_nemo_fixed.cpp`
- **Historical value**: Shows evolution of NeMo integration approach

### `test_nemo_improvements.cpp`  
- **Created**: June 23, 2025
- **Reason for archival**: References missing dependencies and old model structure
- **Historical value**: Documents feature extraction improvements and debugging approaches

### `test_nemo_standalone.cpp`
- **Created**: June 23, 2025
- **Reason for archival**: Superseded by more comprehensive `test_real_nemo_fixed.cpp`
- **Historical value**: Early standalone testing approach

### `test_real_nemo.cpp.broken`
- **Created**: June 23, 2025
- **Reason for archival**: Broken version preserved during debugging
- **Historical value**: Shows debugging process and issue resolution

### `run_nemo_test.sh`
- **Created**: June 23, 2025
- **Reason for archival**: References non-existent binaries and old model paths
- **Historical value**: Shows test automation approach

## Key Changes Since Archival

1. **Model Paths Updated**: From `nemo_fastconformer_streaming/` to `fastconformer_ctc_export/`
2. **Model Files**: From `fastconformer_streaming.onnx` to `model.onnx`
3. **Export Process**: Now uses `export_model_ctc_patched.py` with dependency conflict resolution
4. **Test Structure**: Consolidated into fewer, more comprehensive test programs

## Migration Guide

If you need functionality from these archived scripts:

### For test_real_nemo.cpp → test_real_nemo_fixed.cpp
- Update model path to `models/fastconformer_ctc_export/model.onnx`
- Use the fixed version which has better error handling and command-line options

### For test_nemo_improvements.cpp functionality
- The feature extraction improvements are now integrated into the main implementation
- Use `test_nemo_ctc_simple.cpp` for basic functionality testing

### For run_nemo_test.sh functionality
- Use the verification script: `test/verify_nemo_setup.sh`
- Build and run tests manually following instructions in `test/README.md`

## Development Timeline Context

These scripts were created during the initial NeMo integration phase (June 17-23, 2025) when:
- Model export process was still being stabilized
- Dependency conflicts were being resolved
- Feature extraction implementation was being debugged
- Multiple test approaches were being evaluated

The current active tests in `../test/` represent the final, working solutions that emerged from this development process.