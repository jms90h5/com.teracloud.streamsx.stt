# Debug Data Archive

This directory contains development and debugging artifacts that were moved from the toolkit root directory during the reorganization to follow Teracloud Streams standards.

## Directory Structure

### `features/`
Feature extraction debugging files from the development phase:
- `*_features_*.{npy,bin}` - Feature extraction outputs at various stages
- `correct_features_*` - Reference feature data for validation
- `cpp_features_*` - C++ feature extraction outputs
- `python_features_*` - Python feature extraction outputs for comparison

### `model_outputs/`  
Model inference debugging outputs:
- `*_logits*.npy` - Raw model logits for debugging
- Model prediction intermediate results

### `working_files/`
Miscellaneous development artifacts:
- `*.nemo` - Large NeMo model files (450MB+ each)
- `working_*.{bin,npy}` - Intermediate processing files
- `*.json` - Debug configuration and metadata
- `*.cpp` - One-off debug programs
- `global_cmvn.stats` - Feature normalization statistics

### Root Level
- `fastconformer_*` - Model export directories from development
- `extracted_fastconformer/` - Model artifact extraction results  
- `models_backup_*` - Historical model backups
- `venv_*` - Python virtual environments from development

## Purpose

These files were essential during the debugging and development phase to:
1. Compare C++ vs Python feature extraction
2. Validate model inference pipelines
3. Debug feature normalization issues
4. Test different model formats and exports

## Safe to Remove

All files in this archive are safe to delete if disk space is needed, as they are:
- Development artifacts, not production dependencies
- Can be regenerated if needed for future debugging
- Not referenced by any build scripts or SPL operators
- Backed up in the migration_backup/ directory

## Migration Notes

Moved during toolkit reorganization on 2025-06-25 to comply with Teracloud Streams Toolkit Development Guide standards. The toolkit root directory now contains only essential files per official guidelines.