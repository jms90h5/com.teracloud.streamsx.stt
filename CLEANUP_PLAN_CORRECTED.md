# Corrected Cleanup Plan - Build System Safe

**Date**: June 24, 2025  
**Status**: Corrected after build system analysis  
**Critical**: Fixes Makefile issues before cleanup

## 🚨 CRITICAL ISSUES FOUND

Analysis revealed that the cleanup plan would break the build system:

1. **samples/Makefile references missing script**: `export_proven_working_model.py` 
2. **Working export script would be archived**: `export_model_ctc_patched.py` is needed
3. **Build dependencies not properly analyzed**: Must verify all scripts before archival

## Phase 0: Fix Build System Issues (MANDATORY)

### Fix samples/Makefile References
```bash
# Fix missing script references in samples/Makefile
sed -i 's/export_proven_working_model\.py/export_model_ctc_patched.py/g' samples/Makefile
```

### Verify Build Dependencies
```bash
# Test that toolkit builds
cd impl && make clean && make
cd ../samples && make clean && make all
```

## Phase 1: Conservative File Analysis

### Files That MUST STAY (Required for Build/Test/Run)

#### Essential Build Scripts
```bash
# KEEP - Build system dependencies:
export_model_ctc_patched.py                # Working model export (referenced by Makefile)
download_nemo_cache_aware_conformer.py     # Model download utility
setup_onnx_runtime.sh                      # ONNX setup (may be referenced)
download_sherpa_onnx_model.sh              # Model download (may be referenced)
```

#### Essential Test/Utility Scripts
```bash
# KEEP - User utilities:
test_nemo_simple.py                        # Testing utility
wav_to_raw.py                              # Audio conversion utility
download_nemo_simple.py                    # Alternative download method
```

#### Essential Documentation
```bash
# KEEP - User-facing documentation:
README.md                                  # Main entry point
ARCHITECTURE.md                            # System design
OPERATOR_GUIDE.md                          # SPL operator reference
QUICK_START_NEXT_SESSION.md                # Quick start guide
README_MODELS.md                           # Model information

# KEEP - Critical technical guides:
NEMO_EXPORT_SOLUTIONS.md                   # Export solutions
NEMO_PYTHON_DEPENDENCIES.md                # Dependency issues
PYTHON_UPGRADE_2025_06_24.md               # Python 3.11 setup
CRITICAL_FINDINGS_2025_06_23.md            # Working model ID
Claude.md                                  # Development guidelines
```

### Files Safe to Archive (Development History Only)

#### Superseded Export Scripts
```bash
# ARCHIVE - Superseded export attempts:
export_model_ctc.py                        # Original with strict checks
export_model_ctc_py39.py                   # Python 3.9 attempt
export_fastconformer_simple.py             # Early exploration
export_fastconformer_with_preprocessing.py # Preprocessing attempt
export_hybrid_model.py                     # Hybrid export attempt
export_large_fastconformer.py              # Large model attempt
export_large_fixed.py                      # Fixed version attempt
export_nemo_fastconformer_proper.py        # Proper export attempt
export_nemo_model_final.py                 # Final attempt
export_nvidia_fastconformer.py             # NVIDIA export
```

#### Debugging and Analysis Scripts
```bash
# ARCHIVE - Debugging tools:
check_model_io.py                          # Model I/O inspection
check_model_preprocessing.py               # Preprocessing check
check_model_shapes.py                      # Model shape verification
create_working_cmvn.py                     # CMVN creation
extract_cmvn_stats.py                      # CMVN extraction
extract_correct_features.py                # Feature extraction
extract_matching_features.py               # Feature matching
find_feature_params.py                     # Parameter finding
fix_tokens.py                              # Token fixing
inspect_nemo_export.py                     # Export inspection
save_python_features.py                    # Feature saving
save_python_features_debug.py              # Debug feature saving
save_python_logits.py                      # Logits saving
save_working_features.py                   # Working features
validate_features.py                       # Feature validation
verify_audio.py                            # Audio verification
```

#### Historical Documentation
```bash
# ARCHIVE - Development history:
NEMO_INTEGRATION_PLAN.md                   # Planning
NEMO_INTEGRATION_STATUS.md                 # Status tracking
NEMO_WORK_PLAN.md                          # Work planning
NEMO_RESTORATION_PLAN.md                   # Recovery planning
IMPLEMENTATION_COMPLETE.md                 # Status updates
IMPLEMENTATION_SUMMARY.md                  # Implementation summary
STATUS_UPDATE_2025_06_23_FINAL.md          # Final status
WORK_PROGRESS_TRACKING.md                  # Progress tracking
# ... (all other historical docs)
```

## Phase 2: Fix Makefile Issues

### Update samples/Makefile
```bash
# Replace all references to missing export script
cd samples
cp Makefile Makefile.backup

# Fix export script references
sed -i 's/export_proven_working_model\.py/export_model_ctc_patched.py/g' Makefile

# Verify the changes
echo "=== Makefile Changes ==="
diff -u Makefile.backup Makefile || true
```

### Test Makefile Fixes
```bash
# Verify make targets work
cd samples
make help                    # Should show correct script name
make status                  # Should check for correct files
```

## Phase 3: Verify Build System Integrity

### Complete Build Test
```bash
# Test full build process
cd /home/streamsadmin/workspace/teracloud/toolkits/com.teracloud.streamsx.stt

# 1. Clean everything
make clean-all
cd samples && make clean && cd ..

# 2. Build implementation
cd impl && make clean && make && cd ..

# 3. Build toolkit
make build-all

# 4. Build samples
cd samples && make all && cd ..

# 5. Check for working model (or export if needed)
if [ ! -f "models/fastconformer_ctc_export/model.onnx" ]; then
    echo "Exporting model..."
    python export_model_ctc_patched.py
fi

# 6. Test sample status
cd samples && make status && cd ..
```

### Dependency Verification
```bash
# Check all referenced files exist
echo "=== Verifying Build Dependencies ==="

# Scripts referenced by Makefiles
echo "Checking export_model_ctc_patched.py: $([ -f export_model_ctc_patched.py ] && echo '✅' || echo '❌')"

# Libraries referenced by build system
echo "Checking libs2t_impl.so: $([ -f impl/lib/libs2t_impl.so ] && echo '✅' || echo '❌')"
echo "Checking libnemo_stt.so: $([ -f impl/lib/libnemo_stt.so ] && echo '✅' || echo '❌')"

# Models referenced by samples
echo "Checking fastconformer model: $([ -f models/fastconformer_ctc_export/model.onnx ] && echo '✅' || echo '❌')"
echo "Checking model tokens: $([ -f models/fastconformer_ctc_export/tokens.txt ] && echo '✅' || echo '❌')"

# Test data referenced by samples
echo "Checking test audio: $([ -f test_data/audio/librispeech-1995-1837-0001.wav ] && echo '✅' || echo '❌')"
```

## Phase 4: Safe Archival (Only After Verification)

### Create Archive Structure
```bash
mkdir -p archive/dev_scripts/export_attempts
mkdir -p archive/dev_scripts/debugging_tools
mkdir -p archive/docs/historical
mkdir -p archive/test_outputs
mkdir -p archive/build_artifacts
```

### Archive Development Scripts
```bash
# Archive superseded export scripts
git mv export_model_ctc.py archive/dev_scripts/export_attempts/
git mv export_model_ctc_py39.py archive/dev_scripts/export_attempts/
git mv export_fastconformer_simple.py archive/dev_scripts/export_attempts/
git mv export_fastconformer_with_preprocessing.py archive/dev_scripts/export_attempts/
git mv export_hybrid_model.py archive/dev_scripts/export_attempts/
git mv export_large_fastconformer.py archive/dev_scripts/export_attempts/
git mv export_large_fixed.py archive/dev_scripts/export_attempts/
git mv export_nemo_fastconformer_proper.py archive/dev_scripts/export_attempts/
git mv export_nemo_model_final.py archive/dev_scripts/export_attempts/
git mv export_nvidia_fastconformer.py archive/dev_scripts/export_attempts/

# Archive debugging tools
git mv check_model_*.py archive/dev_scripts/debugging_tools/
git mv extract_*.py archive/dev_scripts/debugging_tools/
git mv save_python_*.py archive/dev_scripts/debugging_tools/
git mv validate_features.py archive/dev_scripts/debugging_tools/
git mv verify_audio.py archive/dev_scripts/debugging_tools/
git mv find_feature_params.py archive/dev_scripts/debugging_tools/
git mv fix_tokens.py archive/dev_scripts/debugging_tools/
git mv inspect_nemo_export.py archive/dev_scripts/debugging_tools/
git mv create_working_cmvn.py archive/dev_scripts/debugging_tools/
```

### Archive Historical Documentation
```bash
# Archive development history documents
git mv NEMO_INTEGRATION_PLAN.md archive/docs/historical/
git mv NEMO_INTEGRATION_STATUS.md archive/docs/historical/
git mv NEMO_WORK_PLAN.md archive/docs/historical/
git mv NEMO_RESTORATION_PLAN.md archive/docs/historical/
git mv IMPLEMENTATION_COMPLETE.md archive/docs/historical/
git mv IMPLEMENTATION_SUMMARY.md archive/docs/historical/
git mv STATUS_UPDATE_2025_06_23_FINAL.md archive/docs/historical/
git mv WORK_PROGRESS_TRACKING.md archive/docs/historical/
git mv WORKING_SAMPLE_PATTERN_FINAL.md archive/docs/historical/
git mv STREAMS_SAMPLES_FIX_PLAN.md archive/docs/historical/
git mv STREAMS_SAMPLE_RESULTS.md archive/docs/historical/
git mv STREAMS_SAMPLE_EXECUTION_PLAN.md archive/docs/historical/
git mv SYSTEMATIC_TESTING_PLAN.md archive/docs/historical/
git mv RESTORE_WORKING_SAMPLES_PLAN.md archive/docs/historical/
git mv REAL_MODEL_SUCCESS.md archive/docs/historical/
git mv OPERATOR_UPDATES_COMPLETE.md archive/docs/historical/
git mv MAJOR_BREAKTHROUGH_SUMMARY.md archive/docs/historical/
git mv HONEST_CURRENT_STATE_JUNE_17_2025.md archive/docs/historical/
git mv FIX_SUMMARY_2025_06_23.md archive/docs/historical/
git mv FIX_NEMO_MODEL.md archive/docs/historical/
git mv FEATURE_EXTRACTION_ROOT_CAUSE.md archive/docs/historical/
git mv FASTCONFORMER_TEST_GUIDE.md archive/docs/historical/
git mv FAKE_DATA_REMOVAL_SESSION.md archive/docs/historical/
git mv CURRENT_STATE_TRUTH.md archive/docs/historical/
git mv CLEANUP_PLAN.md archive/docs/historical/
git mv SOLUTION_SUMMARY.md archive/docs/historical/
```

## Phase 5: Post-Cleanup Verification

### Final Build Test
```bash
# Test everything still works after cleanup
cd /home/streamsadmin/workspace/teracloud/toolkits/com.teracloud.streamsx.stt

# 1. Clean and rebuild
make clean-all && make build-all

# 2. Rebuild samples
cd samples && make clean && make all && cd ..

# 3. Test model export (if needed)
python export_model_ctc_patched.py

# 4. Verify samples status
cd samples && make status && cd ..
```

### Verification Checklist
- [ ] `make clean-all && make build-all` succeeds
- [ ] `cd samples && make all` succeeds
- [ ] `python export_model_ctc_patched.py` works
- [ ] `cd samples && make status` shows correct dependencies
- [ ] All referenced scripts exist and are executable
- [ ] All referenced libraries exist and are linkable
- [ ] Documentation links are not broken

## Final Directory Structure

```
com.teracloud.streamsx.stt/
├── README.md                           # Main documentation
├── ARCHITECTURE.md                     # System design
├── OPERATOR_GUIDE.md                   # SPL operator reference
├── QUICK_START_NEXT_SESSION.md         # Quick start guide
├── README_MODELS.md                    # Model information
├── NEMO_EXPORT_SOLUTIONS.md            # Export solutions
├── NEMO_PYTHON_DEPENDENCIES.md         # Dependency guide
├── PYTHON_UPGRADE_2025_06_24.md        # Python 3.11 setup
├── CRITICAL_FINDINGS_2025_06_23.md     # Working model info
├── Claude.md                           # Development guidelines
├── toolkit.xml, info.xml, Makefile     # Core toolkit files
├── requirements.txt                    # Dependencies
│
├── Essential User Scripts:
│   ├── export_model_ctc_patched.py     # Working export (REQUIRED)
│   ├── download_nemo_cache_aware_conformer.py # Model download
│   ├── download_nemo_simple.py         # Alternative download
│   ├── test_nemo_simple.py             # Testing utility
│   ├── wav_to_raw.py                   # Audio conversion
│   ├── setup_onnx_runtime.sh           # ONNX setup
│   └── download_sherpa_onnx_model.sh   # Model download
│
├── Core Directories:
│   ├── com.teracloud.streamsx.stt/     # SPL operators
│   ├── impl/                           # C++ implementation
│   ├── deps/                           # Dependencies
│   ├── models/                         # Model files
│   ├── samples/                        # Sample applications (FIXED)
│   ├── test/                           # Test suite
│   ├── test_data/                      # Test audio files
│   └── docs/                           # Additional documentation
│
└── Archive (Development History):
    └── archive/
        ├── dev_scripts/
        │   ├── export_attempts/        # Superseded export scripts
        │   └── debugging_tools/        # Development utilities
        └── docs/historical/            # Progress and planning docs
```

## Summary of Critical Fixes

1. **Fixed samples/Makefile**: Updated to reference existing `export_model_ctc_patched.py`
2. **Preserved working scripts**: Kept all scripts referenced by build system
3. **Verified dependencies**: Checked all Makefile references exist
4. **Safe archival only**: Only archive files confirmed not needed for build/test/run
5. **Complete verification**: Test full build process before and after cleanup

This corrected plan ensures the toolkit remains fully functional while cleaning up development artifacts safely.