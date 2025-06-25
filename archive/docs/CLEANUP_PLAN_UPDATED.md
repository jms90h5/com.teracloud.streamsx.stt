# Updated Cleanup Plan for com.teracloud.streamsx.stt

**Date**: June 24, 2025  
**Principle**: Be conservative - keep anything potentially useful for toolkit users

## Files to KEEP (Essential & User-Facing)

### Core Toolkit Files ✅
```
- toolkit.xml
- info.xml
- Makefile
- com.teracloud.streamsx.stt/ (SPL namespace directory)
- impl/ (C++ implementation - except .o files)
- deps/ (dependencies)
- models/ (downloaded models)
```

### Essential User Scripts ✅
**Justification**: These scripts help users set up, export, and test models
```
- export_model_ctc_patched.py         # WORKING export solution with patches
- download_nemo_cache_aware_conformer.py  # Downloads required NeMo model
- download_nemo_simple.py             # Alternative download method
- test_nemo_simple.py                 # Simple test for exported models
- wav_to_raw.py                       # Audio format conversion utility
- setup_onnx_runtime.sh               # ONNX setup for users
- download_sherpa_onnx_model.sh       # Model download utility
```

### Primary Documentation ✅
**Justification**: Core user documentation and essential guides
```
- README.md                           # Main documentation
- ARCHITECTURE.md                     # System design reference
- OPERATOR_GUIDE.md                   # SPL operator usage
- QUICK_START_NEXT_SESSION.md         # Quick start guide
- README_MODELS.md                    # Model information
```

### Critical Technical Documentation ✅
**Justification**: Documents known issues and solutions users need
```
- NEMO_EXPORT_SOLUTIONS.md            # Comprehensive export solution guide
- NEMO_PYTHON_DEPENDENCIES.md         # Dependency conflict explanation
- PYTHON_UPGRADE_2025_06_24.md        # Python 3.11 upgrade guide
- CRITICAL_FINDINGS_2025_06_23.md     # Working model identification
- Claude.md                           # Development guidelines
```

### Test Infrastructure ✅
**Justification**: Users need to verify their installation
```
- test/ directory (all test files)
  - test_real_nemo.cpp
  - test_nemo_improvements.cpp
  - test_nemo_standalone.cpp
  - run_nemo_test.sh
  - verify_nemo_setup.sh
  - README.md
- test_data/ directory (test audio files)
```

### Sample Applications ✅
**Justification**: Examples users need to understand toolkit usage
```
- samples/ directory (all samples)
  - All working samples demonstrating different features
  - Sample-specific documentation
```

### Configuration Files ✅
```
- requirements.txt                    # Python dependencies
- .gitignore                         # Version control settings
```

## Files to ARCHIVE (Development Artifacts)

### Python Development Scripts 📦
**Justification**: One-time debugging/exploration scripts not needed by users
```
Archive to: archive/dev_scripts/
- check_onnx_model.py                # Debugging tool
- comprehensive_nemo_fix.py          # Development attempt
- convert_extracted_to_onnx.py       # One-time conversion
- convert_nemo_to_onnx_minimal.py    # Superseded by export_model_ctc_patched.py
- decode_transcript.py               # Debugging utility
- export_fastconformer_simple.py     # Early export attempt
- export_nemo_cache_aware_onnx.py    # Superseded version
- extract_nemo_model.py              # One-time extraction
- final_nemo_direct_inference.py     # Testing script
- fix_all_union_syntax.py            # One-time fix
- fix_nemo_python39.py               # Superseded by Python 3.11
- generate_real_transcript.py        # Test data generation
- export_model_ctc.py                # Superseded by patched version
- export_model_ctc_py39.py           # Python 3.9 attempt (we use 3.11 now)
```

### Historical Documentation 📦
**Justification**: Development history/progress tracking, not user-facing
```
Archive to: archive/docs/
- NEMO_INTEGRATION_PLAN.md           # Historical planning
- NEMO_INTEGRATION_STATUS.md         # Superseded by current docs
- NEMO_WORK_PLAN.md                  # Historical planning
- NEMO_RESTORATION_PLAN.md           # Historical recovery plan
- NEMO_CACHE_AWARE_IMPLEMENTATION.md # Implementation details
- IMPLEMENTATION_COMPLETE.md         # Status update
- IMPLEMENTATION_SUMMARY.md          # Status update
- SOLUTION_SUMMARY.md                # Superseded by NEMO_EXPORT_SOLUTIONS.md
- STATUS_UPDATE_2025_06_23_FINAL.md  # Historical status
- WORK_PROGRESS_TRACKING.md          # Development tracking
- WORKING_SAMPLE_PATTERN_FINAL.md    # Development notes
- STREAMS_SAMPLES_FIX_PLAN.md        # Historical fix plan
- STREAMS_SAMPLE_RESULTS.md          # Test results
- STREAMS_SAMPLE_EXECUTION_PLAN.md   # Historical plan
- SYSTEMATIC_TESTING_PLAN.md         # Development testing
- RESTORE_WORKING_SAMPLES_PLAN.md    # Historical recovery
- REAL_MODEL_SUCCESS.md              # Superseded by current docs
- OPERATOR_UPDATES_COMPLETE.md       # Status update
- MAJOR_BREAKTHROUGH_SUMMARY.md      # Historical discovery
- HONEST_CURRENT_STATE_JUNE_17_2025.md # Historical status
- FIX_SUMMARY_2025_06_23.md          # Historical fix summary
- FIX_NEMO_MODEL.md                  # Superseded by solutions doc
- FEATURE_EXTRACTION_ROOT_CAUSE.md   # Debugging notes
- FASTCONFORMER_TEST_GUIDE.md        # Superseded by current guides
- FAKE_DATA_REMOVAL_SESSION.md       # Development cleanup
- CURRENT_STATE_TRUTH.md             # Historical status
- CLEANUP_PLAN.md                    # Original cleanup plan
```

### Generated/Runtime Files 🗑️
**Justification**: These are created at runtime and shouldn't be in version control
```
Remove completely:
- transcription_results/ (entire directory)
- *.o files (build artifacts)
- compiled binaries in test/ and samples/
- ibm_culture_transcript.txt
- get_transcription.sh
- venv_nemo/ (virtual environment)
- venv_nemo_py311/ (virtual environment)
```

## Documentation Consolidation Plan

### Create Unified Documentation Structure:
1. **README.md** - Main entry point with links to other docs
2. **ARCHITECTURE.md** - Technical architecture (keep as-is)
3. **OPERATOR_GUIDE.md** - SPL operator reference (keep as-is)
4. **SETUP_GUIDE.md** - Combine:
   - NEMO_PYTHON_DEPENDENCIES.md
   - PYTHON_UPGRADE_2025_06_24.md
   - NEMO_EXPORT_SOLUTIONS.md
5. **TROUBLESHOOTING.md** - Extract key issues from:
   - CRITICAL_FINDINGS_2025_06_23.md
   - Various fix documents

## Execution Steps

1. **Create archive directories**:
   ```bash
   mkdir -p archive/dev_scripts
   mkdir -p archive/docs/historical
   ```

2. **Move development scripts** (with git mv to preserve history)

3. **Move historical documentation** (with git mv to preserve history)

4. **Clean build artifacts**:
   ```bash
   find . -name "*.o" -delete
   make clean
   ```

5. **Update .gitignore** to exclude:
   - Virtual environments (venv*)
   - Build artifacts (*.o, *.so)
   - Runtime outputs

6. **Consolidate documentation** (in separate commits)

## Verification After Cleanup

1. **Build test**: `cd impl && make clean && make`
2. **Toolkit test**: `spl-make-toolkit -i . --no-mixed-mode -m`
3. **Sample build**: `cd samples && make all`
4. **Export test**: `python export_model_ctc_patched.py`
5. **Documentation review**: Ensure all user paths work

## Notes

- **Conservative approach**: When in doubt, keep the file
- **User perspective**: Keep anything a user might need
- **Archive vs delete**: Archive development artifacts for history
- **Git history**: Use `git mv` to preserve file history
- **Documentation**: Consolidate overlapping docs in Phase 2