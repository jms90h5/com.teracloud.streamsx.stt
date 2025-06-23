# Detailed Cleanup Plan for com.teracloud.streamsx.stt

Based on systematic analysis of the codebase, this plan removes development artifacts while preserving essential components.

## Files to REMOVE (Development Artifacts)

### Python Development Scripts (16 files)
**Rationale**: One-time setup/conversion scripts, not part of toolkit runtime
```
- check_onnx_model.py
- comprehensive_nemo_fix.py
- convert_extracted_to_onnx.py
- convert_nemo_to_onnx_minimal.py
- decode_transcript.py
- download_nemo_cache_aware_conformer.py
- export_fastconformer_simple.py
- export_nemo_cache_aware_onnx.py
- extract_nemo_model.py
- final_nemo_direct_inference.py
- fix_all_union_syntax.py
- fix_nemo_python39.py
- generate_real_transcript.py
- test_nemo_simple.py
- wav_to_raw.py
- download_nemo_simple.py
```

### Generated Output Files
**Rationale**: Runtime-generated files, should be recreated by tests
```
- transcription_results/ (entire directory)
- ibm_culture_transcript.txt
- get_transcription.sh
```

### Temporary Documentation Files
**Rationale**: Planning documents that served their purpose
```
- TRANSCRIPTION_FIX_PLAN.md
- NEMO_DEBUG_PLAN.md
- README_MODELS.md
- REAL_MODEL_SUCCESS.md
```

### Build Artifacts
**Rationale**: Generated files that should be rebuilt
```
- impl/include/*.o (object files)
- test/test_real_nemo (compiled executable)
- test/test_nemo_improvements (compiled executable)
- Any other compiled binaries in samples/
```

## Files to KEEP (Essential Components)

### Core Toolkit Files
```
- toolkit.xml
- info.xml
- Makefile
- com.teracloud.streamsx.stt/ (SPL namespace directory)
- impl/ (C++ implementation - except .o files)
- deps/ (dependencies)
- models/ (downloaded models)
```

### Essential Documentation
```
- README.md (main documentation)
- ARCHITECTURE.md (system design)
- NEMO_INTEGRATION_STATUS.md (current status)
- NEMO_PYTHON_DEPENDENCIES.md (setup guide)
- NEMO_WORK_PLAN.md (roadmap)
- FIX_NEMO_MODEL.md (known issues and solutions)
```

### Test and Sample Infrastructure
```
- test/ directory (keep for verification)
  - test_real_nemo.cpp (essential for testing)
  - test_nemo_improvements.cpp (feature verification)
  - test_nemo_standalone.cpp (standalone testing)
  - run_nemo_test.sh (test runner)
  - verify_nemo_setup.sh (setup verification)
  - README.md (test documentation)

- samples/ directory (all samples)
  - CppONNX_OnnxSTT/ (working ONNX sample)
  - GenericSTTSample/ (working NeMo sample)
  - UnifiedSTTSample/ (unified interface)

- test_data/ directory (test audio files)
```

### Setup and Configuration
```
- requirements.txt (Python dependencies)
- setup_onnx_runtime.sh (ONNX setup)
- download_sherpa_onnx_model.sh (model download)
```

## Cleanup Execution Order

1. **Remove Python development scripts** (16 files)
2. **Remove generated output directories** (transcription_results/)
3. **Remove temporary documentation** (4 files)
4. **Clean build artifacts** (.o files, compiled binaries)
5. **Update .gitignore** to exclude regenerable files
6. **Fix any merge conflicts** in README.md

## Expected Results After Cleanup

- **File count**: ~160 files (down from 205)
- **Preserved functionality**: All core toolkit features
- **Maintained tests**: All verification capabilities
- **Clean documentation**: Essential docs only
- **Production ready**: No development artifacts

## Verification Steps

After cleanup:
1. **Build test**: `cd impl && make clean && make`
2. **Toolkit test**: `spl-make-toolkit -i . --no-mixed-mode -m`
3. **Sample build**: `cd samples/CppONNX_OnnxSTT && make`
4. **Documentation review**: Ensure all docs are accurate and current

## Backup Strategy

- Current full state backed up as: `com.teracloud.streamsx.stt.cleaned_backup_[timestamp]`
- Original state available in: `com.teracloud.streamsx.stt.checkpoint4.tar.gz`
- Can restore either state if needed