# Detailed Cleanup Plan (Updated)

This plan removes development and test artifacts that are no longer part of the active toolkit.

## 1. Remove One‑Time Python Scripts
The following Python helper scripts in the toolkit root are for one‑time downloads, exports, or debugging
and are not required at runtime or for end users.
```bash
rm -f \
  check_model_io.py \
  check_model_preprocessing.py \
  check_model_shapes.py \
  create_working_cmvn.py \
  download_nemo_cache_aware_conformer.py \
  download_nemo_simple.py \
  export_fastconformer_with_preprocessing.py \
  export_hybrid_model.py \
  export_large_fastconformer.py \
  export_large_fixed.py \
  export_model_ctc_patched.py \
  export_model_ctc.py \
  export_model_ctc_py39.py \
  export_nemo_fastconformer_proper.py \
  export_nemo_model_final.py \
  export_nvidia_fastconformer.py \
  extract_cmvn_stats.py \
  extract_correct_features.py \
  extract_matching_features.py \
  find_feature_params.py \
  fix_tokens.py \
  inspect_nemo_export.py \
  save_python_features_debug.py \
  save_python_features.py \
  save_python_logits.py \
  save_working_features.py \
  validate_features.py \
  verify_audio.py
```

## 2. Remove Compiled Test Binaries
Remove standalone C++ test executables under `test/` that can be rebuilt from source.
```bash
rm -f \
  test/test_nemo_ctc_simple \
  test/test_nemo_standalone \
  test/test_real_nemo \
  test/test_nemo_improvements
```

## 3. Remove Legacy Documentation
Move old planning documents from root into the archive. These are superseded by the active docs in `docs/`.
```bash
rm -f \
  archive/docs/CLEANUP_PLAN.md \
  archive/docs/IMPLEMENTATION_COMPLETE.md \
  archive/docs/IMPLEMENTATION_SUMMARY.md \
  archive/docs/OPERATOR_UPDATES_COMPLETE.md \
  archive/docs/SOLUTION_SUMMARY.md
```

## 4. Clean Build Artifacts
Remove any generated object files and intermediate build outputs.
```bash
rm -rf impl/build impl/lib/*.o
rm -rf **/*.o
rm -f samples/CppONNX_OnnxSTT/test_fastconformer_nemo
```

## 5. Update .gitignore
Ensure regenerable files are ignored:
```
# Python caches
__pycache__/

# Build artifacts
impl/build/
impl/lib/*.o

# Test executables
test/test_nemo_ctc_simple*
test/test_nemo_standalone*
test/test_real_nemo*
test/test_nemo_improvements*

# One-time scripts
*.py
```

After running the above commands, commit the changes to complete cleanup.