# Claude Assistant Guidelines

## Important Rules

1. **Memory Management**: Do not run commands that use a large amount of heap memory. Ask the user to run memory-intensive commands instead.

2. **Compilation Warnings**: The user does not like warnings. Always fix all compilation warnings before proceeding with other tasks. Treat warnings as errors that must be resolved.

## Project Context

This is the Teracloud Streams STT (Speech-to-Text) toolkit project. The toolkit integrates various speech recognition models including:
- WeNet models
- NeMo FastConformer models exported to ONNX
- Support for both CTC and transducer-based models

## Current Focus

Working on integrating NeMo FastConformer CTC model exported to ONNX format into the Streams toolkit.

## CRITICAL STATUS UPDATE (2025-06-23)

**MUST READ**: See `CRITICAL_FINDINGS_2025_06_23.md` for current state. Key points:

1. **Working Model**: `models/fastconformer_nemo_export/ctc_model.onnx` (471MB) - NOT the 2.6MB model
2. **Working Features**: Exist in `working_input.bin` with stats: min=-10.73, max=6.68, mean=-3.91
3. **Main Issue**: C++ feature extraction produces wrong statistics (min=-23.03, max=5.84)
4. **Model Interface**: Large model uses `processed_signal` input, not `audio_signal`
5. **Verified Working**: `python3 test_working_input_again.py` produces correct transcription

**DO NOT** trust files named "correct_features" - they don't produce correct output.
**DO NOT** use the 2.6MB model in `nemo_fastconformer_streaming/` - wrong interface.