# Quick Start Guide for Next Session

## First Steps
1. **Read**: `CRITICAL_FINDINGS_2025_06_23.md` for complete context
2. **Verify**: Run `python3 test_working_input_again.py` to confirm model still works
3. **Check**: `working_input.bin` exists (279,680 bytes)

## What Works
```bash
# This produces correct transcription:
cd /home/streamsadmin/workspace/terracloud/toolkits/com.teracloud.streamsx.stt
python3 test_working_input_again.py
# Output: "it was the first great sorrow of his life..."
```

## What's Broken
```bash
# This uses wrong model and produces empty output:
cd test
./test_real_nemo_fixed --nemo-model ../opt/models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx ...
```

## Key Files
- **Working Model**: `models/fastconformer_nemo_export/ctc_model.onnx` (471MB)
- **Working Features**: `working_input.bin` (min=-10.73, max=6.68)
- **Wrong Model**: `opt/models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx` (2.6MB)

## Next Tasks
1. Update C++ to use correct model path
2. Fix input tensor names (`processed_signal` not `audio_signal`)
3. Debug why C++ features have wrong statistics
4. Test SPL samples with fixes

## Do NOT
- Use files named "correct_features" (they're wrong)
- Trust documentation claiming "complete success"
- Change working_input.bin or test_working_input_again.py