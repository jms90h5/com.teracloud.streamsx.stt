# Known Issues and Solutions

This document captures all the issues encountered during toolkit development and their solutions to prevent repeated debugging sessions.

## Critical Issues Resolved

### 1. Model Input Dimension Mismatch

**Issue**: Model inference fails with error:
```
ERROR: Got invalid dimensions for input: audio_signal for the following indices
 index: 1 Got: 125 Expected: 80
```

**Root Cause**: The NeMo FastConformer model expects input shape `[batch, features, time]` but the implementation was providing `[batch, time, features]`.

**Solution**: Fixed in `impl/include/NeMoCTCImpl.cpp` lines 206-225:
```cpp
// Transpose from [time, features] to [features, time]
std::vector<float> transposed_features(mel_features.size());
for (int t = 0; t < n_frames; t++) {
    for (int f = 0; f < n_mels; f++) {
        transposed_features[f * n_frames + t] = mel_features[t * n_mels + f];
    }
}
std::vector<int64_t> audio_shape = {1, n_mels, n_frames};  // [batch, features, time]
```

**Status**: ✅ FIXED - The transpose operation is now part of the implementation.

### 2. Python Virtual Environment Blocks SPL Compilation

**Issue**: SPL compilation fails with:
```
make: *** No rule to make target '.../impl/venv/lib/python3.11/site-packages/setuptools/_vendor/jaraco/text/Lorem', needed by 'SimpleTest.sab'.
```

**Root Cause**: The Python setuptools package contains a test file named "Lorem" (Lorem ipsum text) that the SPL compiler tries to process as a dependency.

**Workarounds**:
1. **Temporary**: Move venv before compiling SPL
   ```bash
   mv impl/venv /tmp/stt_venv_backup
   # Compile SPL samples
   mv /tmp/stt_venv_backup impl/venv
   ```

2. **Permanent**: Already implemented - `.toolkit-ignore` file excludes `impl/venv/`
   
3. **Alternative**: Use Python installed outside the toolkit directory

**Status**: ⚠️ WORKAROUND - The .toolkit-ignore should work but may need SPL compiler updates.

### 3. SPL Output Attribute Naming

**Issue**: Compilation fails with:
```
error: class ... has no member named 'set_transcription'
```

**Root Cause**: The NeMoSTT operator tries to set an attribute called `transcription` but the sample used `text`.

**Solution**: Match the attribute name in the output schema:
```spl
// Wrong:
stream<rstring text> Transcription = NeMoSTT(AudioStream) { ... }

// Correct:
stream<rstring transcription> Transcription = NeMoSTT(AudioStream) { ... }
```

**Status**: ✅ FIXED - Samples updated with correct attribute names.

### 4. Empty Tokens File After Model Export

**Issue**: The `tokens.txt` file is empty or 0 bytes after model export.

**Root Cause**: The export script may not generate the vocabulary file in some cases.

**Solution**: Created `fix_tokens.py` script that extracts vocabulary from the model:
```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
)
tokenizer = model.tokenizer
with open("opt/models/fastconformer_ctc_export/tokens.txt", 'w') as f:
    for i in range(tokenizer.tokenizer.get_piece_size()):
        piece = tokenizer.tokenizer.id_to_piece(i)
        f.write(f"{piece}\n")
```

**Status**: ✅ FIXED - Documented in README.md and setup scripts.

### 5. Model Path Issues in Tests

**Issue**: C++ tests fail with "model.onnx not found" due to incorrect paths.

**Root Cause**: Test files had wrong relative paths `../models/` instead of `../opt/models/`.

**Solution**: Updated all test files to use correct paths:
```cpp
std::string model_path = "../opt/models/fastconformer_ctc_export/model.onnx";
std::string tokens_path = "../opt/models/fastconformer_ctc_export/tokens.txt";
```

**Status**: ✅ FIXED - Test files updated.

### 6. 125 Frame Limitation

**Issue**: Transcriptions are truncated to ~2 seconds of audio.

**Location**: `impl/include/NeMoCTCImpl.cpp` line 171

**Root Cause**: Debug code limits processing to 125 frames:
```cpp
const int EXPECTED_FRAMES = 125;
```

**Solution**: This should be removed or made configurable for production use.

**Status**: ⚠️ TODO - Currently still limited, needs fix for full-length audio.

### 7. ONNX Runtime Header Conflicts

**Issue**: C++ compilation fails with `NO_EXCEPTION` macro conflicts.

**Root Cause**: ONNX Runtime headers define macros that conflict with Streams SDK.

**Solution**: Already implemented - use `onnx_wrapper.hpp` instead of direct ONNX includes.

**Status**: ✅ FIXED - Wrapper header isolates ONNX dependencies.

## Testing Checklist

After any toolkit changes, verify these work:

1. **C++ Library Build**:
   ```bash
   cd impl && make clean && make
   ```

2. **Model Export**:
   ```bash
   source ./activate_python.sh
   python impl/bin/export_model_ctc_patched.py
   # Verify tokens.txt has 1024 lines
   wc -l opt/models/fastconformer_ctc_export/tokens.txt
   ```

3. **C++ Test**:
   ```bash
   cd test
   g++ -std=c++14 -O2 -I../impl/include -I../lib/onnxruntime/include \
       test_with_audio.cpp ../impl/lib/libs2t_impl.so \
       -L../lib/onnxruntime/lib -lonnxruntime -ldl \
       -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../lib/onnxruntime/lib' \
       -o test_with_audio
   ./test_with_audio
   # Should output: "it was the first great"
   ```

4. **SPL Compilation** (with venv workaround):
   ```bash
   mv impl/venv /tmp/backup
   cd samples
   sc -a -t .. -M SimpleTest --output-directory output/SimpleTest SimpleTest.spl
   mv /tmp/backup impl/venv
   ```

## Environment Requirements

- Python 3.11+ (for venv and model export)
- GCC with C++14 support
- Teracloud Streams 7.2.0.1+
- 5GB free disk space (for model and dependencies)

## Model Details

- **Model**: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
- **Size**: ~460MB download, ~438MB ONNX file
- **Vocabulary**: 1024 SentencePiece tokens
- **Input**: 80 mel-spectrogram features
- **Expected shape**: [batch, 80, time_frames]

## Quick Verification Script

Create `verify_all.sh`:
```bash
#!/bin/bash
echo "1. Checking model files..."
[ -f "opt/models/fastconformer_ctc_export/model.onnx" ] && echo "✓ Model found" || echo "✗ Model missing"
[ -s "opt/models/fastconformer_ctc_export/tokens.txt" ] && echo "✓ Tokens found" || echo "✗ Tokens missing"

echo "2. Checking C++ library..."
[ -f "impl/lib/libs2t_impl.so" ] && echo "✓ Library built" || echo "✗ Library missing"

echo "3. Running C++ test..."
cd test && ./test_nemo_ctc_simple && echo "✓ C++ test passed" || echo "✗ C++ test failed"
```

## References

- Original issue discussion: See git history for `impl/include/NeMoCTCImpl.cpp`
- Test audio: `samples/audio/librispeech_3sec.wav`
- Expected output: "it was the first great song of his life"