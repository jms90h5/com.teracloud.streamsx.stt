# NeMo FastConformer Integration Plan

## Current Status

### ✅ What's Working
1. **Model Export**: Successfully exported NeMo FastConformer model to ONNX format
   - Model file: `ctc_model.onnx` (471.5 MB)
   - Vocabulary: 1024 BPE tokens + blank token
   - Tokenizer: SentencePiece model

2. **Preprocessing**: Identified correct preprocessing requirements
   - Log mel-spectrogram features (80 mel bins)
   - NO normalization (`normalize: NA`)
   - Natural log scale
   - Small dither (1e-5) added to audio
   - Hann window, 25ms frame, 10ms stride

3. **Inference**: Model runs and produces transcriptions
   - CTC decoding works
   - BPE tokens properly handled
   - Real-time capable performance

### ❌ Issues to Address
1. **Transcription Accuracy**: Model produces different output than expected
   - LibriSpeech file: Expected "it was the first great song" 
   - Actual: "it was the first great sorrow of his life..."
   - This may be correct - need to verify ground truth

2. **C++ Implementation**: Needs updating with correct preprocessing
   - Current implementations use incorrect feature extraction
   - Need to match Python preprocessing exactly

3. **Streaming Support**: Not yet implemented
   - Model supports dynamic input lengths
   - Need overlapping window approach for streaming

## Next Steps

### Phase 1: Verify Model Accuracy (High Priority)
1. **Find ground truth transcriptions**
   - Check LibriSpeech dataset for actual transcriptions
   - Create test set with verified transcriptions
   - Test on multiple audio files to establish baseline accuracy

2. **Validate preprocessing**
   - Compare with original NeMo preprocessing
   - Test with NeMo's own inference pipeline
   - Document exact preprocessing requirements

### Phase 2: C++ Implementation (High Priority)
1. **Update feature extraction**
   ```cpp
   // Key requirements:
   // - Log mel-spectrogram (not normalized)
   // - Natural log (not log10)
   // - Add dither before processing
   // - Exact window parameters
   ```

2. **Fix test programs**
   - Update `test_fastconformer_nemo.cpp` with correct preprocessing
   - Ensure kaldi-native-fbank parameters match NeMo
   - Add dither to audio before feature extraction

3. **Create reference implementation**
   - Simple, self-contained C++ program
   - Minimal dependencies
   - Clear documentation

### Phase 3: Streaming Implementation (Medium Priority)
1. **Design streaming architecture**
   - Chunk size: 160ms (optimal for model)
   - Overlap: 50% for smooth transitions
   - State management for CTC decoding

2. **Implement streaming pipeline**
   - Audio buffering
   - Feature extraction on chunks
   - Continuous CTC decoding
   - Handle partial results

### Phase 4: Fix Existing Streams Samples (High Priority)
1. **Update OnnxSTT operator implementation**
   - Fix `impl/include/OnnxSTTImpl.cpp` with correct preprocessing
   - Ensure proper model loading for NeMo exported models
   - Handle BPE vocabulary correctly
   - Add proper CTC decoding

2. **Fix existing SPL sample programs**
   - `samples/OnnxSTTSample/sample/OnnxSTT.spl`
   - `samples/UnifiedSTTSample/application/UnifiedSTT.spl`
   - Update model paths to use NeMo export
   - Fix preprocessing parameters
   - Ensure proper audio format handling

3. **Test and debug SPL applications**
   - Compile with `sc` command
   - Submit jobs with `streamtool`
   - Verify transcription output
   - Check performance metrics

### Phase 5: Enhanced Streams Integration (Medium Priority)
1. **Create new SPL samples**
   - NeMo-specific transcription sample
   - Real-time streaming sample
   - Multi-model comparison sample

2. **Performance optimization**
   - Implement batching for throughput
   - Add GPU support if available
   - Profile and optimize bottlenecks

3. **Documentation**
   - Update README files
   - Create user guide for NeMo models
   - Document SPL patterns for STT
   - Troubleshooting guide

## Technical Requirements

### Preprocessing Pipeline
```python
# Correct preprocessing steps:
1. Load audio (16kHz, mono)
2. Add dither: audio += 1e-5 * random_noise
3. Compute mel-spectrogram:
   - n_fft: 512
   - win_length: 400 (25ms)
   - hop_length: 160 (10ms)
   - window: hann
   - n_mels: 80
4. Convert to log scale: log(mel + 1e-10)
5. NO normalization
```

### Model Input/Output
- **Input 1**: `processed_signal` [batch, 80, time]
- **Input 2**: `processed_signal_length` [batch]
- **Output 1**: `log_probs` [batch, time_out, 1025]
- **Output 2**: `encoded_lengths` [batch]

### CTC Decoding
- Blank token ID: 1024
- BPE tokens use SentencePiece
- Greedy decoding: Remove blanks and repeated tokens
- Handle subword merging (▁ prefix)

## Validation Criteria

### Functionality
- [ ] Model loads without errors
- [ ] Preprocessing matches NeMo exactly
- [ ] Produces meaningful transcriptions
- [ ] Handles various audio lengths
- [ ] Streaming mode works correctly

### Performance
- [ ] Real-time factor < 0.3 (3x faster than real-time)
- [ ] Memory usage < 1GB
- [ ] Latency < 200ms for streaming chunks
- [ ] CPU usage reasonable (< 100% single core)

### Integration
- [ ] Works with Streams toolkit
- [ ] SPL operator compiles and runs
- [ ] Sample applications work
- [ ] Documentation complete

## Risk Mitigation

1. **Model Accuracy**: If transcriptions remain incorrect
   - Re-export with different settings
   - Try different NeMo models
   - Consider using original PyTorch model

2. **Performance Issues**: If too slow
   - Use int8 quantization
   - Optimize feature extraction
   - Reduce model size

3. **Integration Challenges**: If Streams integration fails
   - Create standalone service
   - Use REST API approach
   - Implement custom operator

## Timeline Estimate

- Phase 1 (Verification): 1-2 days
- Phase 2 (C++ Implementation): 2-3 days  
- Phase 3 (Streaming): 2-3 days
- Phase 4 (Integration): 3-4 days

Total: 8-12 days for complete integration