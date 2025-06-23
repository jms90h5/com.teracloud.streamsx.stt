# NeMo FastConformer Integration Summary

## Current State (2024-06-22)

### What Works ✅

1. **ONNX Model Loading and Inference**
   - NeMo FastConformer CTC model successfully exported to ONNX
   - C++ ONNX Runtime inference works correctly
   - When given correct features, C++ produces accurate transcriptions
   - Model architecture: FastConformer with CTC output head
   - Vocabulary: 1024 tokens (including blank token at ID 1024)

2. **C++ Infrastructure**
   - NeMoCTCImpl class successfully implements model loading and inference
   - CTC decoding algorithm works correctly
   - Proper tensor shape handling ([batch, features, time])
   - Integration with Streams toolkit compiles successfully

3. **Python Reference Implementation**
   - test_nemo_correct.py produces correct transcriptions
   - Uses librosa for feature extraction with specific parameters:
     - 80 mel bins
     - 512 FFT size
     - 400 sample window (25ms)
     - 160 sample hop (10ms)
     - Hann window
     - Power=2.0 (power spectrogram)
     - Natural log scale
     - Dither of 1e-5
     - NO normalization

### What Doesn't Work ❌

1. **C++ Feature Extraction**
   - ImprovedFbank implementation produces different features than librosa
   - Average difference of 3.66 in feature values
   - Results in all blank token predictions
   - Key differences:
     - Different mel filterbank implementation
     - Possible FFT implementation differences
     - Window function application differences

2. **Streams Integration**
   - Library linking issues prevent running Streams applications
   - NeMoSTT operator not properly linked to libs2t_impl.so
   - ONNX Runtime library path issues in Streams environment

### Root Cause

The fundamental issue is that the C++ feature extraction (ImprovedFbank) doesn't match librosa's mel spectrogram implementation. Even small differences in:
- Mel filterbank construction
- FFT computation
- Window function application
- Power spectrum calculation

Result in features that the model cannot process correctly, leading to all blank predictions.

### Verified Working Configuration

- **Test Audio**: test_data/audio/librispeech-1995-1837-0001.wav
- **Expected Output**: "it was the first great sorrow of his life"
- **Working Features**: Shape [80, 874], Min: -10.73, Max: 6.68, Mean: -3.91
- **Model Input**: [batch=1, features=80, time=874]

### Next Steps

1. **Option A: Fix C++ Feature Extraction**
   - Port librosa's exact mel spectrogram implementation to C++
   - Or use an existing C++ library that matches librosa (e.g., essentia)

2. **Option B: Use Pre-computed Features**
   - Compute features in Python and pass to C++ for inference
   - Not suitable for real-time streaming

3. **Option C: Retrain Model**
   - Train model with features from available C++ implementation
   - Requires access to training data and compute

4. **Fix Streams Integration**
   - Resolve library linking issues
   - Ensure ONNX Runtime is properly accessible in Streams environment
   - Test end-to-end SPL application execution

### Code Files

- **Working Python**: `test_nemo_correct.py`
- **C++ Implementation**: `impl/include/NeMoCTCImpl.cpp`
- **Test Program**: `test_nemo_direct.cpp`
- **SPL Samples**: `samples/com.teracloud.streamsx.stt.sample/NeMoCTCSample.spl`

### Key Insight

The model is highly sensitive to the exact feature extraction method. Even with the same parameters, different implementations (librosa vs custom C++) produce features different enough to break inference. This is a common issue when porting Python ML models to C++.