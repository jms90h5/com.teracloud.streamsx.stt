# FastConformer ONNX Model Test Guide

## Overview

This guide documents the test programs for the NeMo FastConformer model that has been exported to ONNX format. The model performs CTC-based speech recognition and is designed for streaming applications.

## Model Information

- **Model Location**: `models/fastconformer_nemo_export/`
- **Model File**: `ctc_model.onnx` (471.5 MB)
- **Vocabulary**: 1024 BPE tokens + 1 blank token (ID: 1024)
- **Tokenizer**: SentencePiece model at `tokenizer/tokenizer.model`
- **Input**: Mel-spectrogram features (80 mel bins)
- **Output**: Log probabilities for CTC decoding

### Model Parameters
- Sample Rate: 16,000 Hz
- Window Size: 25ms
- Window Stride: 10ms  
- FFT Size: 512
- Mel Bins: 80
- Window Type: Hann

## Test Programs

### 1. Python Tests

#### test_nemo_model.py
Basic model validation test that checks:
- Model loading
- Input/output tensor information
- Dynamic input size support

```bash
cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt
python test_nemo_model.py
```

#### test_nemo_transcription.py
Full transcription test with:
- WAV file loading
- Simple feature extraction
- CTC decoding
- Text output

```bash
python test_nemo_transcription.py
```

### 2. C++ Tests

#### test_fastconformer_simple.cpp
Simplified test without external dependencies:
- Basic mel-like feature extraction
- ONNX Runtime inference
- Greedy CTC decoding

Compile and run:
```bash
# Compile
make -f Makefile.nemo test_fastconformer_simple

# Run with proper library paths
LD_LIBRARY_PATH=/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/deps/onnxruntime/lib:$LD_LIBRARY_PATH ./test_fastconformer_simple
```

#### test_fastconformer_nemo.cpp
Comprehensive test with proper feature extraction:
- Uses kaldi-native-fbank for accurate mel-spectrograms
- Full CTC decoding pipeline
- Performance benchmarking

Compile and run:
```bash
# Compile
make -f Makefile.nemo test_fastconformer_nemo

# Run
LD_LIBRARY_PATH=/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/deps/onnxruntime/lib:/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/impl/lib:$LD_LIBRARY_PATH ./test_fastconformer_nemo
```

## Expected Results

The test audio file `test_data/audio/librispeech-1995-1837-0001.wav` should produce:
```
Expected transcription: "it was the first great song"
```

Note: The actual output may vary slightly due to:
- BPE tokenization (subword units)
- Model accuracy
- Feature extraction differences

## Troubleshooting

### Library Loading Issues
If you get library loading errors, ensure the LD_LIBRARY_PATH includes:
- ONNX Runtime: `deps/onnxruntime/lib`
- Kaldi Native Fbank: `impl/lib`

### Feature Extraction
The simple test uses basic energy-based features. For accurate results, use the comprehensive test with kaldi-native-fbank.

### Memory Usage
The model is large (471MB). Ensure sufficient memory is available.

### Input Shape
The model expects:
- Input: `[batch_size, n_mels, time_steps]`
- Length: `[batch_size]` 

Where:
- batch_size = 1 (for testing)
- n_mels = 80
- time_steps = variable (dynamic)

## Next Steps

1. **Integration with Streams**:
   - Use the C++ implementation as a basis for the OnnxSTT operator
   - Implement streaming support with overlapping windows
   - Add real-time VAD integration

2. **Performance Optimization**:
   - Implement int8 quantization (model supports it)
   - Use ONNX Runtime optimization features
   - Profile and optimize feature extraction

3. **Enhanced Decoding**:
   - Implement beam search decoding
   - Add language model rescoring
   - Support for confidence scores

## Building the Complete Pipeline

To build all test programs:
```bash
make -f Makefile.nemo all
```

To run all tests:
```bash
make -f Makefile.nemo test
```

## Performance Metrics

Expected performance on modern hardware:
- Feature Extraction: ~10-20ms per second of audio
- Inference: ~50-100ms per second of audio  
- Real-time Factor: < 0.2 (5x faster than real-time)

## Additional Resources

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [CTC Decoding Explained](https://distill.pub/2017/ctc/)