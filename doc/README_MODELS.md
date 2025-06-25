# Teracloud Streams STT Toolkit - Model Support

This toolkit supports multiple speech-to-text models through different operators.

## Supported Models

### 1. NVIDIA NeMo Models
- **Operator**: `NeMoSTT`
- **Model Format**: `.nemo` files
- **Example Model**: `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`
- **Features**: 
  - Cache-aware streaming
  - Trained on 10,000+ hours of speech data
  - Excellent accuracy for real-time streaming

### 2. ONNX Models
- **Operator**: `OnnxSTT`
- **Model Format**: `.onnx` files
- **Example Models**:
  - Sherpa ONNX Zipformer models
  - Exported Whisper models
  - Any ONNX-compatible ASR model

### 3. WeNet Models
- **Operator**: `WenetSTT`
- **Model Format**: WeNet checkpoint files
- **Features**: Streaming-capable models

## Using Different Models

### Command Line Usage

```bash
# Using NeMo model
./output/bin/standalone \
    modelType=nemo \
    modelPath=/path/to/model.nemo \
    audioFile=/path/to/audio.wav

# Using ONNX model  
./output/bin/standalone \
    modelType=onnx \
    modelPath=/path/to/model.onnx \
    audioFile=/path/to/audio.wav
```

### SPL Application Example

```spl
// For NeMo models
stream<rstring text> Transcription = NeMoSTT(AudioStream) {
    param
        modelPath: "/path/to/model.nemo";
        audioFormat: "mono16k";
        chunkDurationMs: 5000;
}

// For ONNX models
stream<rstring text, boolean isFinal, float64 confidence> 
    Transcription = OnnxSTT(AudioStream) {
    param
        encoderModel: "/path/to/encoder.onnx";
        vocabFile: "/path/to/tokens.txt";
        sampleRate: 16000;
}
```

## Model Configuration

### NeMo Models
- Download from HuggingFace or NVIDIA NGC
- Ensure Python environment has NeMo dependencies installed
- Models include vocabulary and configuration

### ONNX Models
- Export from PyTorch/TensorFlow models
- Requires separate vocabulary file
- Can use different execution providers (CPU, CUDA)

## Performance Considerations

- **NeMo**: Best accuracy, requires Python runtime
- **ONNX**: Good performance, pure C++ implementation
- **WeNet**: Lightweight, good for embedded systems

## Adding New Models

To add support for a new model type:

1. Create a new operator in the toolkit
2. Implement the model loading and inference logic
3. Update the generic sample application
4. Document the model requirements

## Troubleshooting

### NeMo Models
- Ensure Python 3.9+ is installed
- Install required packages: `pip install nemo-toolkit`
- Check model compatibility with NeMo version

### ONNX Models
- Verify ONNX Runtime is installed
- Check model input/output shapes
- Ensure vocabulary file matches model