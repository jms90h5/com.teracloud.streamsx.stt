# Streams STT Operator Guide

## OnnxSTT Operator

The OnnxSTT operator provides ONNX-based speech-to-text transcription with support for multiple model types including NeMo FastConformer CTC models.

### Input Port
- **Type**: `tuple<blob audioChunk, uint64 audioTimestamp>`
- **Description**: Raw audio data in 16-bit PCM format
- **audioChunk**: Binary blob containing audio samples
- **audioTimestamp**: Timestamp in milliseconds

### Output Port
- **Type**: `tuple<rstring text, boolean isFinal, float64 confidence>`
- **Description**: Transcription results
- **text**: Transcribed text
- **isFinal**: Whether this is a final transcription
- **confidence**: Confidence score (0.0-1.0)

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| encoderModel | rstring | Yes | - | Path to ONNX model file |
| vocabFile | rstring | Yes | - | Path to vocabulary/tokens file |
| cmvnFile | rstring | Yes | - | Path to CMVN stats (use "none" for NeMo) |
| modelType | rstring | No | "CACHE_AWARE_CONFORMER" | Model type: "CACHE_AWARE_CONFORMER" or "NEMO_CTC" |
| blankId | int32 | No | 0 | Blank token ID for CTC (NeMo uses 1024) |
| sampleRate | int32 | No | 16000 | Audio sample rate in Hz |
| chunkSizeMs | int32 | No | 100 | Processing chunk size in milliseconds |
| provider | rstring | No | "CPU" | ONNX provider: "CPU", "CUDA", "TensorRT" |
| numThreads | int32 | No | 4 | Number of CPU threads |

### Example: NeMo FastConformer CTC

```spl
composite NeMoCTCTranscription {
    graph
        // Audio source
        stream<blob audioChunk, uint64 audioTimestamp> AudioStream = 
            FileAudioSource() {
                param
                    filename: "speech.wav";
                    sampleRate: 16000;
            }
        
        // Speech-to-text with NeMo CTC
        stream<rstring text, boolean isFinal, float64 confidence> Transcription = 
            OnnxSTT(AudioStream) {
                param
                    encoderModel: getThisToolkitDir() + "/models/fastconformer_nemo_export/ctc_model.onnx";
                    vocabFile: getThisToolkitDir() + "/models/fastconformer_nemo_export/tokens.txt";
                    cmvnFile: "none";  // NeMo doesn't use CMVN
                    modelType: "NEMO_CTC";
                    blankId: 1024;  // NeMo CTC blank token
                    sampleRate: 16000;
                    provider: "CPU";
                    numThreads: 4;
            }
        
        // Output results
        () as ResultSink = Custom(Transcription) {
            logic
                onTuple Transcription: {
                    printStringLn("Text: " + text);
                    printStringLn("Final: " + (rstring)isFinal);
                    printStringLn("Confidence: " + (rstring)confidence);
                }
        }
}
```

### Example: Cache-Aware Conformer

```spl
stream<rstring text, boolean isFinal, float64 confidence> Transcription = 
    OnnxSTT(AudioStream) {
        param
            encoderModel: "models/conformer_encoder.onnx";
            vocabFile: "models/vocabulary.txt";
            cmvnFile: "models/global_cmvn.stats";
            modelType: "CACHE_AWARE_CONFORMER";
            blankId: 0;
            sampleRate: 16000;
}
```

## FileAudioSource Composite

Helper operator for reading audio files and converting to the expected stream format.

### Output Port
- **Type**: `tuple<blob audioChunk, uint64 audioTimestamp>`

### Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| filename | rstring | Yes | - | Path to audio file |
| blockSize | uint32 | No | 3200 | Bytes per chunk (100ms @ 16kHz) |
| sampleRate | int32 | No | 16000 | Sample rate |
| bitsPerSample | int32 | No | 16 | Bits per sample |
| channelCount | int32 | No | 1 | Number of channels |

## Troubleshooting

### Common Issues

**Empty transcriptions**
- Verify audio format matches model expectations (16kHz, mono, 16-bit)
- Check model and vocabulary files exist and are readable
- For NeMo models, ensure cmvnFile is set to "none"

**"Model input/output schema mismatch" errors**
- Verify modelType matches the actual model architecture
- Check blankId is correct (0 for most models, 1024 for NeMo)
- Ensure model was exported with correct input/output names

**Poor transcription quality**
- Verify audio quality and noise levels
- Check if model expects normalized or unnormalized features
- Try adjusting chunkSizeMs for better results

**Memory errors**
- Rebuild implementation library: `cd impl && make clean && make`
- Check for version mismatches between ONNX Runtime libraries

### Performance Tuning

**CPU Performance**
- Increase numThreads up to number of physical cores
- Use larger chunkSizeMs (200-500ms) for batch processing
- Consider using TensorRT provider for NVIDIA GPUs

**Memory Usage**
- Smaller chunkSizeMs reduces memory footprint
- Monitor for memory leaks in long-running applications

**Latency Optimization**
- Use smaller chunkSizeMs (50-100ms) for real-time applications
- Enable streaming mode when implemented
- Consider GPU acceleration for large-scale deployments

## Model Preparation

### NeMo Model Export
```python
# Export NeMo model to ONNX
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("model_name")
model.export(
    "ctc_model.onnx",
    input_names=["processed_signal", "processed_signal_length"],
    output_names=["log_probs", "encoded_lengths"]
)
```

### Vocabulary Extraction
```python
# Extract vocabulary from NeMo model
with open("tokens.txt", "w") as f:
    for token in model.tokenizer.vocab:
        f.write(token + "\n")
```

## Best Practices

1. **Audio Preprocessing**
   - Ensure consistent sample rate (resample if needed)
   - Use mono audio for best compatibility
   - Apply noise reduction if audio quality is poor

2. **Resource Management**
   - Initialize operator once and reuse for multiple files
   - Process audio in reasonable chunk sizes
   - Monitor memory usage in production

3. **Error Handling**
   - Check for valid model files before starting
   - Handle punctuation markers appropriately
   - Log confidence scores for quality monitoring

4. **Production Deployment**
   - Test with representative audio samples
   - Benchmark performance on target hardware
   - Set up monitoring for transcription quality