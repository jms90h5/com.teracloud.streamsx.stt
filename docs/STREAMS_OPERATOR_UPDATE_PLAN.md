# Detailed Plan for Updating Streams C++ Operators

## Overview
This document outlines the plan to update the Streams toolkit C++ operators to use the working approach from the standalone test programs, specifically for the NeMo FastConformer CTC model integration.

## Current Status
- ✅ Standalone test program works correctly with NeMo CTC model
- ✅ Proper feature extraction using ImprovedFbank
- ✅ Memory management issues resolved
- ✅ Model produces accurate transcriptions
- ✅ Performance: ~0.2 RTF (5x faster than real-time)

## Key Components That Work

### 1. Feature Extraction
- **ImprovedFbank** with specific settings:
  - Sample rate: 16000 Hz
  - Mel bins: 80
  - Window size: 25ms
  - Window stride: 10ms
  - FFT size: 512
  - Apply log: true
  - Dither: 1e-5
  - Normalize per feature: false (critical for NeMo)

### 2. Model Configuration
- Model type: NEMO_CTC
- Blank ID: 1024 (NeMo specific)
- Input tensors: processed_signal, processed_signal_length
- Output tensors: log_probs, encoded_lengths

### 3. CTC Decoding
- Greedy decoding with blank token handling
- BPE token processing with ▁ prefix handling

## Detailed Update Plan

### Phase 1: Update OnnxSTT Operator Core (Priority: High)

#### 1.1 Update OnnxSTTImpl Class
- [x] Already supports NEMO_CTC model type
- [x] Already uses NeMoCTCModel for CTC processing
- [ ] Add streaming buffer management for real-time processing
- [ ] Implement chunk-based processing for streaming

#### 1.2 Update SPL Operator Interface
- [x] modelType parameter already added
- [x] blankId parameter already added
- [ ] Add streamingMode parameter (optional, default: false)
- [ ] Add chunkOverlapMs parameter for streaming (optional, default: 50ms)

#### 1.3 Memory Management
- [x] Fixed ONNX tensor name storage using std::string
- [ ] Implement circular buffer for streaming audio
- [ ] Add proper cleanup in operator destructor

### Phase 2: Streaming Implementation (Priority: High)

#### 2.1 Audio Buffering
```cpp
class StreamingBuffer {
    std::vector<float> buffer_;
    size_t write_pos_ = 0;
    size_t read_pos_ = 0;
    size_t capacity_;
    
    void append(const float* data, size_t size);
    bool getChunk(std::vector<float>& chunk, size_t chunk_size, size_t overlap);
};
```

#### 2.2 Feature Caching
- Cache overlapping features between chunks
- Implement sliding window for feature extraction
- Handle chunk boundaries properly

#### 2.3 CTC State Management
- Track previous non-blank tokens across chunks
- Implement proper token merging at boundaries
- Handle partial words at chunk boundaries

### Phase 3: Operator Code Generation Updates (Priority: Medium)

#### 3.1 Update OnnxSTT_cpp.cgt
- [ ] Add streaming state management
- [ ] Implement proper punctuation handling for streaming
- [ ] Add performance metrics collection

#### 3.2 Update OnnxSTT_h.cgt
- [ ] Add streaming-related member variables
- [ ] Add chunk processing methods
- [ ] Add state reset methods for streaming

### Phase 4: Sample Applications (Priority: Medium)

#### 4.1 Batch Processing Sample
- [ ] Update BasicNeMoTest.spl for batch file processing
- [ ] Add performance benchmarking
- [ ] Include accuracy metrics

#### 4.2 Real-time Streaming Sample
- [ ] Create NeMoStreamingSTT.spl
- [ ] Implement audio source operator for microphone input
- [ ] Add WebSocket sink for real-time results

#### 4.3 Multi-model Sample
- [ ] Create sample comparing different models
- [ ] Include latency/accuracy trade-offs
- [ ] Support model switching at runtime

### Phase 5: Testing and Validation (Priority: High)

#### 5.1 Unit Tests
- [ ] Test feature extraction accuracy
- [ ] Test CTC decoding correctness
- [ ] Test streaming chunk handling

#### 5.2 Integration Tests
- [ ] Test with various audio formats
- [ ] Test with different chunk sizes
- [ ] Test error handling and recovery

#### 5.3 Performance Tests
- [ ] Measure RTF for different audio lengths
- [ ] Test memory usage patterns
- [ ] Benchmark against Python implementation

### Phase 6: Documentation (Priority: Medium)

#### 6.1 API Documentation
- [ ] Document all operator parameters
- [ ] Provide usage examples
- [ ] Include troubleshooting guide

#### 6.2 Integration Guide
- [ ] Step-by-step setup instructions
- [ ] Model conversion guide
- [ ] Performance tuning tips

## Implementation Timeline

### Week 1
- Complete Phase 1: Core operator updates
- Begin Phase 2: Streaming implementation

### Week 2
- Complete Phase 2: Streaming implementation
- Complete Phase 3: Code generation updates
- Begin Phase 5: Testing

### Week 3
- Complete Phase 4: Sample applications
- Complete Phase 5: Testing and validation
- Complete Phase 6: Documentation

## Key Technical Considerations

### 1. Thread Safety
- Ensure all operator methods are thread-safe
- Use proper synchronization for streaming buffers
- Handle concurrent tuple processing

### 2. Error Handling
- Graceful handling of model loading failures
- Recovery from temporary audio dropouts
- Proper error reporting to Streams runtime

### 3. Resource Management
- Efficient memory usage for long-running streams
- Proper cleanup on operator shutdown
- ONNX session lifecycle management

### 4. Performance Optimization
- Minimize memory allocations in hot path
- Use ONNX Runtime optimization features
- Consider GPU acceleration support

## Success Criteria

1. **Functionality**
   - Operator produces same results as standalone test
   - Streaming mode works without dropping audio
   - All samples run without errors

2. **Performance**
   - Maintain <0.25 RTF for batch processing
   - Achieve <100ms latency for streaming
   - Memory usage stable over long runs

3. **Reliability**
   - No memory leaks
   - Graceful error handling
   - Stable under load

4. **Usability**
   - Clear documentation
   - Easy-to-follow examples
   - Helpful error messages

## Next Steps

1. Review and approve this plan
2. Create feature branch for implementation
3. Begin Phase 1 implementation
4. Set up continuous testing infrastructure
5. Schedule code reviews at each phase completion