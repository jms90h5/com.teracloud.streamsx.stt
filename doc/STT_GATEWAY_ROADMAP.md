# STT Gateway Migration Roadmap

## Overview

This document outlines the remaining work to complete the STT Gateway migration into com.teracloud.streamsx.stt toolkit.

## Completed ✅

### Phase 1: Foundation (December 2024)
- [x] Unified type system (UnifiedTypes.spl)
- [x] Backend adapter interface (STTBackendAdapter)
- [x] NeMo backend adapter implementation
- [x] UnifiedSTT operator with code generation
- [x] Watson adapter placeholder structure
- [x] Documentation and examples

## In Progress 🔄

### Phase 2: Testing and Validation (Current)
- [ ] Build and test UnifiedSTT operator
- [ ] Verify NeMo adapter functionality
- [ ] Performance benchmarking
- [ ] Integration test suite

## Upcoming Work 📋

### Phase 3: Watson STT Implementation (January 2025)

#### Week 1: WebSocket Infrastructure
```cpp
// Required components
class WebSocketClient {
    // WebSocket connection management
    // Message framing and parsing
    // Async event handling
};

class AuthManager {
    // IBM Cloud IAM authentication
    // Token refresh logic
    // Credential management
};
```

**Dependencies to add:**
- websocketpp or boost::beast
- OpenSSL for TLS
- JSON parser (nlohmann/json or RapidJSON)

#### Week 2: Watson Protocol Implementation
- Audio streaming protocol
- Message format handling
- Result parsing
- Error recovery

#### Week 3: Integration and Testing
- Complete WatsonSTTAdapter
- End-to-end testing
- Performance tuning
- Documentation

### Phase 4: Voice Gateway Support (February 2025)

#### Week 1: VoiceGatewaySource Operator
```spl
composite VoiceGatewaySource(
    output CallerAudio, AgentAudio, CallMeta) {
    // WebSocket to Voice Gateway
    // SIP metadata extraction
    // Real-time audio streaming
}
```

#### Week 2: Telephony Features
- DTMF detection
- Call state management
- Recording capabilities
- Metadata enrichment

### Phase 5: Additional Backends (March 2025)

#### Google Cloud Speech-to-Text
- gRPC client implementation
- Streaming recognition
- Multi-language support

#### Azure Cognitive Services
- REST/WebSocket API
- Custom speech models
- Real-time transcription

#### OpenAI Whisper (Local)
- ONNX model integration
- Multilingual support
- Timestamp generation

### Phase 6: Advanced Features (April 2025)

#### Multi-Backend Orchestration
```spl
composite MultiBackendSTT {
    // Parallel processing
    // Result comparison
    // Confidence-based selection
}
```

#### Enhanced Telephony
- Speaker diarization
- Emotion detection
- Call summarization
- Compliance recording

## Technical Requirements

### Build System Updates
```makefile
# Optional backend compilation
ifdef ENABLE_WATSON_BACKEND
    SOURCES += src/backends/WatsonSTTAdapter.cpp
    LDFLAGS += -lwebsocketpp -lssl -lcrypto
endif

ifdef ENABLE_GOOGLE_BACKEND
    SOURCES += src/backends/GoogleSTTAdapter.cpp
    LDFLAGS += -lgrpc++ -lprotobuf
endif
```

### Testing Infrastructure
```bash
# Unit tests for each backend
test/unit/
├── test_nemo_adapter.cpp
├── test_watson_adapter.cpp
└── test_backend_factory.cpp

# Integration tests
test/integration/
├── test_unified_stt.spl
├── test_backend_fallback.spl
└── test_2channel_telephony.spl
```

## Migration Support

### For STT Gateway Users

1. **Operator Mapping Guide**
   ```
   WatsonSTT → UnifiedSTT(backend: "watson")
   IBMVoiceGatewaySource → VoiceGatewaySource
   ```

2. **Configuration Migration Tool**
   ```python
   # Script to convert gateway configs
   python migrate_stt_gateway.py \
     --input gateway_app.spl \
     --output unified_app.spl
   ```

3. **Compatibility Layer**
   ```spl
   // Temporary backward compatibility
   public composite WatsonSTT = UnifiedSTT {
       param backend: "watson";
   }
   ```

## Performance Targets

### Latency Requirements
- Local (NeMo): < 50ms per chunk
- Cloud (Watson): < 200ms per chunk
- Fallback switching: < 100ms

### Throughput Goals
- Single stream: Real-time (1.0x)
- Multi-stream: 100+ concurrent
- 2-channel: No degradation

### Resource Usage
- Memory: < 1GB per stream
- CPU: < 1 core per stream
- Network: Adaptive bitrate

## Risk Mitigation

### Technical Risks
1. **WebSocket Complexity**
   - Mitigation: Use proven libraries
   - Fallback: REST API mode

2. **Authentication Issues**
   - Mitigation: Comprehensive retry logic
   - Fallback: Multiple auth methods

3. **Network Reliability**
   - Mitigation: Local buffering
   - Fallback: Store-and-forward

### Schedule Risks
1. **Dependency Delays**
   - Mitigation: Modular development
   - Fallback: Phased delivery

2. **Testing Complexity**
   - Mitigation: Automated test suite
   - Fallback: Extended beta period

## Success Metrics

### Functional
- [ ] All gateway features migrated
- [ ] Multi-backend support working
- [ ] 2-channel telephony integrated
- [ ] Performance targets met

### Quality
- [ ] 99.9% uptime for local backends
- [ ] 99% success rate for cloud backends
- [ ] < 1% WER degradation
- [ ] Zero data loss

### Adoption
- [ ] Migration guide published
- [ ] 10+ example applications
- [ ] Customer feedback incorporated
- [ ] Training materials created

## Next Immediate Steps

1. **This Week**
   - Test UnifiedSTT compilation
   - Fix any build issues
   - Create basic test suite

2. **Next Week**
   - Research WebSocket libraries
   - Design Watson protocol handler
   - Set up development environment

3. **End of Month**
   - Working Watson prototype
   - Basic Voice Gateway design
   - Updated documentation

## Questions for Discussion

1. Which WebSocket library? (websocketpp vs boost::beast)
2. JSON parser preference? (nlohmann vs RapidJSON)
3. Authentication strategy? (IAM vs API key)
4. Fallback behavior? (automatic vs manual)
5. Monitoring approach? (metrics, logs, traces)

## Conclusion

The STT Gateway migration is progressing well with Phase 1 complete. The modular architecture allows incremental delivery while maintaining stability. Focus now shifts to Watson implementation and real-world testing.