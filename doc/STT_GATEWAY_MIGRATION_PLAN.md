# STT Gateway Migration Plan

## Executive Summary

This document outlines a detailed plan for migrating IBM STT Gateway functionality into the com.teracloud.streamsx.stt toolkit, with the goal of supporting multiple Speech-to-Text backends including local NeMo models and cloud-based Watson STT services.

## Goals and Objectives

### Primary Goals
1. Create a unified STT interface supporting multiple backends
2. Maintain backward compatibility where feasible
3. Enable seamless switching between local and cloud STT providers
4. Support both real-time and batch processing modes
5. Provide telephony integration capabilities

### Design Principles
- **Modularity**: Each STT backend as a pluggable component
- **Minimal Dependencies**: Core functionality with optional features
- **Performance**: Maintain current speed advantages of local processing
- **Flexibility**: Support various audio sources and formats
- **Extensibility**: Easy addition of new STT providers

## Architecture Overview

### Proposed Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                    │
│         (User SPL Applications)                      │
├─────────────────────────────────────────────────────┤
│              Unified STT Interface                   │
│         (Common operators and schemas)               │
├─────────────────────────────────────────────────────┤
│              Backend Adapter Layer                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ │
│  │  NeMo   │ │ Watson  │ │ Google  │ │  Future  │ │
│  │ Adapter │ │ Adapter │ │ Adapter │ │ Backends │ │
│  └─────────┘ └─────────┘ └─────────┘ └──────────┘ │
├─────────────────────────────────────────────────────┤
│              Core Services Layer                     │
│  (Audio processing, Feature extraction, Utilities)   │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Unified STT Operator** (`UnifiedSTT`)
   - Single operator interface for all backends
   - Runtime backend selection
   - Common input/output schemas

2. **Backend Adapters**
   - NeMoSTTAdapter (existing functionality)
   - WatsonSTTAdapter (new)
   - GoogleSTTAdapter (future)
   - AzureSTTAdapter (future)

3. **Audio Source Operators**
   - FileAudioSource (existing)
   - VoiceGatewaySource (new, from STT Gateway)
   - StreamAudioSource (enhanced)

4. **Supporting Services**
   - Authentication Manager
   - Connection Pool Manager
   - Metrics Collector
   - Error Handler

## Detailed Migration Phases

### Phase 1: Foundation and Interface Design (Weeks 1-3)

#### 1.1 Define Common Schemas
```
// Common input schema
type AudioInput = tuple<
    blob audioData,
    uint64 timestamp,
    rstring encoding,      // "pcm16", "mp3", "opus", etc.
    int32 sampleRate,
    int32 channels,
    rstring languageCode,  // "en-US", "es-ES", etc.
    map<rstring,rstring> metadata  // extensible metadata
>;

// Common output schema  
type TranscriptionOutput = tuple<
    rstring text,
    float64 confidence,
    boolean isFinal,
    list<WordTiming> wordTimings,  // optional
    map<rstring,rstring> metadata,
    rstring backend  // which backend produced this
>;

type WordTiming = tuple<
    rstring word,
    float64 startTime,
    float64 endTime,
    float64 confidence
>;
```

#### 1.2 Create UnifiedSTT Operator Interface
```spl
public composite UnifiedSTT(input AudioStream; output TranscriptionStream) {
    param
        expression<rstring> $backend;  // "nemo", "watson", "google", etc.
        expression<rstring> $modelPath;  // for local models
        expression<rstring> $apiEndpoint;  // for cloud services
        expression<rstring> $apiKey;  // for cloud services
        expression<rstring> $languageCode: "en-US";
        expression<boolean> $enableWordTimings: false;
        expression<boolean> $enablePunctuation: true;
        expression<int32> $maxAlternatives: 1;
        expression<map<rstring,rstring>> $backendConfig;  // backend-specific config
}
```

#### 1.3 Design Backend Adapter Interface
```cpp
class STTBackendAdapter {
public:
    virtual ~STTBackendAdapter() = default;
    
    // Initialize the backend
    virtual bool initialize(const BackendConfig& config) = 0;
    
    // Process audio chunk
    virtual TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) = 0;
    
    // Finalize and get final results
    virtual TranscriptionResult finalize() = 0;
    
    // Get backend capabilities
    virtual BackendCapabilities getCapabilities() const = 0;
    
    // Health check
    virtual bool isHealthy() const = 0;
};
```

#### 1.4 Implement Backend Factory
```cpp
class STTBackendFactory {
public:
    static std::unique_ptr<STTBackendAdapter> createBackend(
        const std::string& backendType,
        const BackendConfig& config);
        
    static std::vector<std::string> getAvailableBackends();
    static bool isBackendAvailable(const std::string& backendType);
};
```

### Phase 2: NeMo Backend Refactoring (Weeks 4-5)

#### 2.1 Refactor Existing NeMoSTT
- Extract current NeMoSTT implementation into NeMoSTTAdapter
- Implement STTBackendAdapter interface
- Maintain all current optimizations

#### 2.2 Create Configuration Mapping
```cpp
class NeMoConfig : public BackendConfig {
    std::string modelPath;
    std::string tokensPath;
    int32_t numThreads = 4;
    std::string provider = "CPU";  // or "CUDA"
    bool enableCaching = true;
    // ... other NeMo-specific options
};
```

#### 2.3 Update Existing Operators
- Modify NeMoSTT to use UnifiedSTT internally
- Maintain backward compatibility
- Add deprecation notices

### Phase 3: Watson STT Integration (Weeks 6-8)

#### 3.1 Implement WatsonSTTAdapter
```cpp
class WatsonSTTAdapter : public STTBackendAdapter {
private:
    std::unique_ptr<WebSocketClient> wsClient;
    std::unique_ptr<AuthManager> authManager;
    WatsonConfig config;
    
public:
    bool initialize(const BackendConfig& config) override;
    TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) override;
    // ... implement all virtual methods
};
```

#### 3.2 Add Required Dependencies (Optional Compilation)
```makefile
# In impl/Makefile
ifdef ENABLE_WATSON_BACKEND
    CXXFLAGS += -DENABLE_WATSON_BACKEND
    SOURCES += src/WatsonSTTAdapter.cpp src/WebSocketClient.cpp
    LDFLAGS += -lboost_system -lssl -lcrypto -lcurl
    
    # Download and build websocketpp if needed
    deps/websocketpp:
        ./scripts/setup_websocketpp.sh
endif
```

#### 3.3 Implement Authentication Manager
```cpp
class AuthManager {
public:
    virtual ~AuthManager() = default;
    virtual std::string getAccessToken() = 0;
    virtual bool refreshToken() = 0;
};

class WatsonIAMAuth : public AuthManager {
    // IBM Cloud IAM authentication
};

class APIKeyAuth : public AuthManager {
    // Simple API key authentication
};
```

### Phase 4: Voice Gateway Support (Weeks 9-10)

#### 4.1 Port IBMVoiceGatewaySource
```spl
public composite VoiceGatewaySource(output AudioStream, CallMetadata) {
    param
        expression<rstring> $connectionUrl;
        expression<int32> $port: 9080;
        expression<boolean> $enableCallRecording: false;
        expression<rstring> $recordingPath: "./recordings";
        
    output
        stream<AudioInput> AudioStream;
        stream<CallInfo> CallMetadata;
}

type CallInfo = tuple<
    rstring callId,
    rstring callerNumber,
    rstring calledNumber,
    timestamp startTime,
    timestamp endTime,
    map<rstring,rstring> sipHeaders
>;
```

#### 4.2 Implement Telephony Audio Handler
- Handle µ-law and A-law codecs
- Support RTP packet reassembly
- Implement jitter buffer
- Handle packet loss

### Phase 5: Advanced Features (Weeks 11-12)

#### 5.1 Batch Processing Support
```spl
public composite BatchSTT(input FileList; output TranscriptionResults) {
    param
        expression<rstring> $backend;
        expression<int32> $parallelism: 4;
        expression<boolean> $continueOnError: true;
        
    graph
        // Parallel processing of multiple files
        @parallel(width = $parallelism)
        stream<TranscriptionOutput> Results = UnifiedSTT(FileProcessor(FileList)) {
            param
                backend: $backend;
        }
}
```

#### 5.2 Multi-Language Support
```cpp
class LanguageDetector {
    std::string detectLanguage(const AudioChunk& audio);
    std::vector<LanguageProbability> getLanguageProbabilities(
        const AudioChunk& audio);
};
```

#### 5.3 Results Post-Processing
```spl
public composite TranscriptionPostProcessor(
    input RawTranscription; 
    output ProcessedTranscription) {
    
    param
        expression<boolean> $enableProfanityFilter: false;
        expression<boolean> $enableNumberFormatting: true;
        expression<boolean> $enablePunctuationRestoration: true;
        expression<list<rstring>> $customVocabulary;
}
```

### Phase 6: Testing and Migration Tools (Weeks 13-14)

#### 6.1 Compatibility Testing Suite
```spl
namespace com.teracloud.streamsx.stt.test;

composite BackendComparisonTest {
    // Run same audio through multiple backends
    // Compare results and performance
}

composite MigrationValidationTest {
    // Verify migrated STT Gateway apps work correctly
}
```

#### 6.2 Migration Utilities
```spl
namespace com.teracloud.streamsx.stt.migration;

public composite WatsonToUnifiedMigrator(
    input WatsonSTTStream; 
    output UnifiedSTTStream) {
    // Adapter to help migrate existing apps
}
```

#### 6.3 Performance Benchmarking
- Latency comparison between backends
- Throughput measurements
- Resource utilization metrics
- Accuracy evaluation framework

## Implementation Details

### Directory Structure
```
com.teracloud.streamsx.stt/
├── impl/
│   ├── include/
│   │   ├── backends/
│   │   │   ├── STTBackendAdapter.hpp
│   │   │   ├── NeMoSTTAdapter.hpp
│   │   │   ├── WatsonSTTAdapter.hpp
│   │   │   └── GoogleSTTAdapter.hpp
│   │   ├── auth/
│   │   │   ├── AuthManager.hpp
│   │   │   └── WatsonIAMAuth.hpp
│   │   ├── telephony/
│   │   │   ├── VoiceGatewayHandler.hpp
│   │   │   └── RTPProcessor.hpp
│   │   └── utils/
│   │       ├── WebSocketClient.hpp
│   │       └── AudioCodecs.hpp
│   └── src/
│       └── [corresponding .cpp files]
├── com.teracloud.streamsx.stt/
│   ├── UnifiedSTT/
│   ├── VoiceGatewaySource/
│   └── BatchSTT/
└── samples/
    ├── unified/
    ├── migration/
    └── benchmarks/
```

### Configuration Management

#### Backend Configuration Schema
```json
{
  "backends": {
    "nemo": {
      "enabled": true,
      "default": true,
      "config": {
        "modelPath": "${STT_MODEL_PATH}",
        "provider": "CPU",
        "numThreads": 4
      }
    },
    "watson": {
      "enabled": true,
      "config": {
        "endpoint": "wss://api.us-south.speech-to-text.watson.cloud.ibm.com",
        "apiKey": "${WATSON_API_KEY}",
        "model": "en-US_BroadbandModel"
      }
    }
  }
}
```

### Error Handling Strategy

1. **Backend Fallback**
   ```cpp
   class FallbackStrategy {
       std::vector<std::string> backendPriority;
       bool autoFallback = true;
       int maxRetries = 3;
   };
   ```

2. **Error Categories**
   - Temporary (network, service unavailable)
   - Permanent (invalid credentials, unsupported format)
   - Data-specific (corrupted audio, unsupported codec)

3. **Recovery Actions**
   - Automatic retry with exponential backoff
   - Fallback to alternative backend
   - Partial results on timeout
   - Detailed error reporting

## Migration Guidelines

### For Existing NeMoSTT Users
1. No immediate changes required
2. Gradual migration path available
3. Performance characteristics maintained

### For STT Gateway Users
1. Update operator names:
   - `WatsonSTT` → `UnifiedSTT` with `backend: "watson"`
   - `IBMVoiceGatewaySource` → `VoiceGatewaySource`

2. Update schemas to use common types

3. Configure authentication:
   ```spl
   stream<TranscriptionOutput> Transcripts = UnifiedSTT(Audio) {
       param
           backend: "watson";
           apiKey: getSubmissionTimeValue("WATSON_API_KEY");
           backendConfig: {
               "model": "en-US_BroadbandModel",
               "acousticCustomizationId": "custom_id"
           };
   }
   ```

## Testing Strategy

### Unit Tests
- Backend adapter implementations
- Audio codec conversions
- Authentication flows
- Error handling paths

### Integration Tests
- End-to-end transcription with each backend
- Backend switching during runtime
- Telephony integration
- Batch processing

### Performance Tests
- Latency measurements
- Throughput benchmarks
- Resource utilization
- Scalability limits

### Compatibility Tests
- Existing NeMoSTT applications
- Migrated STT Gateway applications
- Mixed backend scenarios

## Risk Mitigation

### Technical Risks
1. **Dependency Conflicts**
   - Mitigation: Optional compilation, namespace isolation

2. **Performance Degradation**
   - Mitigation: Maintain direct paths for local processing

3. **API Changes**
   - Mitigation: Version adapters, deprecation notices

### Operational Risks
1. **Cloud Service Outages**
   - Mitigation: Local fallback, circuit breakers

2. **Cost Management**
   - Mitigation: Usage tracking, quotas, alerts

## Success Criteria

1. **Functional**
   - All existing NeMoSTT features work unchanged
   - Watson STT backend achieves feature parity
   - Voice Gateway integration functional

2. **Performance**
   - Local processing maintains current speed
   - Cloud backends meet SLA requirements
   - Switching overhead < 100ms

3. **Usability**
   - Clear migration documentation
   - Intuitive configuration
   - Helpful error messages

## Timeline Summary

- **Weeks 1-3**: Foundation and Interface Design
- **Weeks 4-5**: NeMo Backend Refactoring  
- **Weeks 6-8**: Watson STT Integration
- **Weeks 9-10**: Voice Gateway Support
- **Weeks 11-12**: Advanced Features
- **Weeks 13-14**: Testing and Migration Tools

Total estimated effort: 14 weeks with 2-3 developers

## Future Enhancements

1. **Additional Backends**
   - Google Cloud Speech-to-Text
   - Azure Cognitive Services Speech
   - Amazon Transcribe
   - OpenAI Whisper

2. **Advanced Features**
   - Speaker diarization
   - Emotion detection
   - Language identification
   - Custom vocabulary management

3. **Operational Tools**
   - Web-based configuration UI
   - Real-time monitoring dashboard
   - A/B testing framework
   - Cost optimization analyzer

## Conclusion

This migration plan provides a structured approach to incorporating STT Gateway functionality while maintaining the strengths of the current toolkit. The modular architecture ensures future extensibility while the phased approach minimizes risk and allows for incremental delivery of value.