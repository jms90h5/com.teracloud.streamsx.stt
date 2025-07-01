# STT Gateway Migration Plan v2.0

## Executive Summary

This document outlines an updated plan for migrating IBM STT Gateway functionality into the com.teracloud.streamsx.stt toolkit. This revision incorporates the 2-channel audio support already implemented and provides a comprehensive roadmap for supporting multiple Speech-to-Text backends including local NeMo models and cloud-based Watson STT services.

## Current State (What's Already Implemented)

### 2-Channel Audio Support ✅
- **AudioChannelSplitter** operator for stereo separation
- **StereoFileAudioSource** composite for reading 2-channel files
- **ChannelAudioStream** and **ChannelMetadata** types
- Support for telephony codecs (G.711 µ-law/A-law)
- Channel role assignment (caller/agent)
- Tested with 289-second stereo telephony WAV file

### Existing STT Capabilities ✅
- **NeMoSTT** operator using ONNX Runtime
- **OnnxSTT** operator for advanced configuration
- FastConformer model support
- Real-time throttling for simulated playback
- Basic file audio source

## Goals and Objectives

### Primary Goals
1. Create a unified STT interface supporting multiple backends
2. Leverage existing 2-channel capabilities for telephony scenarios
3. Enable seamless switching between local and cloud STT providers
4. Maintain backward compatibility with existing operators
5. Provide Voice Gateway integration for real-time telephony

### Design Principles
- **Build on Existing**: Leverage current 2-channel and NeMo infrastructure
- **Modularity**: Each STT backend as a pluggable component
- **Minimal Dependencies**: Core functionality with optional features
- **Performance**: Maintain current speed advantages of local processing
- **Telephony-First**: Design with call center use cases in mind

## Updated Architecture

### Proposed Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                    │
│         (User SPL Applications)                      │
├─────────────────────────────────────────────────────┤
│           Telephony Integration Layer                │
│  (Voice Gateway, 2-Channel Processing, Call Meta)   │
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
│  (Audio Channel Split, Codecs, Feature Extraction)  │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Unified STT Operator** (`UnifiedSTT`)
   - Single operator interface for all backends
   - Supports both mono and per-channel processing
   - Runtime backend selection
   - Common input/output schemas

2. **Backend Adapters**
   - NeMoSTTAdapter (refactor existing)
   - WatsonSTTAdapter (new)
   - GoogleSTTAdapter (future)
   - AzureSTTAdapter (future)

3. **Audio Source Operators** (Enhanced)
   - FileAudioSource (existing)
   - StereoFileAudioSource (existing)
   - VoiceGatewaySource (new, from STT Gateway)
   - TelephonyStreamSource (new)

4. **Telephony Integration**
   - AudioChannelSplitter (existing)
   - VoiceGatewayHandler (new)
   - CallMetadataEnricher (new)
   - TranscriptionMerger (new)

## Detailed Migration Phases

### Phase 1: Foundation and Interface Design (Week 1-2)

#### 1.1 Extend Common Schemas for Telephony

```spl
// Extend existing ChannelAudioStream for unified processing
type UnifiedAudioInput = tuple<
    blob audioData,
    uint64 audioTimestamp,    // renamed from timestamp
    rstring encoding,          // "pcm16", "pcm8", "ulaw", "alaw", "opus"
    int32 sampleRate,
    int32 channels,            // 1 for mono, 2 for stereo
    rstring languageCode,      // "en-US", "es-ES", etc.
    ChannelMetadata channelInfo,  // existing type
    map<rstring,rstring> metadata
>;

// Enhanced transcription output with channel support
type UnifiedTranscriptionOutput = tuple<
    rstring text,
    float64 confidence,
    boolean isFinal,
    list<WordTiming> wordTimings,
    ChannelMetadata channelInfo,   // which channel this came from
    uint64 startTime,
    uint64 endTime,
    rstring backend,               // which backend produced this
    map<rstring,rstring> metadata
>;

// Call metadata for telephony scenarios
type CallMetadata = tuple<
    rstring callId,
    rstring callerNumber,
    rstring calledNumber,
    uint64 startTime,
    uint64 endTime,
    rstring direction,             // "inbound" or "outbound"
    map<rstring,rstring> sipHeaders
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
        expression<boolean> $enableSpeakerLabels: false;  // for diarization
        expression<int32> $maxAlternatives: 1;
        expression<map<rstring,rstring>> $backendConfig;
        
    type
        static InputType = UnifiedAudioInput;
        static OutputType = UnifiedTranscriptionOutput;
}
```

### Phase 2: Refactor NeMo Backend (Week 3-4)

#### 2.1 Create NeMoSTTAdapter

```cpp
// impl/include/backends/NeMoSTTAdapter.hpp
#ifndef NEMO_STT_ADAPTER_HPP
#define NEMO_STT_ADAPTER_HPP

#include "STTBackendAdapter.hpp"
#include "NeMoCTCModel.hpp"
#include <memory>

namespace com::teracloud::streamsx::stt {

class NeMoSTTAdapter : public STTBackendAdapter {
private:
    std::unique_ptr<NeMoCTCModel> model_;
    NeMoConfig config_;
    ChannelMetadata currentChannel_;
    
public:
    bool initialize(const BackendConfig& config) override;
    
    TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) override;
    
    TranscriptionResult finalize() override;
    
    BackendCapabilities getCapabilities() const override {
        return {
            .supportsStreaming = true,
            .supportsWordTimings = false,
            .supportsSpeakerLabels = false,
            .supportedLanguages = {"en-US"},
            .supportedEncodings = {"pcm16"},
            .requiresSampleRate = 16000
        };
    }
    
    bool isHealthy() const override;
};

} // namespace

#endif
```

#### 2.2 Adapter Wrapper for Existing NeMoSTT

```spl
// Maintain backward compatibility
public composite NeMoSTT(input AudioStream; output TranscriptionStream) {
    param
        expression<rstring> $modelPath;
        expression<rstring> $vocabPath;
        expression<int32> $sampleRate: 16000;
        
    graph
        // Convert to UnifiedAudioInput format
        stream<UnifiedAudioInput> UnifiedAudio = Custom(AudioStream) {
            logic
                onTuple AudioStream: {
                    // Map existing schema to unified schema
                    UnifiedAudioInput unified = {};
                    unified.audioData = AudioStream.audioData;
                    unified.audioTimestamp = AudioStream.audioTimestamp;
                    unified.encoding = "pcm16";
                    unified.sampleRate = $sampleRate;
                    unified.channels = 1;
                    unified.languageCode = "en-US";
                    submit(unified, UnifiedAudio);
                }
        }
        
        // Use UnifiedSTT with NeMo backend
        stream<UnifiedTranscriptionOutput> UnifiedResult = UnifiedSTT(UnifiedAudio) {
            param
                backend: "nemo";
                modelPath: $modelPath;
                backendConfig: {
                    "vocabPath": $vocabPath,
                    "provider": "CPU"
                };
        }
        
        // Convert back to original schema
        stream<TranscriptionStream> TranscriptionStream = Custom(UnifiedResult) {
            // Map unified output back to original schema
        }
}
```

### Phase 3: Watson STT Integration (Week 5-7)

#### 3.1 Implement WatsonSTTAdapter

```cpp
// impl/include/backends/WatsonSTTAdapter.hpp
#ifndef WATSON_STT_ADAPTER_HPP
#define WATSON_STT_ADAPTER_HPP

#include "STTBackendAdapter.hpp"
#include "WebSocketClient.hpp"
#include "AuthManager.hpp"
#include <memory>
#include <queue>

namespace com::teracloud::streamsx::stt {

class WatsonSTTAdapter : public STTBackendAdapter {
private:
    std::unique_ptr<WebSocketClient> wsClient_;
    std::unique_ptr<AuthManager> authManager_;
    WatsonConfig config_;
    ChannelMetadata currentChannel_;
    std::queue<TranscriptionResult> resultQueue_;
    
    // WebSocket callbacks
    void onMessage(const std::string& message);
    void onError(const std::string& error);
    void onClose();
    
public:
    bool initialize(const BackendConfig& config) override;
    
    TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) override;
    
    TranscriptionResult finalize() override;
    
    BackendCapabilities getCapabilities() const override {
        return {
            .supportsStreaming = true,
            .supportsWordTimings = true,
            .supportsSpeakerLabels = true,
            .supportedLanguages = {"en-US", "es-ES", "fr-FR", "de-DE", "ja-JP"},
            .supportedEncodings = {"pcm16", "opus", "mp3"},
            .requiresSampleRate = 0  // flexible
        };
    }
    
    bool isHealthy() const override;
};

} // namespace

#endif
```

#### 3.2 Watson-Specific Configuration

```cpp
struct WatsonConfig : public BackendConfig {
    std::string apiKey;
    std::string serviceUrl;
    std::string model = "en-US_BroadbandModel";
    bool smartFormatting = true;
    bool profanityFilter = false;
    float speechDetectorSensitivity = 0.5f;
    float backgroundAudioSuppression = 0.5f;
    std::string acousticCustomizationId;
    std::string languageCustomizationId;
};
```

### Phase 4: Voice Gateway Integration (Week 8-9)

#### 4.1 Port VoiceGatewaySource with 2-Channel Support

```spl
public composite VoiceGatewaySource(
    output CallerAudio, AgentAudio, CallMeta) {
    
    param
        expression<rstring> $connectionUrl;
        expression<int32> $port: 9080;
        expression<boolean> $enableCallRecording: false;
        expression<rstring> $recordingPath: "./recordings";
        expression<rstring> $callerRole: "caller";
        expression<rstring> $agentRole: "agent";
        
    type
        static CallerAudioType = ChannelAudioStream;
        static AgentAudioType = ChannelAudioStream;
        static CallMetaType = CallMetadata;
        
    graph
        // WebSocket connection to Voice Gateway
        stream<blob rawAudio, CallMetadata callInfo> VGStream = 
            VoiceGatewayHandler() {
                param
                    url: $connectionUrl;
                    port: $port;
            }
        
        // Split 2-channel telephony audio
        (stream<CallerAudioType> CallerAudio;
         stream<AgentAudioType> AgentAudio) = 
            AudioChannelSplitter(VGStream) {
                param
                    stereoFormat: "interleaved";
                    encoding: "ulaw";  // typical for telephony
                    leftChannelRole: $callerRole;
                    rightChannelRole: $agentRole;
                    sampleRate: 8000;
                    targetSampleRate: 16000;  // upsample for STT
            }
        
        // Pass through call metadata
        stream<CallMetaType> CallMeta = Custom(VGStream) {
            logic
                onTuple VGStream: {
                    submit(callInfo, CallMeta);
                }
        }
}
```

#### 4.2 Real-time Telephony Transcription

```spl
composite TelephonyTranscription {
    param
        expression<rstring> $voiceGatewayUrl;
        expression<rstring> $sttBackend: "nemo";
        
    graph
        // Connect to Voice Gateway
        (stream<ChannelAudioStream> CallerAudio;
         stream<ChannelAudioStream> AgentAudio;
         stream<CallMetadata> CallInfo) = VoiceGatewaySource() {
            param
                connectionUrl: $voiceGatewayUrl;
        }
        
        // Transcribe caller channel
        stream<UnifiedTranscriptionOutput> CallerText = 
            UnifiedSTT(CallerAudio as UnifiedAudioInput) {
                param
                    backend: $sttBackend;
            }
        
        // Transcribe agent channel  
        stream<UnifiedTranscriptionOutput> AgentText = 
            UnifiedSTT(AgentAudio as UnifiedAudioInput) {
                param
                    backend: $sttBackend;
            }
        
        // Merge transcriptions with call context
        stream<CallTranscription> MergedTranscript = 
            TranscriptionMerger(CallerText, AgentText, CallInfo) {
                param
                    mergeStrategy: "timestamp";
                    includeChannelLabels: true;
            }
}
```

### Phase 5: Advanced Features (Week 10-11)

#### 5.1 Multi-Backend Fallback

```spl
public composite FallbackSTT(input AudioStream; output TranscriptionStream) {
    param
        expression<list<rstring>> $backendPriority: ["nemo", "watson"];
        expression<float64> $timeout: 5.0;
        
    graph
        // Try primary backend
        stream<UnifiedTranscriptionOutput> PrimaryResult = 
            UnifiedSTT(AudioStream) {
                param
                    backend: $backendPriority[0];
            }
        
        // Monitor for failures
        stream<UnifiedAudioInput> FallbackAudio = 
            Custom(AudioStream as I, PrimaryResult as P) {
                logic
                    state: {
                        mutable boolean primaryFailed = false;
                    }
                    onTuple I: {
                        // Buffer audio for potential fallback
                    }
                    onPunct P: {
                        if (currentPunct() == Sys.WindowMarker) {
                            // Check for timeout or error
                        }
                    }
            }
        
        // Use fallback backend if needed
        stream<UnifiedTranscriptionOutput> FallbackResult = 
            UnifiedSTT(FallbackAudio) {
                param
                    backend: $backendPriority[1];
            }
}
```

#### 5.2 Batch Processing with Channel Support

```spl
composite BatchTelephonySTT {
    param
        expression<rstring> $inputDir;
        expression<rstring> $outputDir;
        expression<rstring> $backend: "nemo";
        expression<int32> $parallelism: 4;
        
    graph
        // Scan for stereo telephony recordings
        stream<rstring filename> FileList = DirectoryScan() {
            param
                directory: $inputDir;
                pattern: "*.wav";
        }
        
        // Process files in parallel
        @parallel(width = $parallelism)
        stream<CallTranscription> Results = Custom(FileList) {
            logic
                onTuple FileList: {
                    // Read stereo file
                    // Split channels
                    // Transcribe both channels
                    // Merge results
                    // Write output
                }
        }
}
```

### Phase 6: Testing and Migration (Week 12-13)

#### 6.1 Comprehensive Test Suite

```spl
namespace com.teracloud.streamsx.stt.test;

composite UnifiedSTTTestSuite {
    graph
        // Test all backends with same audio
        () as TestNeMo = TestBackend() {
            param backend: "nemo";
        }
        
        () as TestWatson = TestBackend() {
            param backend: "watson";
        }
        
        // Test 2-channel processing
        () as TestStereo = Test2ChannelProcessing() {
            param
                testFile: "samples/audio/telephony-stereo.wav";
        }
        
        // Test failover scenarios
        () as TestFailover = TestBackendFailover() {
            param
                primaryBackend: "watson";
                fallbackBackend: "nemo";
        }
        
        // Performance benchmarks
        () as Benchmark = PerformanceBenchmark() {
            param
                backends: ["nemo", "watson"];
                testDuration: 300.0;  // 5 minutes
        }
}
```

#### 6.2 Migration Helper

```spl
public composite STTGatewayMigrationHelper(
    input WatsonGatewayStream; 
    output UnifiedStream) {
    
    // Automatically convert STT Gateway schemas to unified format
    // Handle authentication migration
    // Map gateway-specific parameters
}
```

## Implementation Timeline

### Immediate Actions (This Week)
1. Create unified schemas building on existing types
2. Design backend adapter interface
3. Set up project structure for new components

### Short Term (Next 2-3 Weeks)  
1. Implement UnifiedSTT operator shell
2. Create NeMoSTTAdapter wrapping existing code
3. Begin WatsonSTTAdapter implementation

### Medium Term (Next 4-6 Weeks)
1. Complete Watson integration with WebSocket support
2. Port Voice Gateway functionality
3. Implement telephony-specific features

### Long Term (Next 2-3 Months)
1. Add additional backends (Google, Azure)
2. Performance optimization
3. Production hardening
4. Comprehensive documentation

## Risk Mitigation

### Technical Risks
1. **WebSocket Implementation Complexity**
   - Mitigation: Use proven libraries (websocketpp, boost::beast)
   - Consider REST API fallback for Watson

2. **Performance Impact of Abstraction**
   - Mitigation: Direct path for NeMo, minimal overhead
   - Benchmark continuously during development

3. **2-Channel Synchronization**
   - Mitigation: Already solved with AudioChannelSplitter
   - Leverage existing timestamp handling

### Operational Risks
1. **Breaking Changes**
   - Mitigation: Maintain backward compatibility layers
   - Provide migration tools and clear documentation

2. **Cloud Service Dependencies**
   - Mitigation: Robust error handling and fallback
   - Local caching where appropriate

## Success Criteria

1. **Functional**
   - All existing applications continue to work
   - Watson STT achieves feature parity with gateway
   - 2-channel telephony transcription works seamlessly

2. **Performance**
   - NeMo backend maintains current speed
   - Watson backend meets real-time requirements
   - Channel splitting adds <5ms overhead

3. **Usability**
   - Single unified interface for all backends
   - Clear migration path from STT Gateway
   - Comprehensive examples for all scenarios

## Next Steps

1. Review and approve this updated plan
2. Create feature branch for development
3. Begin implementation with unified schemas
4. Set up continuous integration for new components
5. Schedule regular progress reviews

## Conclusion

This updated migration plan leverages the significant progress already made with 2-channel audio support and provides a clear path to incorporating STT Gateway functionality. The modular architecture ensures we can deliver value incrementally while maintaining the stability and performance of existing features.