# UnifiedSTT Operator Guide

## Overview

The UnifiedSTT operator provides a single, consistent interface for multiple speech-to-text backends. It currently supports:

- **NeMo** - Local NVIDIA NeMo models using ONNX Runtime (implemented)
- **Watson** - IBM Watson Speech to Text cloud service (in progress)
- **Google** - Google Cloud Speech-to-Text (planned)
- **Azure** - Azure Cognitive Services Speech (planned)

## Key Features

- **Runtime Backend Selection** - Choose STT backend at submission time
- **Automatic Fallback** - Switch to backup backend on failures
- **2-Channel Support** - Built-in support for telephony audio
- **Consistent Output** - Same schema regardless of backend
- **Backend-Specific Options** - Pass custom configuration to each backend

## Usage

### Basic Example - NeMo Backend

```spl
use com.teracloud.streamsx.stt::*;

composite UnifiedSTTExample {
    graph
        // Read audio file
        stream<UnifiedAudioInput> Audio = FileSource() {
            param
                file: "speech.wav";
                format: block;
                blockSize: 8000u;
            output Audio:
                audioData = Block,
                audioTimestamp = TupleNumber() * 500ul,  // 500ms chunks
                encoding = "pcm16",
                sampleRate = 16000,
                channels = 1,
                bitsPerSample = 16,
                languageCode = "en-US";
        }
        
        // Transcribe using NeMo
        stream<UnifiedTranscriptionOutput> Transcript = UnifiedSTT(Audio) {
            param
                backend: "nemo";
                modelPath: getEnvironmentVariable("NEMO_MODEL_PATH");
                vocabPath: getEnvironmentVariable("NEMO_VOCAB_PATH");
        }
        
        // Display results
        () as Display = Custom(Transcript) {
            logic
                onTuple Transcript: {
                    printStringLn("Text: " + text);
                    printStringLn("Confidence: " + (rstring)confidence);
                    printStringLn("Backend: " + backend);
                }
        }
}
```

### 2-Channel Telephony Example

```spl
composite TelephonyTranscription {
    param
        expression<rstring> $audioFile;
        
    graph
        // Read stereo telephony file
        (stream<ChannelAudioStream> CallerAudio;
         stream<ChannelAudioStream> AgentAudio) = StereoFileAudioSource() {
            param
                filename: $audioFile;
                encoding: "ulaw";
                sampleRate: 8000;
        }
        
        // Convert to UnifiedAudioInput for caller
        stream<UnifiedAudioInput> CallerUnified = Custom(CallerAudio) {
            logic
                onTuple CallerAudio: {
                    UnifiedAudioInput unified = {};
                    unified.audioData = CallerAudio.audioData;
                    unified.audioTimestamp = CallerAudio.audioTimestamp;
                    unified.encoding = "pcm16";  // Already converted by splitter
                    unified.sampleRate = 16000;   // Upsampled
                    unified.channels = 1;
                    unified.bitsPerSample = 16;
                    unified.languageCode = "en-US";
                    unified.channelInfo = CallerAudio.channelInfo;
                    submit(unified, CallerUnified);
                }
        }
        
        // Transcribe caller channel
        stream<UnifiedTranscriptionOutput> CallerTranscript = UnifiedSTT(CallerUnified) {
            param
                backend: "nemo";
                modelPath: getEnvironmentVariable("NEMO_MODEL_PATH");
                vocabPath: getEnvironmentVariable("NEMO_VOCAB_PATH");
        }
}
```

### Fallback Configuration

```spl
stream<UnifiedTranscriptionOutput> Transcript = UnifiedSTT(Audio) {
    param
        // Primary backend (cloud)
        backend: "watson";
        apiKey: getSubmissionTimeValue("WATSON_API_KEY");
        apiEndpoint: "wss://api.us-south.speech-to-text.watson.cloud.ibm.com";
        
        // Fallback to local if Watson fails
        fallbackBackend: "nemo";
        
        // Backend-specific configuration
        backendConfig: {
            "model": "en-US_BroadbandModel",    // Watson model
            "modelPath": "/opt/models/nemo.onnx" // NeMo fallback
        };
        
        // Timeout for cloud backends
        timeout: 5.0;
}
```

## Parameters

### Required Parameters

- **backend** (rstring) - Primary STT backend to use ("nemo", "watson", etc.)

### Backend-Specific Parameters

#### NeMo Backend
- **modelPath** (rstring) - Path to ONNX model file
- **vocabPath** (rstring) - Path to vocabulary file

#### Watson Backend (when implemented)
- **apiEndpoint** (rstring) - Watson STT endpoint URL
- **apiKey** (rstring) - Watson API key
- **region** (rstring) - Cloud region

### Optional Parameters

- **languageCode** (rstring) - BCP-47 language code (default: "en-US")
- **enableWordTimings** (boolean) - Enable word-level timings (default: false)
- **enablePunctuation** (boolean) - Enable punctuation (default: true)
- **enableSpeakerLabels** (boolean) - Enable diarization (default: false)
- **maxAlternatives** (int32) - Max alternative transcriptions (default: 1)
- **backendConfig** (map<rstring,rstring>) - Backend-specific options
- **fallbackBackend** (rstring) - Backup backend on failure
- **timeout** (float64) - Timeout in seconds for cloud backends

## Input Schema

The operator expects `UnifiedAudioInput` tuples:

```spl
type UnifiedAudioInput = tuple<
    blob audioData,                    // Audio data
    uint64 audioTimestamp,             // Timestamp in ms
    rstring encoding,                  // Audio encoding
    int32 sampleRate,                  // Sample rate in Hz
    int32 channels,                    // 1 or 2
    int32 bitsPerSample,              // Bits per sample
    rstring languageCode,             // Language code
    ChannelMetadata channelInfo,      // Channel info
    map<rstring,rstring> metadata     // Extra metadata
>;
```

## Output Schema

The operator produces `UnifiedTranscriptionOutput` tuples:

```spl
type UnifiedTranscriptionOutput = tuple<
    rstring text,                      // Transcribed text
    float64 confidence,                // Confidence score
    boolean isFinal,                   // Final vs interim
    list<WordTiming> wordTimings,      // Word timings
    list<SpeakerInfo> speakers,        // Speaker labels
    ChannelMetadata channelInfo,       // Channel info
    uint64 startTime,                  // Start time
    uint64 endTime,                    // End time
    rstring backend,                   // Which backend used
    rstring languageCode,              // Language detected/used
    map<rstring,rstring> metadata,     // Backend metadata
    list<rstring> alternatives         // Alternative transcriptions
>;
```

## Backend Capabilities

### NeMo
- ✅ Streaming transcription
- ❌ Word timings (not yet implemented)
- ❌ Speaker diarization
- ✅ Custom models
- ✅ Languages: en-US, en-GB, en-IN, en-AU
- ✅ Encoding: pcm16
- ✅ Sample rate: 16kHz (fixed)

### Watson (planned)
- ✅ Streaming transcription
- ✅ Word timings
- ✅ Speaker diarization
- ✅ Custom models
- ✅ Multiple languages
- ✅ Multiple encodings
- ✅ Flexible sample rates

## Error Handling

The operator provides robust error handling:

1. **Primary Backend Failure** - Automatically switches to fallback
2. **Network Errors** - Retries with exponential backoff
3. **Invalid Audio** - Logs error and continues
4. **Authentication Failures** - Reports clear error messages

## Performance Considerations

- **Local backends (NeMo)** - Low latency, no network overhead
- **Cloud backends** - Higher latency, requires internet
- **Fallback switching** - Adds ~100ms overhead
- **2-channel processing** - Process channels in parallel

## Migration from Existing Operators

### From NeMoSTT

```spl
// Old
stream<rstring transcription> Result = NeMoSTT(Audio) {
    param
        modelPath: "/opt/models/nemo.onnx";
        vocabPath: "/opt/models/tokens.txt";
}

// New
stream<UnifiedTranscriptionOutput> Result = UnifiedSTT(Audio) {
    param
        backend: "nemo";
        modelPath: "/opt/models/nemo.onnx";
        vocabPath: "/opt/models/tokens.txt";
}
```

### From Watson STT Gateway

```spl
// Old (STT Gateway)
stream<WatsonTranscription> Result = WatsonSTT(Audio) {
    param
        apiKey: getApplicationConfiguration("watson.apikey");
        model: "en-US_BroadbandModel";
}

// New
stream<UnifiedTranscriptionOutput> Result = UnifiedSTT(Audio) {
    param
        backend: "watson";
        apiKey: getApplicationConfiguration("watson.apikey");
        backendConfig: {
            "model": "en-US_BroadbandModel"
        };
}
```

## Best Practices

1. **Always specify fallback** for production cloud deployments
2. **Use local backends** for low-latency requirements
3. **Process channels separately** for telephony audio
4. **Monitor backend health** using getStatus() in logs
5. **Set appropriate timeouts** for cloud backends

## Troubleshooting

### "Backend not available"
- Check backend is compiled in (see Makefile)
- Verify required dependencies installed

### "Model initialization failed"
- Check model file paths are correct
- Verify model format matches backend

### "Authentication failed"
- Check API keys are valid
- Verify endpoint URLs are correct

### Poor transcription quality
- Ensure audio is correct sample rate
- Check audio encoding matches specification
- Verify language code is appropriate