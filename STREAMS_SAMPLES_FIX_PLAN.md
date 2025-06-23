# Streams Sample Programs Fix Plan

## Overview
The existing Streams STT samples need to be updated to work with the NeMo FastConformer model. This document outlines the specific changes needed.

## Current Sample Programs

### 1. OnnxSTTSample (`samples/OnnxSTTSample/sample/OnnxSTT.spl`)
**Current Issues:**
- Uses incorrect model path
- Wrong preprocessing parameters
- Doesn't handle BPE vocabulary
- Missing proper CTC decoding

**Required Changes:**
```spl
// Update model path
expression<rstring> $modelPath : 
    getThisToolkitDir() + "/models/fastconformer_nemo_export/ctc_model.onnx";

// Update vocabulary path  
expression<rstring> $vocabPath :
    getThisToolkitDir() + "/models/fastconformer_nemo_export/tokens.txt";

// Fix audio source parameters
stream<Audio> AudioStream = FileSource() {
    param
        file: getThisToolkitDir() + "/test_data/audio/librispeech_3sec.wav";
        format: wav;
        sampleRate: 16000;  // Must be 16kHz
}

// Update OnnxSTT operator parameters
stream<Transcription> TranscriptionStream = OnnxSTT(AudioStream) {
    param
        modelPath: $modelPath;
        vocabPath: $vocabPath;
        modelType: "nemo_ctc";  // New parameter
        preprocessor: {
            normalize: false,    // Critical: NO normalization
            n_mels: 80,
            n_fft: 512,
            window_size_ms: 25,
            window_stride_ms: 10,
            window_type: "hann",
            dither: 0.00001
        };
}
```

### 2. UnifiedSTTSample (`samples/UnifiedSTTSample/application/UnifiedSTT.spl`)
**Current Issues:**
- Tries to use multiple model types
- Incorrect configuration for NeMo

**Required Changes:**
- Add NeMo model configuration
- Update composite operator to handle BPE tokens
- Fix preprocessing pipeline

### 3. C++ Operator Implementation (`impl/include/OnnxSTTImpl.cpp`)
**Critical Changes Needed:**

1. **Preprocessing Pipeline:**
```cpp
// Add dither to audio
void addDither(std::vector<float>& audio, float dither = 1e-5f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, dither);
    
    for (auto& sample : audio) {
        sample += dist(gen);
    }
}

// Extract log mel features (NO normalization)
std::vector<std::vector<float>> extractLogMelFeatures(
    const std::vector<float>& audio,
    int sample_rate = 16000,
    int n_mels = 80,
    int n_fft = 512,
    int hop_length = 160,
    int win_length = 400) {
    
    // 1. Add dither
    std::vector<float> dithered_audio = audio;
    addDither(dithered_audio);
    
    // 2. Compute mel spectrogram using kaldi-native-fbank
    // 3. Convert to log scale (natural log)
    // 4. NO normalization!
    
    return features;
}
```

2. **CTC Decoding:**
```cpp
std::string greedyCTCDecode(
    const std::vector<std::vector<float>>& log_probs,
    const std::vector<std::string>& vocab,
    int blank_id = 1024) {
    
    std::vector<int> tokens;
    int prev_token = blank_id;
    
    for (const auto& frame : log_probs) {
        int best_token = std::distance(
            frame.begin(), 
            std::max_element(frame.begin(), frame.end())
        );
        
        if (best_token != blank_id && best_token != prev_token) {
            tokens.push_back(best_token);
        }
        prev_token = best_token;
    }
    
    // Handle BPE tokens
    std::string text;
    for (int token : tokens) {
        if (token < vocab.size()) {
            std::string tok = vocab[token];
            if (tok.find("▁") == 0) {
                text += " " + tok.substr(3);  // UTF-8 ▁ is 3 bytes
            } else {
                text += tok;
            }
        }
    }
    
    return text;
}
```

## Testing Plan

### 1. Unit Tests
- Test preprocessing matches Python implementation
- Test CTC decoding with known inputs
- Test BPE token handling

### 2. Integration Tests
```bash
# Compile the toolkit
cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt
make

# Compile sample
cd samples/OnnxSTTSample
make

# Test standalone
./output/bin/standalone

# Submit to Streams
streamtool submitjob output/com.teracloud.streamsx.stt.OnnxSTTSample.sab
```

### 3. Validation Criteria
- [ ] Sample compiles without errors
- [ ] Model loads successfully
- [ ] Audio processing works
- [ ] Produces readable transcription
- [ ] Performance is acceptable (< 0.3 RTF)

## Common Issues and Solutions

### Issue 1: Empty Transcription
**Cause**: Incorrect preprocessing (usually normalization)
**Solution**: Ensure NO normalization is applied to features

### Issue 2: Garbage Output
**Cause**: Wrong vocabulary or token handling
**Solution**: Use correct tokens.txt and handle BPE properly

### Issue 3: Model Loading Errors
**Cause**: Path issues or missing files
**Solution**: Use absolute paths or getThisToolkitDir()

### Issue 4: Performance Issues
**Cause**: Inefficient preprocessing
**Solution**: Use kaldi-native-fbank, optimize buffer sizes

## Implementation Order

1. **Fix C++ preprocessing** (Day 1-2)
   - Update OnnxSTTImpl.cpp
   - Test with standalone C++ program
   - Verify matches Python output

2. **Update SPL operators** (Day 2-3)
   - Add model type parameter
   - Fix preprocessing config
   - Handle BPE vocabulary

3. **Fix sample programs** (Day 3-4)
   - Update OnnxSTTSample
   - Test with real audio
   - Document usage

4. **Performance optimization** (Day 4-5)
   - Profile bottlenecks
   - Optimize feature extraction
   - Add streaming support

## Success Metrics

1. **Functionality**
   - Samples produce correct transcriptions
   - Handle various audio formats
   - Work with streaming data

2. **Performance**
   - Real-time factor < 0.3
   - Memory usage < 500MB
   - CPU usage < 50% single core

3. **Usability**
   - Clear documentation
   - Easy to modify for new models
   - Good error messages