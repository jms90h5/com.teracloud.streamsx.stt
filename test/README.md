# Test Directory - Comprehensive Testing Guide

This directory contains test utilities, verification scripts, and testing documentation for the TeraCloud Streams STT toolkit.

## Overview

The test suite validates the STT toolkit functionality at multiple levels:
- Unit tests for individual components
- Integration tests for the complete pipeline
- Performance benchmarks
- Streaming/real-time processing tests
- SPL operator validation

## Current Test Files

### **Active Test Programs**

#### `test_real_nemo_fixed.cpp`
- **Purpose**: Comprehensive standalone C++ test for NeMo CTC model
- **Features**: Command-line options, WAV file support, performance benchmarking
- **Model**: Uses current `models/fastconformer_ctc_export/model.onnx`
- **Status**: ✅ **Recommended** - Most complete test program

#### `test_nemo_ctc_simple.cpp`
- **Purpose**: Basic functionality test for NeMoCTCImpl class
- **Features**: Simple initialization and transcription test
- **Model**: Uses current `models/fastconformer_ctc_export/model.onnx`
- **Status**: ✅ **Useful** for basic testing

### **Verification Scripts**

#### `verify_nemo_setup.sh`
- **Purpose**: Comprehensive system verification script
- **Checks**: ONNX Runtime, C++ library, models, Python dependencies
- **Status**: ✅ **Updated** with current model paths
- **Usage**: `./verify_nemo_setup.sh`

### **Example Test Templates**

#### `test_audio_transcription_example.cpp`
- **Purpose**: Template for testing with real audio files and expected results
- **Features**: Word accuracy calculation, batch testing support
- **Status**: 📝 **Template** - Requires audio test files

#### `test_streaming_example.cpp`
- **Purpose**: Template for streaming/chunked audio processing
- **Features**: Latency measurement, real-time factor calculation
- **Status**: 📝 **Template** - Shows streaming test structure

## Comprehensive Test Plan

### 1. Component Testing 🔧

#### 1.1 Model Loading and Initialization
- Verify model loads without errors
- Check model metadata (input/output shapes, vocabulary)
- Test with invalid model paths
- Memory usage validation

#### 1.2 Audio Processing
- Test various audio formats (8kHz, 16kHz, 44.1kHz)
- Validate mono/stereo handling
- Test different bit depths (16-bit, 24-bit)
- Edge cases (silence, very short/long audio)

#### 1.3 Feature Extraction
- Validate mel-spectrogram parameters
- Compare with reference Python implementation
- Test windowing and frame overlap
- Verify feature normalization

#### 1.4 Transcription Accuracy
- Test with known audio samples
- Calculate Word Error Rate (WER)
- Validate against expected transcriptions
- Test with various accents and speaking speeds

### 2. Integration Testing 🔗

#### 2.1 End-to-End Pipeline
```bash
# Test complete transcription pipeline
Audio File → Feature Extraction → Model Inference → Text Output
```

#### 2.2 SPL Operator Testing
- FileAudioSource → NeMoSTT → FileSink
- Test with different window configurations
- Validate punctuation handling
- Multi-stream processing

#### 2.3 Streaming Tests
- Real-time audio processing
- Chunk boundary handling
- Partial result generation
- Latency measurements

### 3. Performance Testing 📊

#### 3.1 Benchmarks
- **Latency**: Time from audio input to transcription output
- **Throughput**: Audio hours processed per hour
- **Memory**: Peak memory usage during processing
- **CPU**: Utilization during transcription

#### 3.2 Performance Targets
- Latency: < 50ms for 160ms audio chunk
- Throughput: > 10x real-time on CPU
- Memory: < 500MB for single stream
- Accuracy: > 95% on clean speech

### 4. Robustness Testing 🛡️

#### 4.1 Error Handling
- Invalid audio formats
- Corrupted model files
- Out of memory conditions
- Concurrent access

#### 4.2 Edge Cases
- Empty audio files
- Very noisy audio
- Multiple speakers
- Non-speech audio

## Building and Running Tests

### Prerequisites

1. **Build the toolkit first:**
   ```bash
   cd ../impl && make clean && make
   ```

2. **Export the NeMo model:**
   ```bash
   cd .. && python export_model_ctc_patched.py
   ```

3. **Verify setup:**
   ```bash
   cd test && ./verify_nemo_setup.sh
   ```

### Build Test Programs

#### Simple CTC Test
```bash
cd test
g++ -std=c++14 -O2 -I../impl/include -I../deps/onnxruntime/include \
    test_nemo_ctc_simple.cpp ../impl/lib/libs2t_impl.so \
    -L../deps/onnxruntime/lib -lonnxruntime -ldl \
    -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../deps/onnxruntime/lib' \
    -o test_nemo_ctc_simple

# Run
./test_nemo_ctc_simple
```

#### Comprehensive Fixed Test
```bash
cd test
g++ -std=c++14 -O2 -I../impl/include -I../deps/onnxruntime/include \
    test_real_nemo_fixed.cpp ../impl/lib/libs2t_impl.so \
    -L../deps/onnxruntime/lib -lonnxruntime -ldl \
    -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../deps/onnxruntime/lib' \
    -o test_real_nemo_fixed

# Run with options
./test_real_nemo_fixed --help
./test_real_nemo_fixed --verbose
```

### Library Path Requirements

**Important**: Always set library paths when running tests:
```bash
export LD_LIBRARY_PATH=$PWD/../deps/onnxruntime/lib:$PWD/../impl/lib:$LD_LIBRARY_PATH
```

Or use the rpath settings shown in the build commands above.

## Test Data Organization

### Required Test Data Structure
```
test_data/
├── audio/
│   ├── transcription_tests/
│   │   ├── librispeech-1995-1837-0001.wav     # Known transcription
│   │   ├── digits-zero-to-nine.wav            # Number sequence
│   │   ├── weather-forecast.wav               # Domain-specific
│   │   └── expected_results.json              # Expected outputs
│   ├── format_tests/
│   │   ├── audio_8khz.wav
│   │   ├── audio_44khz.wav
│   │   ├── audio_stereo.wav
│   │   └── audio_24bit.wav
│   ├── edge_cases/
│   │   ├── silence_2sec.wav
│   │   ├── white_noise.wav
│   │   ├── very_short_100ms.wav
│   │   └── very_long_10min.wav
│   └── streaming_tests/
│       └── continuous_speech_1min.wav
```

## Expected Test Results

### Successful Model Test
```
=== Simple NeMo CTC Test ===
Initializing model...

Model info:
Model loaded successfully
Input shape: [1, 80, -1]
Output shape: [1, -1, 1025]
Vocabulary size: 1025

Transcribing 2 seconds of silence...
Result: ''
```

### Performance Benchmark Results
```
=== Performance Test Results ===
Audio duration: 60 seconds
Processing time: 4.2 seconds
Real-time factor: 14.3x
Average latency: 12.5ms
P99 latency: 23.4ms
Peak memory: 387MB
```

## Common Issues and Solutions

### 1. Model Loading Failures
- **Issue**: "Failed to initialize model"
- **Check**: Model path exists: `../models/fastconformer_ctc_export/model.onnx`
- **Fix**: Run `python export_model_ctc_patched.py` to export model

### 2. Library Loading Errors
- **Issue**: "libonnxruntime.so: cannot open shared object file"
- **Check**: `ls ../deps/onnxruntime/lib/libonnxruntime.so`
- **Fix**: Set LD_LIBRARY_PATH or use rpath in build

### 3. Empty Transcriptions
- **Issue**: No text output from audio
- **Check**: Audio format is 16kHz, 16-bit, mono
- **Check**: Model and vocabulary files match
- **Fix**: Verify audio contains speech, not silence

### 4. Poor Accuracy
- **Issue**: Transcription accuracy below expected
- **Check**: Audio quality and noise level
- **Check**: Model was exported correctly
- **Fix**: Use clean audio, verify model export process

## Test Automation

### Future: Automated Test Suite
```makefile
# test/Makefile (planned)
TESTS = test_nemo_ctc_simple test_real_nemo_fixed \
        test_audio_transcription test_streaming \
        test_performance test_error_handling

all: $(TESTS)
run-all: $(TESTS)
	./run_all_tests.sh
```

### Future: CI/CD Integration
- GitHub Actions workflow for automated testing
- Performance regression detection
- Accuracy validation on test sets

## Development Notes

- Tests focus on validating the current working implementation
- All paths are relative to the test directory
- C++ tests validate the core library functionality
- SPL tests validate the Streams operator integration
- Performance tests ensure real-time capability

## Quick Verification

Run this command to verify everything is working:
```bash
cd test && ./verify_nemo_setup.sh && echo "✅ Ready for testing"
```

## Archived Development Tests

Historical test files from the development phase have been moved to `../archive/test_scripts/`. These include debugging scripts and tests for issues that have been resolved. See `../archive/test_scripts/README.md` for details.

## Contributing New Tests

When adding new tests:
1. Follow the existing naming convention: `test_<functionality>.cpp`
2. Include clear documentation in the test file
3. Add build instructions to this README
4. Ensure tests are self-contained and repeatable
5. Include expected results in comments

## Test Priority for New Users

1. Run `verify_nemo_setup.sh` first
2. Build and run `test_nemo_ctc_simple` for basic validation
3. Use `test_real_nemo_fixed` for comprehensive testing
4. Implement custom tests using the provided templates