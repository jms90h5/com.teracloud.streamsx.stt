# Comprehensive Test Plan for STT Toolkit

## Current Test Coverage ✅

1. **Basic Initialization** - `test_nemo_ctc_simple.cpp`
2. **Command-line Testing** - `test_real_nemo_fixed.cpp`
3. **Setup Verification** - `verify_nemo_setup.sh`

## Missing Test Coverage 🚧

### Priority 1: Core Functionality Tests

#### 1.1 Real Audio Transcription Tests
```cpp
// test_audio_transcription.cpp
// Test with known audio files and expected results
struct TestCase {
    std::string audio_file;
    std::string expected_text;
    float min_accuracy;  // e.g., 0.95 for 95% accuracy
};

std::vector<TestCase> test_cases = {
    {"librispeech-1837-0001.wav", "he hoped there would be stew for dinner", 0.95},
    {"digits-zero-to-nine.wav", "zero one two three four five six seven eight nine", 0.98},
    {"silence-2sec.wav", "", 1.0}
};
```

#### 1.2 Streaming/Chunked Processing Test
```cpp
// test_streaming.cpp
// Test real-time streaming with chunks
class StreamingTest {
    void testChunkBoundaries();
    void testPartialResults();
    void testVariableChunkSizes();
    void testLatency();
};
```

#### 1.3 Performance Benchmark Suite
```cpp
// test_performance.cpp
struct PerformanceMetrics {
    double avg_latency_ms;
    double p99_latency_ms;
    double throughput_fps;  // frames per second
    size_t peak_memory_mb;
};
```

### Priority 2: Robustness Tests

#### 2.1 Error Handling Tests
```cpp
// test_error_handling.cpp
void testInvalidModelPath();
void testCorruptedModel();
void testInvalidAudioFormat();
void testOutOfMemory();
void testConcurrentAccess();
```

#### 2.2 Audio Format Tests
```cpp
// test_audio_formats.cpp
void test8kHzAudio();
void test44kHzAudio();
void test24BitAudio();
void testStereoToMono();
void testVariableBitrate();
```

### Priority 3: Integration Tests

#### 3.1 SPL Operator Tests
```bash
# test_spl_integration.sh
# Build and test actual SPL applications
sc -M TestNeMoOperator -t .. --output-directory output
streamtool submitjob output/TestNeMoOperator.sab
```

#### 3.2 End-to-End Pipeline Test
```cpp
// test_pipeline.cpp
void testCompleteTranscriptionPipeline();
void testMultiStreamProcessing();
void testDynamicModelLoading();
```

## Test Data Requirements 📁

### Audio Test Files Needed:
```
test_data/
├── audio/
│   ├── transcription_tests/
│   │   ├── librispeech-1837-0001.wav     # "he hoped there would be stew for dinner"
│   │   ├── digits-zero-to-nine.wav       # "zero one two three four five six seven eight nine"
│   │   ├── weather-forecast.wav          # Known weather forecast text
│   │   └── expected_results.json         # Expected transcriptions
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

## Automated Test Framework 🤖

### Makefile for All Tests
```makefile
# test/Makefile
TESTS = test_nemo_ctc_simple test_real_nemo_fixed test_audio_transcription \
        test_streaming test_performance test_error_handling

all: $(TESTS)

test_%: test_%.cpp
	g++ -std=c++14 -O2 -I../impl/include -I../deps/onnxruntime/include \
	    $< ../impl/lib/libs2t_impl.so \
	    -L../deps/onnxruntime/lib -lonnxruntime -ldl \
	    -Wl,-rpath,'$$ORIGIN/../impl/lib' \
	    -Wl,-rpath,'$$ORIGIN/../deps/onnxruntime/lib' \
	    -o $@

run-all: $(TESTS)
	./run_all_tests.sh

clean:
	rm -f $(TESTS) test_results.xml
```

### Test Runner Script
```bash
#!/bin/bash
# run_all_tests.sh
FAILED=0
PASSED=0

for test in test_*; do
    if [[ -x "$test" && "$test" != *.cpp ]]; then
        echo "Running $test..."
        if ./"$test"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    fi
done

echo "Test Results: $PASSED passed, $FAILED failed"
exit $FAILED
```

## Performance Baselines 📊

### Expected Performance Metrics:
- **Latency**: < 50ms for 160ms audio chunk
- **Throughput**: > 10x real-time on CPU
- **Memory**: < 500MB for single stream
- **Accuracy**: > 95% on LibriSpeech test-clean

## CI/CD Integration 🔄

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup dependencies
        run: |
          ./setup_onnx_runtime.sh
          cd impl && make
      - name: Run tests
        run: |
          cd test
          make all
          make run-all
```

## Test Documentation Template 📝

Each test should include:
```cpp
/**
 * Test: [Test Name]
 * Purpose: [What this test validates]
 * Requirements: [Model files, test data needed]
 * Expected Result: [What constitutes success]
 * Performance Target: [If applicable]
 */
```

## Priority Implementation Order 🎯

1. **Week 1**: Real audio transcription tests with expected results
2. **Week 2**: Streaming/chunked processing tests
3. **Week 3**: Performance benchmarks and error handling
4. **Week 4**: Integration tests and CI/CD setup

## Success Criteria ✅

- All tests pass on clean checkout
- Performance meets or exceeds baselines
- Error cases handled gracefully
- CI/CD runs on every commit
- Test coverage > 80% of core functionality