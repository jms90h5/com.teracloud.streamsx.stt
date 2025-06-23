# Systematic Testing Plan for com.teracloud.streamsx.stt
**Created**: 2025-01-20  
**Purpose**: Track systematic testing of the toolkit to verify actual functionality

## Executive Summary
**Status**: Root cause identified! The NeMoSTT operator requires WindowMarker punctuation to trigger transcription, but FileAudioSource never sends punctuation. This is why transcription files are empty despite the model loading successfully and audio chunks being processed.

## Critical Context from Previous Sessions
- Feature processing was identified as problematic
- Mock data may have been used instead of real audio processing  
- Model was reportedly working correctly but not receiving proper input
- Empty transcription files indicate processing pipeline issues

## New Findings from Code Analysis (2025-01-20)
- **NO MOCK DATA FOUND**: The implementation uses real audio processing via KaldiFbankFeatureExtractor
- **Feature extraction looks correct**: Kaldi is configured for NeMo compatibility (80 mel bins, proper windowing)
- **Audio accumulation issue**: NeMoSTT operator accumulates audio chunks but only transcribes on WindowMarker punctuation
- **Transcription happens in `process(Punctuation)`**: Not during regular tuple processing
- **Output attribute mismatch**: Code outputs to 'transcription' attribute but needs verification

## ROOT CAUSE IDENTIFIED! 🔍
**The NeMoSTT operator waits for WindowMarker punctuation to trigger transcription, but FileAudioSource never sends any punctuation!**

This explains why:
- Audio chunks are received (confirmed by debug output showing "Audio chunk received: 16384 bytes")
- Model loads successfully (confirmed by logs)
- But transcription files remain empty (no WindowMarker = no transcription trigger)

### Solution Options:
1. **Add a Beacon operator** that sends periodic WindowMarkers
2. **Modify FileAudioSource** to send WindowMarker after EOF
3. **Modify NeMoSTT** to transcribe on each chunk or after accumulating enough audio
4. **Use a different operator pattern** that doesn't rely on punctuation

## Testing Phases

### Phase 1: Current State Verification ⏳

#### 1.1 Check for Mock Data Usage ✅
- [x] Search codebase for mock data implementations - **NO MOCK DATA FOUND**
- [x] Verify NeMoCTCImpl.cpp uses real audio data - **CONFIRMED: Uses KaldiFbankFeatureExtractor**
- [x] Check if FileAudioSource actually reads audio files - **CONFIRMED: Reads via FileSource**
- [x] Confirm audio chunk format and content - **CONFIRMED: blob format with timestamps**

**Expected**: Real audio data flows through pipeline  
**Actual**: ✅ Real audio data confirmed, no mock data in use

#### 1.1a Punctuation/Window Issue ✅
- [x] Check if NeMoSTT requires punctuation - **CONFIRMED: Waits for WindowMarker**
- [x] Check if FileAudioSource sends punctuation - **CONFIRMED: NO punctuation sent**
- [x] Verify transcription trigger mechanism - **CONFIRMED: Only triggers on WindowMarker**

**Expected**: Punctuation triggers transcription  
**Actual**: ❌ No punctuation sent, transcription never triggered

#### 1.2 Audio Data Flow Verification
- [ ] Add debug logging to FileAudioSource output
- [ ] Log audio chunk sizes and timestamps
- [ ] Verify audio format (16kHz, 16-bit, mono)
- [ ] Check if chunks contain actual audio samples

**Expected**: Audio chunks contain PCM data from WAV file  
**Actual**: _To be tested_

#### 1.3 Feature Extraction Verification  
- [ ] Check KaldiFbankFeatureExtractor implementation
- [ ] Verify feature dimensions match model expectations
- [ ] Log feature values to ensure they're not zeros/mock
- [ ] Compare with Python reference implementation

**Expected**: 80-dimensional filterbank features  
**Actual**: _To be tested_

#### 1.4 Model Input/Output Verification
- [ ] Log model input tensor shape and values
- [ ] Check model output tensor content
- [ ] Verify vocabulary lookup is working
- [ ] Confirm blank_id and token mapping

**Expected**: Model produces logits for vocabulary tokens  
**Actual**: _To be tested_

### Phase 2: Component Testing 🔧

#### 2.1 FileAudioSource Test
- [ ] Create minimal test reading audio file
- [ ] Verify output chunks match file content
- [ ] Test with different audio formats
- [ ] Measure chunk timing accuracy

**Test Code Location**: `test/test_file_audio_source.spl`  
**Results**: _To be documented_

#### 2.2 Feature Extraction Test
- [ ] Create standalone C++ test for feature extractor
- [ ] Compare output with Python librosa/torchaudio
- [ ] Test with known audio samples
- [ ] Verify CMVN normalization if used

**Test Code Location**: `test/test_feature_extraction.cpp`  
**Results**: _To be documented_

#### 2.3 Model Inference Test
- [ ] Create test with pre-computed features
- [ ] Verify model loads correctly
- [ ] Test inference with known inputs
- [ ] Check output decoding to text

**Test Code Location**: `test/test_model_inference.cpp`  
**Results**: _To be documented_

#### 2.4 End-to-End Pipeline Test
- [ ] Run complete pipeline with debug logging
- [ ] Track data at each stage
- [ ] Identify where transcription fails
- [ ] Compare with working Python implementation

**Test Application**: `samples/PipelineDebugTest.spl`  
**Results**: _To be documented_

### Phase 3: Issue Identification 🔍

#### 3.1 Known Issues
- [ ] Empty transcription files (0 bytes)
- [ ] No console output of transcriptions
- [ ] Potential feature processing mismatch
- [ ] Possible mock data usage

#### 3.2 Root Cause Analysis
**Issue**: Empty transcription output  
**Potential Causes**:
1. Mock data being used instead of real audio
2. Feature extraction producing wrong format
3. Model input preprocessing incorrect
4. Decoder not processing model output
5. Output not being written correctly

**Investigation Steps**: _To be documented_

### Phase 4: Fix Priority List 📝

**Priority 1: Critical Fixes**
- [ ] Remove any mock data usage
- [ ] Fix feature extraction pipeline
- [ ] Ensure model receives correct input

**Priority 2: Functionality Fixes**  
- [ ] Fix transcription output writing
- [ ] Add proper error handling
- [ ] Implement logging for debugging

**Priority 3: Enhancement**
- [ ] Add performance metrics
- [ ] Implement streaming optimizations
- [ ] Add support for other models

### Phase 5: Documentation Updates 📚

#### 5.1 Files to Update
- [ ] README.md - Remove merge conflicts, update status
- [ ] ARCHITECTURE.md - Reflect actual implementation
- [ ] HONEST_CURRENT_STATE_JUNE_17_2025.md - Update with test results
- [ ] Sample documentation - Document working examples

#### 5.2 New Documentation Needed
- [ ] WORKING_CONFIGURATION.md - Document verified working setup
- [ ] TROUBLESHOOTING.md - Common issues and solutions
- [ ] PERFORMANCE.md - Benchmarks and optimization tips

## Test Execution Log

### Session 1: 2025-01-20
**Time Started**: _To be filled_  
**Tests Performed**: _To be documented_  
**Findings**: _To be documented_  
**Next Steps**: _To be documented_

## Immediate Fix Options

### Option 1: Add Window to Sample (Simplest)
Create a new sample that adds a tumbling window to trigger transcription:
```spl
// Add after FileAudioSource
stream<blob audioChunk, uint64 audioTimestamp> WindowedAudio = 
    Aggregate(AudioStream) {
        window
            AudioStream: tumbling, time(5); // 5 second windows
        output
            WindowedAudio: audioChunk = Last(audioChunk),
                          audioTimestamp = Last(audioTimestamp);
    }
```

### Option 2: Modify NeMoSTT operator (Better long-term)
Change the operator to transcribe when:
- Enough audio accumulated (e.g., 5 seconds)
- End of stream detected
- Explicit flush requested

### Option 3: Add Beacon for periodic triggers
```spl
stream<int32 tick> Ticker = Beacon() {
    param
        period: 5.0; // Every 5 seconds
}
// Then join with audio stream
```

## Quick Test Commands

```bash
# Set up environment
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh

# Run basic NeMo test
cd samples/output/BasicNeMoTest
./bin/standalone

# Check for mock data
grep -r "mock" impl/include/
grep -r "dummy" impl/include/
grep -r "fake" impl/include/

# Check feature extraction
grep -r "FbankOptions" impl/include/
grep -r "ComputeFeatures" impl/include/

# Monitor output files
watch -n 1 'ls -la data/*.txt'

# Check debug output for audio chunks
cd samples/output/DebugBasicNeMoTest
./bin/standalone 2>&1 | grep "Audio chunk"
```

## Success Criteria

A test is considered successful when:
1. ✅ Application compiles without errors
2. ✅ Application runs without crashes  
3. ✅ Audio file is read correctly
4. ✅ Features are extracted properly
5. ✅ Model produces valid output
6. ✅ Transcription text appears in output files
7. ✅ Output matches expected: "it was the first great song of his life"

## Notes
- Always check Claude.md for critical reminders about verification
- Do not claim anything works without seeing actual transcription output
- Document every test result, even failures
- Keep this file updated as testing progresses