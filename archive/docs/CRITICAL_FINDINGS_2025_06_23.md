# Critical Findings - June 23, 2025

## Executive Summary

After extensive testing, the NeMo FastConformer integration has a working model but feature extraction mismatches prevent transcription. This document captures all critical findings to prevent rediscovering these issues.

## Working Components

### 1. The Correct Model
- **Path**: `models/fastconformer_nemo_export/ctc_model.onnx` (471MB)
- **Input names**: `processed_signal` (shape: [batch, 80, time]), `processed_signal_length` (shape: [batch])
- **Output names**: `log_probs` (shape: [batch, time, 1025]), `encoded_lengths` (shape: [batch])
- **Vocabulary**: 1024 tokens (0-1023), blank token = 1024
- **Accepts dynamic input sizes** (not fixed to 500 frames)

### 2. Working Features
- **File**: `working_input.bin` (279,680 bytes for 874 frames)
- **Statistics**: min=-10.73, max=6.68, mean=-3.91, std=2.77
- **Produces**: "it was the first great sorrow of his life..." (correct transcription)

### 3. Python Test That Works
```python
# test_working_input_again.py demonstrates the working pipeline:
working_input = np.fromfile("working_input.bin", dtype=np.float32).reshape(1, 80, 874)
outputs = session.run(None, {
    "processed_signal": working_input,
    "processed_signal_length": np.array([874], dtype=np.int64)
})
# Produces correct transcription
```

## Broken Components (UPDATED June 23, 2025)

### 1. Wrong Model Being Used
- **C++ tests use**: `models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx` (2.6MB)
- **This model expects**: `audio_signal` input, fixed 500 frames, different interface
- **Result**: "Invalid Feed Input Name:audio_signal" error
- **STATUS**: ✅ FIXED - Code now uses correct model path and "processed_signal" input name

### 2. Feature Extraction Mismatch (ROOT CAUSE FOUND)
**Original Issue**: C++ features had wrong statistics:
- C++ features: min=-23.03, max=5.84, mean=-4.21
- Working features: min=-10.73, max=6.68, mean=-3.91

**Root Cause**: Zero-width mel filterbank filters
- Mel filter 2 had `center == right`, creating zero-width triangular filter
- This caused zero energy → log(1e-10) = -23.026 for all frames
- Bug was in `ImprovedFbank.cpp` line 56: using floor() instead of round()

**Fix Applied**:
```cpp
// Old: bin_points[i] = floor((n_fft + 1) * hz_points[i] / sample_rate)
// New: bin_points[i] = round(n_fft * hz_points[i] / sample_rate)
// Plus protection against zero-width filters
```

**Current Status**: 
- ✅ Fixed constant -23.026 values
- ✅ Mean now -3.927 (target -3.912) - only 0.4% error
- ⚠️ Min still -15.39 (target -10.73) - better but not perfect

### 3. Configuration Issues
- Default blank_id = 1024 in NeMoCTCModel.hpp (correct for large model)
- STTPipeline sets blank_id = 0 for Zipformer (wrong model)
- Test programs point to wrong model paths
- **STATUS**: ✅ FIXED - Input names and model paths corrected

## Critical Code Locations (UPDATED June 23, 2025)

### Model Loading
- `impl/src/NeMoCacheAwareConformer.cpp:24-26` - ✅ FIXED: Now uses "processed_signal" input name
- `impl/src/STTPipeline.cpp:297` - createNeMoPipeline() function
- `test/test_real_nemo_fixed.cpp:26` - hardcoded wrong model path

### Feature Extraction
- `impl/src/ImprovedFbank.cpp:53-86` - ✅ FIXED: Zero-width filter bug
  - Changed bin calculation from floor() to round()
  - Added protection against zero-width filters
  - Result: No more constant -23.026 values
- Feature statistics now much closer (mean error only 0.4%)
- Remaining min value difference likely due to different windowing or preprocessing

## What Actually Works

1. **ONNX Runtime Integration**: ✅ Works correctly when given right features
2. **CTC Decoding**: ✅ Properly decodes tokens including BPE handling
3. **Model Loading**: ✅ Can load both models successfully
4. **FFT Implementation**: ✅ Already fixed to use kaldi-native-fbank

## What's Still Broken (UPDATED June 23, 2025)

1. **Feature Extraction**: ⚠️ PARTIALLY FIXED - Statistics much closer but min value still off
2. **Model Path**: ✅ FIXED - Code updated to use correct 471MB model
3. **Input Names**: ✅ FIXED - Now uses "processed_signal" 
4. **SPL Samples**: ❓ Need testing with fixed library

## Required Fixes (UPDATED June 23, 2025)

### 1. Update Model Configuration - ✅ COMPLETED
```cpp
// In NeMoCacheAwareConformer.cpp:24-26
input_names_ = {"processed_signal"}    // FIXED
output_names_ = {"log_probs"}          // FIXED
// Model already configured for blank_id = 1024
```

### 2. Fix Feature Extraction - ✅ MOSTLY FIXED
**Problem**: Zero-width mel filters causing constant -23.026 values
**Solution Applied in ImprovedFbank.cpp:53-86**:
```cpp
// Fixed bin calculation:
bin_points[i] = static_cast<int>(std::round(opts_.n_fft * hz_points[i] / opts_.sample_rate));

// Added protection:
if (left == center) center = left + 1;
if (center == right) right = center + 1;
```
**Results**:
- Before: min=-23.03, max=5.84, mean=-4.21
- After: min=-15.39, max=6.68, mean=-3.93
- Target: min=-10.73, max=6.68, mean=-3.91

### 3. Update Test Paths - ✅ NOTED
Model paths need updating in test files to use the 471MB model

## Testing Checklist (UPDATED June 23, 2025)

1. ✅ Model loads without errors
2. ✅ Inference runs without crashes
3. ⚠️ Features mostly match working statistics (mean within 0.4%)
4. ❓ Transcription output needs verification
5. ❓ SPL samples need testing with fixed library

## Do NOT Trust These Files

- `correct_python_features_*.npy` - Despite the name, these don't produce correct output
- `models/fastconformer_ctc_export/` - This 45MB model has limited functionality
- Any documentation claiming the samples are "working" without verification

## Verified Working Configuration

- **Model**: `models/fastconformer_nemo_export/ctc_model.onnx` (471MB)
- **Features**: `working_input.bin` (preprocessed somehow to get correct range)
- **Blank ID**: 1024
- **Input Format**: [batch=1, mels=80, time=dynamic]
- **Expected Output**: "it was the first great sorrow of his life..."

## June 23, 2025 Fix Details

### The Zero-Width Filter Bug

**Discovery Process**:
1. Noticed mel filter bin 2 had constant value -23.026 in ALL frames
2. Recognized -23.026 = log(1e-10), the floor value when energy = 0
3. Debugged mel filterbank initialization and found filter 2 had center=2, right=2
4. Zero-width filter → no energy collected → log(floor) = -23.026

**Why It Happened**:
- Low frequency mel filters (near 0 Hz) have closely spaced center frequencies
- Using floor() for bin calculation caused multiple filters to round to same bin
- Example: 68.48 Hz and 92.76 Hz both mapped to bin 2 with floor((n_fft+1)*f/sr)

**The Fix**:
1. Changed from `floor((n_fft+1)*hz/sr)` to `round(n_fft*hz/sr)`
2. Added explicit protection against zero-width filters
3. This ensures every mel filter covers at least one FFT bin

**Impact**:
- Eliminated constant -23.026 values
- Improved feature statistics dramatically (mean error from 7.5% to 0.4%)
- Toolkit now produces features much closer to working reference

### Remaining Minor Differences

The min value is still -15.39 vs target -10.73. This is likely due to:
- Different audio preprocessing (pre-emphasis, windowing)
- Edge effects in first/last frames
- Different handling of silence/low energy regions

These differences are much smaller and may not impact transcription quality.

## Next Session Starting Point

1. Load this document first
2. Verify fixes are still in place in ImprovedFbank.cpp
3. Test transcription with fixed library to verify output
4. Check if min value difference impacts results
5. Update SPL sample paths if needed