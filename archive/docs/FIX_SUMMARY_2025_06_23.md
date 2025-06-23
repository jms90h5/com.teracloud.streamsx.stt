# Zero-Width Mel Filter Bug Fix Summary

## The Problem

The NeMo FastConformer STT integration was producing incorrect transcriptions despite having a working model. Investigation revealed that mel filter bin 2 had a constant value of -23.026 across ALL audio frames, which is log(1e-10) - the floor value used when energy is zero.

## Root Cause Analysis

### Why Zero Energy?

The mel filterbank implementation had a bug where some filters had zero width:
- Mel filter 2: left=1, center=2, right=2 (center == right!)
- A triangular filter with center=right has no right slope
- No right slope means the filter collects zero energy
- Zero energy → log(1e-10) = -23.026

### Why Zero Width?

The bug was in converting frequencies to FFT bin indices:

```cpp
// BUGGY CODE:
bin_points[i] = floor((n_fft + 1) * hz_points[i] / sample_rate);
```

For low frequencies near 0 Hz, mel filters are closely spaced:
- Filter 2: 44.94 Hz → 68.48 Hz → 92.76 Hz
- With floor(), both 68.48 Hz and 92.76 Hz mapped to bin 2
- Result: center=2, right=2 (zero width!)

## The Fix

### Code Changes in ImprovedFbank.cpp

1. **Fixed bin calculation** (line 57):
```cpp
// OLD: floor((n_fft + 1) * hz / sr)
// NEW: round(n_fft * hz / sr)
bin_points[i] = static_cast<int>(std::round(opts_.n_fft * hz_points[i] / opts_.sample_rate));
```

2. **Added protection** (lines 69-71):
```cpp
// Ensure we don't have zero-width filters
if (left == center) center = left + 1;
if (center == right) right = center + 1;
```

### Why These Changes Work

1. **round() vs floor()**: Rounding prevents adjacent frequency points from mapping to the same bin
2. **Removed +1 from n_fft**: The original formula was incorrect for FFT bin calculation
3. **Explicit protection**: Even if rounding fails, we guarantee non-zero filter width

## Results

### Feature Statistics Improvement

| Metric | Before Fix | After Fix | Target | Improvement |
|--------|------------|-----------|---------|-------------|
| Min    | -23.026    | -15.393   | -10.730 | 33% closer  |
| Max    | 5.846      | 6.676     | 6.678   | 99% match   |
| Mean   | -4.209     | -3.927    | -3.912  | 0.4% error  |

### Key Achievements

1. **Eliminated constant -23.026 values** in mel bin 2
2. **Mean error reduced from 7.5% to 0.4%** 
3. **Max value now matches** the working reference
4. **All mel filters now have non-zero width**

## Verification

The fix was verified by:
1. Creating a debug program to print mel filter configurations
2. Running feature extraction on test audio
3. Comparing statistics with known working features
4. Confirming no more constant values in any mel bin

## Remaining Work

1. **Min value difference** (-15.39 vs -10.73) likely due to:
   - Different windowing functions
   - Pre-emphasis settings
   - Edge effects in first/last frames

2. **SPL sample testing** needs to be done with the fixed library

3. **End-to-end verification** of transcription output

## Lessons Learned

1. **Feature statistics are critical** - Always compare with known working values
2. **Constant values indicate algorithmic bugs** - -23.026 was a clear signal
3. **Low-level DSP bugs can be subtle** - Floor vs round made huge difference
4. **Protective coding helps** - Explicit checks prevent edge cases

This fix demonstrates the importance of careful numerical analysis in audio processing pipelines.