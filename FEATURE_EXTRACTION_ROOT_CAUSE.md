# Feature Extraction Root Cause Analysis

**Date**: 2025-06-23  
**Issue**: C++ implementation produces only blank tokens while Python produces correct transcription  
**Root Cause**: ImprovedFbank uses simple DFT instead of FFT, causing incorrect feature values

## Executive Summary

The NVIDIA FastConformer model integration was failing because the C++ feature extraction produces significantly different mel spectrogram values compared to the Python implementation. The core issue is that ImprovedFbank uses a "simple DFT" implementation while Python uses librosa's optimized FFT. This causes a 13.17 difference in the first mel bin value, leading the model to output only blank tokens.

## The Problem in Detail

### Feature Value Comparison
- **Python (librosa)**: First mel bin = -9.597
- **C++ (ImprovedFbank)**: First mel bin = 3.568
- **Difference**: 13.17 (massive for log mel features)

### Why This Matters
The NVIDIA FastConformer model (stt_en_fastconformer_hybrid_large_streaming_multi) was trained on features extracted using librosa. When features differ by this magnitude, the model cannot recognize any speech patterns and defaults to outputting blank tokens (token 1024).

## Root Cause: DFT vs FFT Implementation

### ImprovedFbank Implementation (Line 104-119 of ImprovedFbank.cpp)
```cpp
// Simple DFT (inefficient but correct for proof of concept)
std::vector<float> power_spectrum(fft_size / 2 + 1);

for (int k = 0; k < fft_size / 2 + 1; ++k) {
    float real = 0.0f, imag = 0.0f;
    
    for (int n = 0; n < fft_size; ++n) {
        float angle = -2.0f * M_PI * k * n / fft_size;
        real += padded_frame[n] * std::cos(angle);
        imag += padded_frame[n] * std::sin(angle);
    }
    
    power_spectrum[k] = real * real + imag * imag;
}
```

### Python/librosa Implementation
```python
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
    win_length=win_length, n_mels=n_mels, fmin=0.0,
    fmax=8000.0, power=2.0, norm=None, htk=False
)
```

## Technical Differences

### 1. Algorithm Complexity
- **DFT (ImprovedFbank)**: O(N²) complexity, computes each frequency bin independently
- **FFT (librosa)**: O(N log N) complexity, uses optimized Cooley-Tukey algorithm

### 2. Numerical Precision
- DFT accumulates more floating-point errors due to many trigonometric operations
- FFT uses recursive decomposition with fewer operations

### 3. Implementation Details
- librosa uses numpy's FFT which is highly optimized (often using FFTW backend)
- ImprovedFbank explicitly states it's "inefficient but correct for proof of concept"

## Parameters That Must Match

For C++ to produce identical features to Python, these parameters must be exact:

### 1. Audio Preprocessing
- Sample rate: 16000 Hz
- Dither: 1e-5 (random noise added to prevent numerical issues)

### 2. Window Function
- Type: Hann window
- Length: 400 samples (25ms)
- Formula: `0.5 * (1.0 - cos(2π * i / (N-1)))`

### 3. FFT Parameters
- FFT size: 512
- Hop length: 160 samples (10ms)
- Power spectrum: magnitude squared

### 4. Mel Filterbank
- Number of mel bins: 80
- Frequency range: 0-8000 Hz
- Mel scale formula: `2595 * log10(1 + freq/700)`
- No normalization (norm=None in librosa)

### 5. Logarithm
- Natural log (not log10)
- Floor value: 1e-10 to prevent log(0)

## Attempted Solutions

### 1. ImprovedFbank (Current Implementation)
- ✅ Correct mel filterbank implementation
- ✅ Proper window function
- ❌ Uses DFT instead of FFT
- **Result**: Features don't match Python

### 2. Kaldi-native-fbank
- ✅ Uses proper FFT
- ❌ Different default parameters
- ❌ Complex configuration
- **Result**: Also produces different features

### 3. Simple_fbank (Removed)
- ❌ Generated completely fake features
- **Result**: Always output "sh"

## Potential Solutions

### 1. Fix ImprovedFbank FFT (Recommended)
Replace the simple DFT with a proper FFT implementation:
```cpp
// Use a proper FFT library like FFTW or KissFFT
#include <fftw3.h>

std::vector<float> FbankComputer::computeFFT(const std::vector<float>& frame) {
    // Allocate FFTW arrays
    double* in = fftw_alloc_real(opts_.n_fft);
    fftw_complex* out = fftw_alloc_complex(opts_.n_fft/2 + 1);
    
    // Create plan
    fftw_plan plan = fftw_plan_dft_r2c_1d(opts_.n_fft, in, out, FFTW_ESTIMATE);
    
    // Copy and pad input
    for (int i = 0; i < opts_.n_fft; ++i) {
        in[i] = (i < frame.size()) ? frame[i] : 0.0;
    }
    
    // Execute FFT
    fftw_execute(plan);
    
    // Compute power spectrum
    std::vector<float> power_spectrum(opts_.n_fft/2 + 1);
    for (int i = 0; i < opts_.n_fft/2 + 1; ++i) {
        power_spectrum[i] = out[i][0]*out[i][0] + out[i][1]*out[i][1];
    }
    
    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return power_spectrum;
}
```

### 2. Configure Kaldi Correctly
Create exact configuration to match librosa:
```cpp
knf::FbankOptions opts;
opts.frame_opts.samp_freq = 16000;
opts.frame_opts.frame_length_ms = 25;
opts.frame_opts.frame_shift_ms = 10;
opts.mel_opts.num_bins = 80;
opts.mel_opts.low_freq = 0;
opts.mel_opts.high_freq = 8000;
opts.frame_opts.dither = 1e-5;
opts.frame_opts.window_type = "hann";
opts.use_log_fbank = true;
opts.use_power = true;
```

### 3. Port librosa to C++
Create a C++ wrapper around Python librosa (not recommended for production):
```cpp
// Use pybind11 to call Python librosa from C++
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

py::scoped_interpreter guard{};
auto librosa = py::module::import("librosa");
auto mel_spec = librosa.attr("feature").attr("melspectrogram");
// ... call with exact same parameters
```

## Verification Steps

To verify a fix works correctly:

1. **Compare Raw Features**
   - Save Python features: `np.save("python_features.npy", features)`
   - Save C++ features to same format
   - Compare numerically: max absolute difference should be < 0.001

2. **Check Feature Statistics**
   - Python range: min=-23.0259, max=5.83632
   - C++ should match within 0.1

3. **Test Model Output**
   - Python produces varied tokens with actual words
   - C++ should produce same token sequence

4. **Benchmark Transcription**
   - Test audio: "A MAN SAID TO THE UNIVERSE SIR I EXIST"
   - Both should produce identical transcription

## Lessons Learned

1. **"Proof of concept" code can persist**: ImprovedFbank was explicitly marked as inefficient but was being used in production
2. **Feature extraction is critical**: Even small differences in features can completely break model performance
3. **Always verify against reference implementation**: The Python code worked correctly and should have been the comparison baseline
4. **Don't assume "improved" means "correct"**: ImprovedFbank was an improvement over fake data but still incorrect

## Current Status

- ✅ Fake data code has been completely removed
- ✅ ImprovedFbank is being used (real features, but wrong values)
- ❌ Features don't match Python due to DFT vs FFT
- ❌ Model outputs only blank tokens

## Next Steps

1. Implement proper FFT in ImprovedFbank using FFTW or similar library
2. Verify features match Python implementation exactly
3. Test that model produces correct transcriptions
4. Document the working configuration for future reference

## References

- Python implementation: `/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/save_python_features.py`
- C++ implementation: `/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/impl/src/ImprovedFbank.cpp`
- Model config: `/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/models/fastconformer_nemo_export/model_config.yaml`
- Test programs: `/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/test_cpp_features.cpp`