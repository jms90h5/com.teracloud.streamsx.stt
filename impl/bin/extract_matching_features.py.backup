#!/usr/bin/env python3
"""
Extract features to match the working ones
"""

import numpy as np
import librosa

# Load audio
audio, sr = librosa.load("test_audio_16k.wav", sr=16000)
print(f"Audio shape: {audio.shape}")

# Try different feature extraction parameters to match working features
def try_features(audio, params_name, **kwargs):
    # Default params
    params = {
        'y': audio,
        'sr': 16000,
        'n_fft': 512,
        'hop_length': 160,
        'win_length': 400,
        'n_mels': 80,
        'window': 'hann',
        'center': True,
        'pad_mode': 'reflect',
        'power': 2.0
    }
    params.update(kwargs)
    
    # Add dither
    audio_dithered = audio + 1e-5 * np.random.randn(len(audio))
    params['y'] = audio_dithered
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(**params)
    
    # Convert to log
    log_mel = np.log(mel_spec + 1e-10)
    
    # Transpose to [time, features]
    features = log_mel.T
    
    print(f"\n{params_name}:")
    print(f"  Shape: {features.shape}")
    print(f"  Stats - Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
    print(f"  First 5: {features[0, :5]}")
    
    return features

# Try different parameters
f1 = try_features(audio, "Default (power=2.0)")
f2 = try_features(audio, "Power=1.0", power=1.0)
f3 = try_features(audio, "No center", center=False)
f4 = try_features(audio, "fmax=8000", fmax=8000)
f5 = try_features(audio, "fmin=20", fmin=20)

# Load working features to compare
working = np.load("python_features.npy")
print(f"\nWorking features:")
print(f"  Shape: {working.shape}")
print(f"  Stats - Min: {working.min():.4f}, Max: {working.max():.4f}, Mean: {working.mean():.4f}")
print(f"  First 5: {working[0, :5]}")

# Check which is closest
def compare_stats(f1, f2):
    return abs(f1.min() - f2.min()) + abs(f1.max() - f2.max()) + abs(f1.mean() - f2.mean())

print("\nCloseness to working features (lower is better):")
print(f"  Default: {compare_stats(f1, working):.4f}")
print(f"  Power=1.0: {compare_stats(f2, working):.4f}")
print(f"  No center: {compare_stats(f3, working):.4f}")
print(f"  fmax=8000: {compare_stats(f4, working):.4f}")
print(f"  fmin=20: {compare_stats(f5, working):.4f}")