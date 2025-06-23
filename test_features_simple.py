#!/usr/bin/env python3
"""
Simple feature extraction to understand what's different
"""

import numpy as np
import torch
import torchaudio
import librosa

# Load test audio
wav_path = "test_audio_16k.wav"
audio, sr = librosa.load(wav_path, sr=16000)
print(f"Loaded audio: {audio.shape}, sample rate: {sr}")

# Apply dither
audio = audio + np.random.normal(0, 1e-5, audio.shape)

# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=16000,
    n_fft=512,
    hop_length=160,
    win_length=400,
    n_mels=80,
    fmin=0,
    fmax=8000,
    window='hann'
)

# Convert to log scale
log_mel = np.log(np.maximum(mel_spec, 1e-10))

# Transpose to [time, features]
features = log_mel.T

print(f"Feature shape: {features.shape}")
print(f"Feature stats - Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
print(f"First 5 features of first frame: {features[0, :5]}")

# Check first 125 frames
if features.shape[0] >= 125:
    features_125 = features[:125, :]
    print(f"\nFirst 125 frames shape: {features_125.shape}")
    print(f"First 5 features (transposed to [features, time]):")
    features_transposed = features_125.T  # [80, 125]
    for i in range(5):
        print(f"  {features_transposed.flatten()[i]:.4f}")
        
# Save for comparison
np.save("python_features_simple.npy", features)