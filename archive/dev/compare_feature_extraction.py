#!/usr/bin/env python3
"""
Compare Python and C++ feature extraction
"""

import numpy as np
import torch
import torchaudio
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

# Load test audio
wav_path = "test_audio_16k.wav"
audio, sr = torchaudio.load(wav_path)
print(f"Loaded audio: {audio.shape}, sample rate: {sr}")

# Create NeMo feature extractor  
feature_extractor = FilterbankFeatures(
    n_fft=512,
    window_size=0.025,
    window_stride=0.01,
    window="hann",
    sample_rate=16000,
    normalize="per_feature",  # Critical for NeMo
    dither=1e-5,
    features=80,  # This is n_mels
    log=True,
    log_zero_guard_type="clamp",
    log_zero_guard_value=1e-10,
    mel_low_freq=0,
    mel_high_freq=None,
)

# Extract features
with torch.no_grad():
    features, _ = feature_extractor(audio, torch.tensor([audio.shape[-1]]))
    features = features.squeeze(0).numpy()

print(f"Feature shape: {features.shape}")
print(f"Feature stats - Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")
print(f"First 5 features of first frame: {features[0, :5]}")

# Save for comparison
np.save("python_features_test.npy", features)

# Also check what happens with first 125 frames
if features.shape[0] >= 125:
    features_125 = features[:125, :]
    print(f"\nFirst 125 frames shape: {features_125.shape}")
    print(f"First 5 features (transposed to [features, time]):")
    features_transposed = features_125.T  # [80, 125]
    for i in range(5):
        print(f"  {features_transposed.flatten()[i]:.4f}")
else:
    print(f"\nWARNING: Only {features.shape[0]} frames available")