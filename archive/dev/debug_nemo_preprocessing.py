#!/usr/bin/env python3
"""
Debug NeMo's actual preprocessing to understand what's different
"""

import torch
import numpy as np
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

# Load test audio
import librosa
audio, sr = librosa.load("test_data/audio/librispeech-1995-1837-0001.wav", sr=16000)
print(f"Audio shape: {audio.shape}")

# Create NeMo feature extractor with exact config
preprocessor = FilterbankFeatures(
    sample_rate=16000,
    n_window_size=400,  # 25ms at 16kHz
    n_window_stride=160,  # 10ms at 16kHz  
    window="hann",
    normalize="per_feature",
    n_fft=512,
    preemph=0.0,  # No pre-emphasis
    nfilt=80,  # Number of mel bins
    lowfreq=0,
    highfreq=None,  # Nyquist
    log=True,  # Log mel spectrogram
    log_zero_guard_type="clamp",
    log_zero_guard_value=1e-10,
    dither=1e-5,
    pad_to=0,
    frame_splicing=1,
)

# Process audio
audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension
audio_len = torch.tensor([len(audio)])

with torch.no_grad():
    features, _ = preprocessor(audio_tensor, audio_len)
    features_np = features.squeeze(0).numpy()  # Remove batch dimension

print(f"NeMo features shape: {features_np.shape}")
print(f"NeMo features stats - Min: {features_np.min():.4f}, Max: {features_np.max():.4f}, Mean: {features_np.mean():.4f}")
print(f"First 5 features of first frame: {features_np[0, :5]}")

# Save for comparison
np.save("nemo_features.npy", features_np)

# Also check what happens without normalization
preprocessor_no_norm = FilterbankFeatures(
    sample_rate=16000,
    n_window_size=400,
    n_window_stride=160,  
    window="hann",
    normalize=None,  # No normalization
    n_fft=512,
    preemph=0.0,
    nfilt=80,
    lowfreq=0,
    highfreq=None,
    log=True,
    log_zero_guard_type="clamp",
    log_zero_guard_value=1e-10,
    dither=1e-5,
    pad_to=0,
    frame_splicing=1,
)

with torch.no_grad():
    features_no_norm, _ = preprocessor_no_norm(audio_tensor, audio_len)
    features_no_norm_np = features_no_norm.squeeze(0).numpy()

print(f"\nNo normalization features shape: {features_no_norm_np.shape}")
print(f"No norm stats - Min: {features_no_norm_np.min():.4f}, Max: {features_no_norm_np.max():.4f}, Mean: {features_no_norm_np.mean():.4f}")

# Compare with librosa
mel_spec = librosa.feature.melspectrogram(
    y=audio + 1e-5 * np.random.randn(len(audio)),
    sr=16000,
    n_fft=512,
    hop_length=160,
    win_length=400,
    n_mels=80,
    window='hann',
    center=True,
    pad_mode='reflect',
    power=2.0
)
log_mel = np.log(mel_spec + 1e-10).T  # Transpose to [time, features]

print(f"\nLibrosa features shape: {log_mel.shape}")
print(f"Librosa stats - Min: {log_mel.min():.4f}, Max: {log_mel.max():.4f}, Mean: {log_mel.mean():.4f}")

# Check frame count differences
print(f"\nFrame counts:")
print(f"  NeMo: {features_np.shape[0]}")
print(f"  Librosa: {log_mel.shape[0]}")
print(f"  Expected from audio length: {(len(audio) - 400) // 160 + 1}")