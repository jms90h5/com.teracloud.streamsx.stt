#!/usr/bin/env python3
"""
Detailed comparison of features
"""

import numpy as np

# Load the working features
working_features = np.load("python_features.npy")
print(f"Working features shape: {working_features.shape}")
print(f"Working features stats - Min: {working_features.min():.4f}, Max: {working_features.max():.4f}, Mean: {working_features.mean():.4f}")
print(f"First 5 values (time-major): {working_features[0, :5]}")

# When transposed to [features, time] for model
working_transposed = working_features.T
print(f"\nWorking transposed shape: {working_transposed.shape}")
print(f"First 5 values (feature-major): {working_transposed.flatten()[:5]}")

# Load C++ features
cpp_features = np.fromfile("cpp_features_debug.bin", dtype=np.float32)
cpp_reshaped = cpp_features.reshape(80, 125)
print(f"\nC++ features shape: {cpp_reshaped.shape}")
print(f"C++ features stats - Min: {cpp_features.min():.4f}, Max: {cpp_features.max():.4f}, Mean: {cpp_features.mean():.4f}")
print(f"First 5 values (feature-major): {cpp_features[:5]}")

# Key insight: Check if features need different preprocessing
print("\n=== Checking preprocessing differences ===")

# The working features have very different stats
print(f"Working: Min={working_features.min():.2f}, Max={working_features.max():.2f}, Mean={working_features.mean():.2f}")
print(f"C++: Min={cpp_features.min():.2f}, Max={cpp_features.max():.2f}, Mean={cpp_features.mean():.2f}")

# Check the audio file
import librosa
audio, sr = librosa.load("test_audio_16k.wav", sr=16000)
print(f"\nAudio stats: shape={audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")

# What audio produced the working features?
# Working features have 874 frames, which means ~14 seconds of audio
print(f"\nWorking features suggest audio length: {874 * 0.01:.2f} seconds")
print(f"Current audio length: {len(audio) / 16000:.2f} seconds")