#!/usr/bin/env python3
"""
Debug why C++ features don't match working Python features
"""

import numpy as np
import soundfile as sf
import json

# Load the working features that produce correct output
working_input = np.fromfile("working_input.bin", dtype=np.float32)
working_features = working_input.reshape(1, 80, 874)[0]  # [80, 874] features x time

# Load Python features
python_features = np.load("python_features.npy")  # Should be [time, features]

print("Working features (correct output):")
print(f"  Shape: {working_features.shape}")
print(f"  Stats: min={working_features.min():.4f}, max={working_features.max():.4f}, mean={working_features.mean():.4f}")

print("\nPython features:")
print(f"  Shape: {python_features.shape}")
print(f"  Stats: min={python_features.min():.4f}, max={python_features.max():.4f}, mean={python_features.mean():.4f}")

# Check if they're just transposed
python_transposed = python_features.T
print("\nPython features transposed:")
print(f"  Shape: {python_transposed.shape}")
print(f"  Stats: min={python_transposed.min():.4f}, max={python_transposed.max():.4f}, mean={python_transposed.mean():.4f}")

# Check if they match
if python_transposed.shape == working_features.shape:
    diff = np.abs(python_transposed - working_features).max()
    print(f"\nMax difference between python transposed and working: {diff}")
    
    if diff < 0.001:
        print("✅ Python features match working features!")
    else:
        print("❌ Python features don't match working features")
        # Find where they differ
        diff_idx = np.where(np.abs(python_transposed - working_features) > 0.01)
        if len(diff_idx[0]) > 0:
            print(f"First difference at feature {diff_idx[0][0]}, time {diff_idx[1][0]}")
            print(f"  Python: {python_transposed[diff_idx[0][0], diff_idx[1][0]]}")
            print(f"  Working: {working_features[diff_idx[0][0], diff_idx[1][0]]}")
else:
    print("❌ Shape mismatch")

# Now check C++ features if they exist
import os
if os.path.exists("cpp_features.bin"):
    cpp_features = np.fromfile("cpp_features.bin", dtype=np.float32)
    # Try to figure out shape - should be time x features
    n_frames = len(cpp_features) // 80
    if n_frames * 80 == len(cpp_features):
        cpp_features = cpp_features.reshape(n_frames, 80)
        print(f"\nC++ features:")
        print(f"  Shape: {cpp_features.shape}")
        print(f"  Stats: min={cpp_features.min():.4f}, max={cpp_features.max():.4f}, mean={cpp_features.mean():.4f}")
        
        # Compare with working features
        cpp_transposed = cpp_features.T
        if cpp_transposed.shape[1] == working_features.shape[1]:
            print("\nComparing first few frames:")
            for i in range(min(3, cpp_transposed.shape[1])):
                cpp_frame = cpp_transposed[:, i]
                working_frame = working_features[:, i]
                frame_diff = np.abs(cpp_frame - working_frame).mean()
                print(f"  Frame {i} mean diff: {frame_diff:.4f}")
else:
    print("\nNo cpp_features.bin found - need to run C++ test first")

# Check what preprocessing parameters create the working features
print("\n\nAnalyzing working features to deduce preprocessing:")
# Check if normalized (mean ~0, std ~1 per feature)
per_feature_mean = working_features.mean(axis=1)  # Mean across time
per_feature_std = working_features.std(axis=1)   # Std across time
print(f"Per-feature mean: min={per_feature_mean.min():.4f}, max={per_feature_mean.max():.4f}, avg={per_feature_mean.mean():.4f}")
print(f"Per-feature std: min={per_feature_std.min():.4f}, max={per_feature_std.max():.4f}, avg={per_feature_std.mean():.4f}")

if abs(per_feature_mean.mean()) < 0.1 and abs(per_feature_std.mean() - 1.0) < 0.1:
    print("✅ Features appear to be normalized (mean~0, std~1)")
else:
    print("❌ Features are NOT normalized")