#!/usr/bin/env python3
"""
Analyze C++ features
"""

import numpy as np

# Load C++ features (125 frames * 80 features = 10000 values)
cpp_features = np.fromfile("cpp_features_debug.bin", dtype=np.float32)
print(f"Loaded {len(cpp_features)} C++ features")

# Reshape to [features, time] as stored
cpp_reshaped = cpp_features.reshape(80, 125)  # C++ stores in [features, time]
print(f"Reshaped to: {cpp_reshaped.shape}")

# First 5 values (should match C++ output)
print(f"\nFirst 5 values (as stored): {cpp_features[:5]}")

# Check stats
print(f"\nC++ Feature stats:")
print(f"  Min: {cpp_features.min():.4f}, Max: {cpp_features.max():.4f}")
print(f"  Mean: {cpp_features.mean():.4f}, Std: {cpp_features.std():.4f}")

# Load Python features for comparison
py_simple = np.load("python_features_simple.npy")
print(f"\nPython simple features shape: {py_simple.shape}")
print(f"Python stats - Min: {py_simple.min():.4f}, Max: {py_simple.max():.4f}, Mean: {py_simple.mean():.4f}")

# Compare first frame
cpp_first_frame = cpp_reshaped[:, 0]  # First time frame
py_first_frame = py_simple[0, :]  # First time frame

print(f"\nFirst frame comparison:")
print(f"C++ first 5 features: {cpp_first_frame[:5]}")
print(f"Python first 5 features: {py_first_frame[:5]}")

# Check if normalization is different
print(f"\nChecking for per-feature normalization...")
# In NeMo, per-feature normalization means each mel bin is normalized across time
for i in range(5):
    cpp_feature_i = cpp_reshaped[i, :]  # All time frames for feature i
    print(f"Feature {i} - Mean: {cpp_feature_i.mean():.4f}, Std: {cpp_feature_i.std():.4f}")