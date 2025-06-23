#!/usr/bin/env python3
"""
Compare features from different sources
"""

import numpy as np

# Load features
nemo_no_norm = np.load("nemo_features_no_norm.npy")  # [80, 874]
cpp_features = np.fromfile("cpp_features_debug.bin", dtype=np.float32).reshape(80, 125)

print("Feature comparison:")
print(f"NeMo shape: {nemo_no_norm.shape}")
print(f"C++ shape: {cpp_features.shape}")

print(f"\nNeMo stats - Min: {nemo_no_norm.min():.4f}, Max: {nemo_no_norm.max():.4f}, Mean: {nemo_no_norm.mean():.4f}")
print(f"C++ stats - Min: {cpp_features.min():.4f}, Max: {cpp_features.max():.4f}, Mean: {cpp_features.mean():.4f}")

# Compare first frame
print(f"\nFirst frame comparison (first 10 features):")
nemo_frame0 = nemo_no_norm[:, 0]
cpp_frame0 = cpp_features[:, 0]

for i in range(10):
    diff = cpp_frame0[i] - nemo_frame0[i]
    print(f"  Feature {i}: NeMo={nemo_frame0[i]:.4f}, C++={cpp_frame0[i]:.4f}, diff={diff:.4f}")

# The key insight: Are the features systematically different?
print(f"\nSystematic differences:")
print(f"  Mean difference: {np.mean(cpp_features[:, :125] - nemo_no_norm[:, :125]):.4f}")

# Check if it's a scale or offset issue
scale = nemo_no_norm[:, :125].std() / cpp_features.std()
print(f"  Std ratio (NeMo/C++): {scale:.4f}")

# Save first 125 frames of NeMo for direct comparison
nemo_125 = nemo_no_norm[:, :125]
nemo_125.tofile("nemo_features_125.bin")
print(f"\nSaved first 125 frames of NeMo features to nemo_features_125.bin")