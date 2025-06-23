#!/usr/bin/env python3
"""
Find what preprocessing creates the working features
"""

import numpy as np

# Load working features
working = np.load("python_features.npy")
print(f"Working features shape: {working.shape}")
print(f"Working stats - Min: {working.min():.4f}, Max: {working.max():.4f}, Mean: {working.mean():.4f}, Std: {working.std():.4f}")

# Check if features might be normalized
working_per_feature_mean = working.mean(axis=0)  # Mean across time for each feature
working_per_feature_std = working.std(axis=0)    # Std across time for each feature

print(f"\nPer-feature statistics:")
print(f"  Mean range: [{working_per_feature_mean.min():.4f}, {working_per_feature_mean.max():.4f}]")
print(f"  Std range: [{working_per_feature_std.min():.4f}, {working_per_feature_std.max():.4f}]")

# Check if normalized (mean ~0, std ~1)
overall_mean_of_means = working_per_feature_mean.mean()
overall_mean_of_stds = working_per_feature_std.mean()
print(f"  Average of per-feature means: {overall_mean_of_means:.4f}")
print(f"  Average of per-feature stds: {overall_mean_of_stds:.4f}")

# Load the exact working input that produces correct output
working_input = np.fromfile("working_input.bin", dtype=np.float32).reshape(1, 80, 874)
working_flat = working_input[0]  # Remove batch dimension [80, 874]

print(f"\nWorking input that produces correct output:")
print(f"  Shape: {working_flat.shape}")
print(f"  First 5 values: {working_flat.flatten()[:5]}")

# This should match the transposed working features
working_transposed = working.T
print(f"\nWorking features transposed:")
print(f"  Shape: {working_transposed.shape}")
print(f"  First 5 values: {working_transposed.flatten()[:5]}")

# Check if they match
diff = np.abs(working_flat - working_transposed).max()
print(f"\nMax difference: {diff}")