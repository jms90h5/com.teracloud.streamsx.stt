#!/usr/bin/env python3
"""
Compare C++ and Python features side by side
"""

import numpy as np

# Load C++ features
cpp_features = np.fromfile("cpp_features_debug.bin", dtype=np.float32)
cpp_reshaped = cpp_features.reshape(80, 125)  # [mel, time]

# Load correct Python features
py_features = np.fromfile("correct_features_125.bin", dtype=np.float32)
py_reshaped = py_features.reshape(80, 125)  # [mel, time]

print("Feature comparison:")
print(f"C++ shape: {cpp_reshaped.shape}")
print(f"Python shape: {py_reshaped.shape}")

print(f"\nC++ stats - Min: {cpp_features.min():.4f}, Max: {cpp_features.max():.4f}, Mean: {cpp_features.mean():.4f}")
print(f"Python stats - Min: {py_features.min():.4f}, Max: {py_features.max():.4f}, Mean: {py_features.mean():.4f}")

print(f"\nFirst 5 values:")
print(f"C++:    {cpp_features[:5]}")
print(f"Python: {py_features[:5]}")

# Compare first frame (all 80 features for time 0)
print(f"\nFirst frame comparison (80 features):")
cpp_frame0 = cpp_reshaped[:, 0]
py_frame0 = py_reshaped[:, 0]

diffs = []
for i in range(10):
    diff = cpp_frame0[i] - py_frame0[i]
    diffs.append(diff)
    print(f"  Feature {i}: C++={cpp_frame0[i]:.4f}, Py={py_frame0[i]:.4f}, diff={diff:.4f}")

print(f"\nAverage absolute difference: {np.mean(np.abs(cpp_reshaped - py_reshaped)):.4f}")

# Check for systematic differences
print(f"\nSystematic differences:")
print(f"  Mean difference: {np.mean(cpp_reshaped - py_reshaped):.4f}")
print(f"  Std of differences: {np.std(cpp_reshaped - py_reshaped):.4f}")

# Check if it's a scale issue
scale = np.mean(py_reshaped) / np.mean(cpp_reshaped)
print(f"  Scale factor (Python/C++): {scale:.4f}")