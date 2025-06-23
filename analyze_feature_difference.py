#!/usr/bin/env python3
"""
Analyze the difference between C++ and working features
"""

import numpy as np

# Load features
working_input = np.fromfile("working_input.bin", dtype=np.float32)
working_features = working_input.reshape(1, 80, 874)[0]  # [80, 874] features x time

cpp_features = np.fromfile("cpp_features.bin", dtype=np.float32)
cpp_features = cpp_features.reshape(-1, 80)  # [time, features]
cpp_transposed = cpp_features.T  # [features, time]

print("Working features shape:", working_features.shape)
print("C++ features shape:", cpp_transposed.shape)

# Align shapes (C++ has 871 frames, working has 874)
min_frames = min(working_features.shape[1], cpp_transposed.shape[1])
working_aligned = working_features[:, :min_frames]
cpp_aligned = cpp_transposed[:, :min_frames]

# Compare statistics
print(f"\nWorking stats: min={working_aligned.min():.4f}, max={working_aligned.max():.4f}, mean={working_aligned.mean():.4f}")
print(f"C++ stats: min={cpp_aligned.min():.4f}, max={cpp_aligned.max():.4f}, mean={cpp_aligned.mean():.4f}")

# Find where they differ most
diff = cpp_aligned - working_aligned
print(f"\nDifference stats: min={diff.min():.4f}, max={diff.max():.4f}, mean={diff.mean():.4f}")

# Find frames with largest differences
frame_diffs = np.abs(diff).mean(axis=0)  # Average diff per frame
worst_frames = np.argsort(frame_diffs)[-5:]
print(f"\nWorst frames (indices): {worst_frames}")
print(f"Worst frame diffs: {frame_diffs[worst_frames]}")

# Check specific features where minimum occurs
cpp_min_idx = np.where(cpp_aligned == cpp_aligned.min())
work_min_idx = np.where(working_aligned == working_aligned.min())

print(f"\nC++ minimum occurs at feature {cpp_min_idx[0][0]}, frame {cpp_min_idx[1][0]}")
print(f"Working minimum occurs at feature {work_min_idx[0][0]}, frame {work_min_idx[1][0]}")

# Compare the same location
if len(cpp_min_idx[0]) > 0:
    feat, frame = cpp_min_idx[0][0], cpp_min_idx[1][0]
    print(f"\nAt C++ minimum location [{feat}, {frame}]:")
    print(f"  C++ value: {cpp_aligned[feat, frame]}")
    print(f"  Working value: {working_aligned[feat, frame]}")
    print(f"  Difference: {cpp_aligned[feat, frame] - working_aligned[feat, frame]}")

# Check if it's a constant offset
mean_diff_per_feature = diff.mean(axis=1)
print(f"\nMean difference per feature:")
print(f"  Min: {mean_diff_per_feature.min():.4f}")
print(f"  Max: {mean_diff_per_feature.max():.4f}")
print(f"  Std: {mean_diff_per_feature.std():.4f}")

# Check if it's related to the log operation
# In log space, a multiplicative factor becomes additive
# log(a*x) = log(a) + log(x)
# So if C++ = Working + offset, then in linear space: exp(C++) = exp(offset) * exp(Working)
avg_offset = diff.mean()
print(f"\nAverage offset in log space: {avg_offset:.4f}")
print(f"This corresponds to a multiplicative factor of: {np.exp(avg_offset):.4f}")

# Skip visualization without matplotlib

# Check if the issue is in specific frequency bands
low_freq_diff = diff[:20, :].mean()  # First 20 mel bins
mid_freq_diff = diff[30:50, :].mean()  # Middle mel bins
high_freq_diff = diff[60:, :].mean()  # High mel bins

print(f"\nDifference by frequency range:")
print(f"  Low freq (0-20): {low_freq_diff:.4f}")
print(f"  Mid freq (30-50): {mid_freq_diff:.4f}")
print(f"  High freq (60-80): {high_freq_diff:.4f}")