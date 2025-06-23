#!/usr/bin/env python3
"""
Compare C++ and Python features to find differences
"""
import numpy as np
import struct

def load_binary_features(filename, dtype=np.float32):
    """Load features from binary file"""
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Convert bytes to float array
    num_floats = len(data) // 4  # 4 bytes per float32
    features = struct.unpack(f'{num_floats}f', data)
    return np.array(features, dtype=dtype)

def main():
    # Load Python features
    python_features = np.load("python_features_time_mel.npy")
    print(f"Python features shape: {python_features.shape}")
    print(f"Python features range: [{python_features.min():.3f}, {python_features.max():.3f}]")
    print(f"Python features mean: {python_features.mean():.3f}, std: {python_features.std():.3f}")
    
    # Load C++ features
    try:
        cpp_features_raw = load_binary_features("cpp_features_debug.bin")
        # Assume C++ saves 125 frames x 80 features
        cpp_features = cpp_features_raw.reshape(125, 80)
        print(f"\nC++ features shape: {cpp_features.shape}")
        print(f"C++ features range: [{cpp_features.min():.3f}, {cpp_features.max():.3f}]")
        print(f"C++ features mean: {cpp_features.mean():.3f}, std: {cpp_features.std():.3f}")
    except Exception as e:
        print(f"\nError loading C++ features: {e}")
        return
    
    # Compare first few values
    print(f"\nFirst 10 values comparison:")
    print(f"{'Index':<8} {'Python':<12} {'C++':<12} {'Diff':<12} {'Ratio':<12}")
    print("-" * 60)
    
    min_frames = min(python_features.shape[0], cpp_features.shape[0])
    for i in range(10):
        py_val = python_features.flat[i]
        cpp_val = cpp_features.flat[i]
        diff = py_val - cpp_val
        ratio = py_val / cpp_val if cpp_val != 0 else float('inf')
        print(f"{i:<8} {py_val:<12.6f} {cpp_val:<12.6f} {diff:<12.6f} {ratio:<12.6f}")
    
    # Check if features are similar
    print(f"\nFeature statistics:")
    diffs = python_features[:min_frames].flatten() - cpp_features[:min_frames].flatten()
    print(f"Mean absolute difference: {np.abs(diffs).mean():.6f}")
    print(f"Max absolute difference: {np.abs(diffs).max():.6f}")
    print(f"RMS difference: {np.sqrt((diffs**2).mean()):.6f}")
    
    # Check if C++ is using log or log10
    print(f"\nChecking if C++ might be using log10 instead of natural log:")
    log10_factor = np.log(10)
    scaled_cpp = cpp_features * log10_factor
    scaled_diffs = python_features[:min_frames].flatten() - scaled_cpp[:min_frames].flatten()
    print(f"After scaling by log(10), mean abs diff: {np.abs(scaled_diffs).mean():.6f}")
    
    # Check if there's a constant offset
    mean_diff = diffs.mean()
    print(f"\nMean difference (potential offset): {mean_diff:.6f}")
    offset_corrected_diffs = diffs - mean_diff
    print(f"After offset correction, mean abs diff: {np.abs(offset_corrected_diffs).mean():.6f}")

if __name__ == "__main__":
    main()