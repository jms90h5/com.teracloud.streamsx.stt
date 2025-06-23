#!/usr/bin/env python3
"""
Debug the first frame issue
"""

import numpy as np
import os

# Load features
working_input = np.fromfile("working_input.bin", dtype=np.float32)
working_features = working_input.reshape(1, 80, 874)[0]  # [80, 874]

cpp_features = np.fromfile("cpp_features.bin", dtype=np.float32)
cpp_features = cpp_features.reshape(-1, 80)  # [time, features]

print("First frame comparison:")
print("Working first frame (first 10 values):", working_features[:10, 0])
print("C++ first frame (first 10 values):", cpp_features[0, :10])

print("\nFirst 5 frames, feature 2 (where C++ minimum occurs):")
print("Working:", working_features[2, :5])
print("C++ transposed:", cpp_features[:5, 2])

# Load debug files
print("\nFrom debug files:")
with open("cpp_features_debug.txt", "r") as f:
    print("C++ debug output:")
    for line in f:
        print(" ", line.rstrip())

if os.path.exists("python_features.txt"):
    with open("python_features.txt", "r") as f:
        print("\nPython debug output:")
        for line in f:
            print(" ", line.rstrip())

# Check if the issue is silence at the beginning
print("\nChecking audio signal at beginning...")
# The working features suggest the audio might have silence or very low energy at start
# Low log mel values like -23 could indicate near-silence being processed differently