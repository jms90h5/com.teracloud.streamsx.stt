#!/usr/bin/env python3
"""
Quick test to see what length value Python is actually using
"""

import numpy as np
import json

# Load debug info
with open("python_debug_info.json", "r") as f:
    info = json.load(f)

print(f"Feature shape: {info['feature_shape']}")
print(f"Model input shape: {info['model_input_shape']}")
print(f"Output shape: {info['output_shape']}")

# The key insight: processed_signal_length should be the number of TIME frames
# For shape [batch, features, time], the length is the last dimension
print(f"\nprocessed_signal_length should be: {info['feature_shape'][0]}")

# Let's also check what encoded_lengths output we get
logits = np.load("python_logits.npy")
print(f"\nLogits shape: {logits.shape}")
print(f"Output has {logits.shape[1]} time steps")
print(f"This suggests the model downsampled by factor: {info['feature_shape'][0] / logits.shape[1]:.2f}")