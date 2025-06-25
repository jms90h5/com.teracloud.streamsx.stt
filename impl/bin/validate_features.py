#!/usr/bin/env python3
"""
Validate that saved features produce correct output
"""

import numpy as np
import onnxruntime as ort

# Load saved features
features = np.load("python_features.npy")
print(f"Loaded features shape: {features.shape}")

# Create session
session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Test 1: Direct transpose (what we do in C++)
print("\n=== Test 1: Direct transpose ===")
features_transposed = features.T
model_input = features_transposed[np.newaxis, :, :].astype(np.float32)
input_length = np.array([features.shape[0]], dtype=np.int64)

print(f"Input shape: {model_input.shape}")
print(f"First 5 values: {model_input[0, 0, :5]}")

outputs = session.run(None, {
    "processed_signal": model_input,
    "processed_signal_length": input_length
})

predictions = np.argmax(outputs[0][0], axis=-1)
print(f"Predictions: {predictions[:10]}")
print(f"Unique: {np.unique(predictions)}")

# Test 2: Check feature statistics
print("\n=== Feature Statistics ===")
print(f"Min: {features.min():.4f}, Max: {features.max():.4f}")
print(f"Mean: {features.mean():.4f}, Std: {features.std():.4f}")

# Save the exact model input that works
np.save("working_model_input.npy", model_input)
print("\nSaved working model input to working_model_input.npy")

# Also save in a simple binary format for C++
model_input.tofile("working_input.bin")
print(f"Saved as binary ({model_input.size} floats) to working_input.bin")