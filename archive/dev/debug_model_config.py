#!/usr/bin/env python3
"""
Debug the exact ONNX model configuration and providers
"""

import onnxruntime as ort
import numpy as np

# Check ONNX Runtime version and providers
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Create session with explicit options
session_options = ort.SessionOptions()
session_options.inter_op_num_threads = 1
session_options.intra_op_num_threads = 1
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# Load model
session = ort.InferenceSession(
    "models/fastconformer_nemo_export/ctc_model.onnx",
    sess_options=session_options,
    providers=['CPUExecutionProvider']
)

print(f"\nModel metadata:")
metadata = session.get_modelmeta()
print(f"Producer: {metadata.producer_name}")
print(f"Graph name: {metadata.graph_name}")
print(f"Custom metadata: {metadata.custom_metadata_map}")

print(f"\nProviders in use: {session.get_providers()}")

# Test with a simple constant input to see if model works at all
print("\n=== Testing with constant input ===")
# Create input of all 1s
const_features = np.ones((1, 80, 100), dtype=np.float32)
const_length = np.array([100], dtype=np.int64)

outputs = session.run(None, {
    "processed_signal": const_features,
    "processed_signal_length": const_length
})

print(f"Output shape: {outputs[0].shape}")
print(f"Encoded length: {outputs[1]}")

# Check predictions
predictions = np.argmax(outputs[0][0], axis=-1)
print(f"Unique predictions: {np.unique(predictions)}")
print(f"All blanks? {np.all(predictions == 1024)}")

# Test with random noise
print("\n=== Testing with random noise ===")
noise_features = np.random.randn(1, 80, 100).astype(np.float32)
outputs = session.run(None, {
    "processed_signal": noise_features,
    "processed_signal_length": const_length
})

predictions = np.argmax(outputs[0][0], axis=-1)
print(f"Unique predictions: {np.unique(predictions)}")
print(f"All blanks? {np.all(predictions == 1024)}")

# Save session configuration for C++ to match
config = {
    "inter_op_num_threads": 1,
    "intra_op_num_threads": 1,
    "graph_optimization_level": "ORT_ENABLE_EXTENDED",
    "providers": session.get_providers(),
    "onnx_version": ort.__version__
}

import json
with open("onnx_config.json", "w") as f:
    json.dump(config, f, indent=2)
    
print(f"\nSaved ONNX configuration to onnx_config.json")