#!/usr/bin/env python3
"""
Save Python logits for comparison
"""

import numpy as np
import onnxruntime as ort

# Load features
features = np.load("python_features.npy")
features_transposed = features.T
model_input = features_transposed[np.newaxis, :, :].astype(np.float32)
input_length = np.array([features.shape[0]], dtype=np.int64)

# Create session
session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Run inference
outputs = session.run(None, {
    "processed_signal": model_input,
    "processed_signal_length": input_length
})

# Save first few logit values
logits = outputs[0][0]  # [time, vocab]
print(f"Logits shape: {logits.shape}")
print(f"First 5 logit values at position 0:")
for i in range(5):
    print(f"  Token {i}: {logits[0, i]:.4f}")

# Save for comparison
np.save("python_logits_detailed.npy", logits)