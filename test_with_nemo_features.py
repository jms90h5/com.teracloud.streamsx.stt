#!/usr/bin/env python3
"""
Test model with actual NeMo features
"""

import numpy as np
import onnxruntime as ort

# Load NeMo features (already in [features, time] format)
features = np.load("nemo_features.npy")
print(f"NeMo features shape: {features.shape}")
print(f"Stats - Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")

# Create ONNX session
session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Model expects [batch, features, time]
model_input = features[np.newaxis, :, :].astype(np.float32)
length_input = np.array([features.shape[1]], dtype=np.int64)  # number of time frames

print(f"\nModel input shape: {model_input.shape}")
print(f"Length input: {length_input}")

# Run inference
outputs = session.run(None, {
    "processed_signal": model_input,
    "processed_signal_length": length_input
})

predictions = np.argmax(outputs[0][0], axis=-1)
print(f"\nFirst 10 predictions: {predictions[:10]}")

# Decode
vocab = []
with open("models/fastconformer_nemo_export/tokens.txt", 'r') as f:
    for line in f:
        vocab.append(line.strip())

# Simple CTC decode
prev_token = -1
text = ""
for token_id in predictions:
    if token_id != 1024 and token_id != prev_token and token_id < len(vocab):
        token = vocab[token_id]
        # Handle BPE tokens
        if len(token) >= 3 and ord(token[0]) == 0xE2 and ord(token[1]) == 0x96 and ord(token[2]) == 0x81:
            if text:
                text += " "
            text += token[3:]
        else:
            text += token
    prev_token = token_id

print(f"\nTranscription: '{text}'")
print(f"Expected: 'it was the first great sorrow of his life'")