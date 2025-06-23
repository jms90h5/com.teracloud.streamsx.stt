#!/usr/bin/env python3
"""
Test with the previously working input
"""

import numpy as np
import onnxruntime as ort

# Load the working input that produced correct transcription
working_input = np.fromfile("working_input.bin", dtype=np.float32)
print(f"Loaded {len(working_input)} floats")

# Reshape to [batch, features, time]
# 69920 = 1 * 80 * 874
working_input = working_input.reshape(1, 80, 874)
print(f"Reshaped to: {working_input.shape}")

# Create session
session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Run inference
outputs = session.run(None, {
    "processed_signal": working_input,
    "processed_signal_length": np.array([874], dtype=np.int64)
})

# Get predictions
predictions = np.argmax(outputs[0][0], axis=-1)
print(f"\nFirst 10 predictions: {predictions[:10]}")
print(f"Expected: [1024 1024 1024 1024 1024 1024   15 1024   24 1024]")

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