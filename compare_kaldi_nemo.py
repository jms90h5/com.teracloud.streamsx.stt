#!/usr/bin/env python3
"""
Compare Kaldi and NeMo features
"""

import numpy as np

# Load Kaldi features (80 x 125)
kaldi_features = np.fromfile("kaldi_features.bin", dtype=np.float32).reshape(80, 125)
print(f"Kaldi features shape: {kaldi_features.shape}")
print(f"Kaldi stats - Min: {kaldi_features.min():.4f}, Max: {kaldi_features.max():.4f}, Mean: {kaldi_features.mean():.4f}")

# Load NeMo features
nemo_features = np.fromfile("nemo_features_125.bin", dtype=np.float32).reshape(80, 125)
print(f"\nNeMo features shape: {nemo_features.shape}")
print(f"NeMo stats - Min: {nemo_features.min():.4f}, Max: {nemo_features.max():.4f}, Mean: {nemo_features.mean():.4f}")

# Compare first frame
print(f"\nFirst frame comparison (first 10 features):")
kaldi_frame0 = kaldi_features[:, 0]
nemo_frame0 = nemo_features[:, 0]

for i in range(10):
    diff = kaldi_frame0[i] - nemo_frame0[i]
    print(f"  Feature {i}: Kaldi={kaldi_frame0[i]:.4f}, NeMo={nemo_frame0[i]:.4f}, diff={diff:.4f}")

# Overall comparison
print(f"\nOverall differences:")
print(f"  Mean absolute difference: {np.mean(np.abs(kaldi_features - nemo_features)):.4f}")
print(f"  Max absolute difference: {np.max(np.abs(kaldi_features - nemo_features)):.4f}")

# Test with model
import onnxruntime as ort

session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Test with Kaldi features
model_input = kaldi_features[np.newaxis, :, :].astype(np.float32)
outputs = session.run(None, {
    "processed_signal": model_input,
    "processed_signal_length": np.array([125], dtype=np.int64)
})

predictions = np.argmax(outputs[0][0], axis=-1)
print(f"\nKaldi features predictions: {predictions[:10]}")

# Decode
vocab = []
with open("models/fastconformer_nemo_export/tokens.txt", 'r') as f:
    for line in f:
        vocab.append(line.strip())

prev_token = -1
text = ""
for token_id in predictions:
    if token_id != 1024 and token_id != prev_token and token_id < len(vocab):
        token = vocab[token_id]
        if len(token) >= 3 and ord(token[0]) == 0xE2 and ord(token[1]) == 0x96 and ord(token[2]) == 0x81:
            if text:
                text += " "
            text += token[3:]
        else:
            text += token
    prev_token = token_id

print(f"Kaldi transcription: '{text}'")