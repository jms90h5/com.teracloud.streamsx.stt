#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort

# Load saved correct features
features = np.load('correct_python_features_time_mel.npy')
print(f'Loaded features shape: {features.shape}')

# Take first 500 frames
if features.shape[0] >= 500:
    features = features[:500, :]
else:
    # Pad if needed
    pad_len = 500 - features.shape[0]
    features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')

# Reshape for model
input_data = features.reshape(1, 500, 80).astype(np.float32)
print(f'Input shape: {input_data.shape}')
print(f'Feature stats: min={input_data.min():.3f}, max={input_data.max():.3f}, mean={input_data.mean():.3f}')

# Load model and run
session = ort.InferenceSession('models/fastconformer_ctc_export/model.onnx')
outputs = session.run(None, {'audio_signal': input_data})
logits = outputs[0]

# Get predictions
predictions = np.argmax(logits[0], axis=-1)
print(f'Output shape: {logits.shape}')
print(f'Predictions (first 30): {predictions[:30]}')
print(f'Unique tokens: {list(np.unique(predictions))[:20]}...')

# Load vocab to decode
vocab = []
with open('models/fastconformer_ctc_export/tokens_with_ids.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            vocab.append(parts[0])

print(f'Vocab size: {len(vocab)}')
print(f'Token 414 is: {vocab[414] if 414 < len(vocab) else "OUT_OF_RANGE"}')

# Simple CTC decode
prev_token = -1
decoded_tokens = []
for token in predictions:
    if token != prev_token and token < len(vocab) and token != 1022:  # 1022 is blank
        decoded_tokens.append(vocab[token])
    prev_token = token

print(f'Decoded tokens: {decoded_tokens[:20]}')
print(f'Decoded text: {" ".join(decoded_tokens)}')