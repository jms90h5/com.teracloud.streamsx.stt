#!/usr/bin/env python3
"""
Test with unnormalized features
"""

import torch
import numpy as np
import onnxruntime as ort
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

# Load test audio
import librosa
audio, sr = librosa.load("test_data/audio/librispeech-1995-1837-0001.wav", sr=16000)

# Create NeMo feature extractor WITHOUT normalization
preprocessor = FilterbankFeatures(
    sample_rate=16000,
    n_window_size=400,
    n_window_stride=160,  
    window="hann",
    normalize=None,  # NO normalization
    n_fft=512,
    preemph=0.0,
    nfilt=80,
    lowfreq=0,
    highfreq=None,
    log=True,
    log_zero_guard_type="clamp",
    log_zero_guard_value=1e-10,
    dither=1e-5,
    pad_to=0,
    frame_splicing=1,
)

# Process audio
audio_tensor = torch.tensor(audio).unsqueeze(0)
audio_len = torch.tensor([len(audio)])

with torch.no_grad():
    features, _ = preprocessor(audio_tensor, audio_len)
    features_np = features.squeeze(0).numpy()

print(f"Features shape: {features_np.shape}")
print(f"Stats - Min: {features_np.min():.4f}, Max: {features_np.max():.4f}, Mean: {features_np.mean():.4f}")

# Create ONNX session
session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Model expects [batch, features, time]
model_input = features_np[np.newaxis, :, :].astype(np.float32)
length_input = np.array([features_np.shape[1]], dtype=np.int64)

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
        if len(token) >= 3 and ord(token[0]) == 0xE2 and ord(token[1]) == 0x96 and ord(token[2]) == 0x81:
            if text:
                text += " "
            text += token[3:]
        else:
            text += token
    prev_token = token_id

print(f"\nTranscription: '{text}'")
print(f"Expected: 'it was the first great sorrow of his life'")

# Save these features for C++ comparison
np.save("nemo_features_no_norm.npy", features_np)