#!/usr/bin/env python3
"""Test model transcription with the librispeech sample"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
import os

# Load audio
audio_path = "samples/audio/librispeech_3sec.wav"
audio, sr = sf.read(audio_path)
print(f"Loaded audio: {len(audio)} samples at {sr}Hz")

# Load model
model_path = "opt/models/fastconformer_ctc_export/model.onnx"
session = ort.InferenceSession(model_path)
print(f"Model loaded: {model_path}")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Input: {input_name}, Output: {output_name}")

# Prepare input - model expects [batch, features, time]
# For raw audio, features dimension is 1
audio_tensor = audio.reshape(1, 1, -1).astype(np.float32)
audio_length = np.array([audio_tensor.shape[2]], dtype=np.int64)
inputs = {
    input_name: audio_tensor,
    "length": audio_length
}

# Run inference
outputs = session.run([output_name], inputs)
logits = outputs[0]
print(f"Output shape: {logits.shape}")

# Decode (simple greedy decoding)
predictions = np.argmax(logits[0], axis=-1)
print(f"Raw predictions shape: {predictions.shape}")

# Load vocabulary
with open("opt/models/fastconformer_ctc_export/tokens.txt", "r") as f:
    vocab = [line.strip() for line in f]
print(f"Vocabulary size: {len(vocab)}")

# Decode tokens
tokens = []
prev_token = -1
for token_id in predictions:
    if token_id != prev_token and token_id < len(vocab) - 1:  # Skip blank token
        tokens.append(vocab[token_id])
    prev_token = token_id

# Join tokens
text = "".join(tokens).replace("▁", " ").strip()
print(f"\nTranscription: '{text}'")
print("\nExpected: 'it was the first great song of his life...'")