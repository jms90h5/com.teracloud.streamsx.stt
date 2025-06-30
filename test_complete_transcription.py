#!/usr/bin/env python3
"""Complete transcription test with feature extraction"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa

def extract_log_mel_features(audio, sr=16000, n_mels=80, n_fft=512, 
                            hop_length=160, win_length=400, dither=1e-5):
    """Extract log mel features matching NeMo preprocessing"""
    
    # Add dither (small noise) as NeMo does
    if dither > 0:
        audio = audio + dither * np.random.randn(len(audio))
    
    # Use librosa for mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0  # Power spectrogram
    )
    
    # Convert to log scale (natural log, as NeMo uses)
    log_mel = np.log(mel_spec + 1e-10)
    
    return log_mel.astype(np.float32)

# Load audio
audio_path = "samples/audio/librispeech_3sec.wav"
audio, sr = sf.read(audio_path)
print(f"Loaded audio: {len(audio)} samples at {sr}Hz, duration: {len(audio)/sr:.2f}s")

# Extract features
features = extract_log_mel_features(audio, sr=sr)
print(f"Extracted features shape: {features.shape}")

# Prepare for model input [batch, features, time]
features_tensor = features.T.reshape(1, features.shape[0], features.shape[1])
print(f"Model input shape: {features_tensor.shape}")

# Load model
model_path = "opt/models/fastconformer_ctc_export/model.onnx"
session = ort.InferenceSession(model_path)
print(f"Model loaded: {model_path}")

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
length = np.array([features_tensor.shape[2]], dtype=np.int64)

outputs = session.run([output_name], {
    input_name: features_tensor,
    "length": length
})

logits = outputs[0]
print(f"Output shape: {logits.shape}")

# Decode (simple greedy decoding)
predictions = np.argmax(logits[0], axis=-1)

# Load vocabulary
with open("opt/models/fastconformer_ctc_export/tokens.txt", "r") as f:
    vocab = [line.strip() for line in f]
print(f"Vocabulary size: {len(vocab)}")

# Decode tokens - skip blanks (token 1024)
tokens = []
prev_token = -1
blank_id = 1024

for token_id in predictions:
    if token_id != prev_token and token_id != blank_id:
        tokens.append(vocab[token_id])
    prev_token = token_id

# Join tokens
text = "".join(tokens).replace("▁", " ").strip()
print(f"\nTranscription: '{text}'")
print("\nExpected: 'it was the first great song of his life...'")