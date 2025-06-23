#!/usr/bin/env python3
"""Test the large FastConformer model with correct interface"""
import numpy as np
import onnxruntime as ort
import wave
import struct

# Load audio
print("Loading audio...")
with wave.open('test_data/audio/librispeech-1995-1837-0001.wav', 'rb') as wav:
    n_frames = wav.getnframes()
    frames = wav.readframes(n_frames)
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sr = wav.getframerate()

print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f} seconds")

# Simple log mel feature extraction
def extract_log_mel_features(audio, sr=16000, n_mels=80, n_fft=512, 
                            hop_length=160, win_length=400):
    """Extract log mel features (simplified)"""
    num_frames = (len(audio) - win_length) // hop_length + 1
    
    # Create simple features for testing
    features = np.zeros((n_mels, num_frames), dtype=np.float32)
    
    # Hann window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / (win_length - 1))
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        if end <= len(audio):
            frame = audio[start:end] * window
            # Simple energy-based features
            energy = np.sum(frame ** 2)
            log_energy = np.log(energy + 1e-10)
            # Distribute across mel bins
            for j in range(n_mels):
                features[j, i] = log_energy - j * 0.1
    
    return features

print("Extracting features...")
features = extract_log_mel_features(audio)
print(f"Features shape: {features.shape}")

# Load model
print("\nLoading model...")
session = ort.InferenceSession('models/fastconformer_nemo_export/ctc_model.onnx')

# Prepare inputs - note the transposed shape!
input_features = features.astype(np.float32)  # [80, time]
input_length = np.array([features.shape[1]], dtype=np.int64)

print(f"Input shape: {input_features.shape}")
print(f"Input length: {input_length}")

# Run inference
print("\nRunning inference...")
try:
    outputs = session.run(None, {
        'processed_signal': np.expand_dims(input_features, 0),  # [1, 80, time]
        'processed_signal_length': input_length
    })
    
    log_probs = outputs[0]  # [batch, time, vocab]
    encoded_lengths = outputs[1]
    
    print(f"Output shape: {log_probs.shape}")
    print(f"Encoded length: {encoded_lengths[0]}")
    
    # Get predictions
    predictions = np.argmax(log_probs[0], axis=-1)
    print(f"\nPredictions (first 30): {predictions[:30]}")
    print(f"Unique tokens: {np.unique(predictions)[:10]}...")
    
    # Load vocabulary
    vocab = []
    with open('models/fastconformer_nemo_export/tokens.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Simple CTC decode (1024 is blank for this model)
    decoded_tokens = []
    prev_token = -1
    for i in range(encoded_lengths[0]):
        token = predictions[i]
        if token != prev_token and token < len(vocab):
            decoded_tokens.append(vocab[token])
        prev_token = token
    
    print(f"\nDecoded tokens (first 20): {decoded_tokens[:20]}")
    print(f"Decoded text: {' '.join(decoded_tokens)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()