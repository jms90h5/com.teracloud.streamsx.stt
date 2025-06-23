#!/usr/bin/env python3
"""Test full Python pipeline with large FastConformer model"""
import numpy as np
import onnxruntime as ort
import wave

def load_audio(filename):
    """Load audio from WAV file"""
    with wave.open(filename, 'rb') as wav:
        n_frames = wav.getnframes()
        frames = wav.readframes(n_frames)
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        sr = wav.getframerate()
    return audio, sr

def extract_features_simple(audio, sr=16000, n_mels=80, n_fft=512, 
                           hop_length=160, win_length=400, dither=1e-5):
    """Simple feature extraction without librosa"""
    # Add dither
    if dither > 0:
        audio = audio + dither * np.random.randn(len(audio))
    
    num_frames = (len(audio) - win_length) // hop_length + 1
    features = np.zeros((n_mels, num_frames), dtype=np.float32)
    
    # Hann window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / (win_length - 1))
    
    # Simple mel-scale filterbank (approximate)
    mel_filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        center = (i + 1) * (n_fft // 2) / (n_mels + 1)
        for j in range(n_fft // 2 + 1):
            if abs(j - center) < n_fft / (2 * n_mels):
                mel_filters[i, j] = 1.0 - abs(j - center) / (n_fft / (2 * n_mels))
    
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        end = start + win_length
        
        if end <= len(audio):
            # Window the frame
            frame = audio[start:end] * window
            
            # Pad to FFT size
            padded_frame = np.zeros(n_fft)
            padded_frame[:win_length] = frame
            
            # Simple DFT (not efficient but works)
            spectrum = np.fft.rfft(padded_frame)
            power_spectrum = np.abs(spectrum) ** 2
            
            # Apply mel filterbank
            mel_energies = np.dot(mel_filters, power_spectrum)
            
            # Log scale
            features[:, frame_idx] = np.log(mel_energies + 1e-10)
    
    # Adjust scale to match working features
    # Working features have mean=-3.91, std=2.77
    current_mean = features.mean()
    current_std = features.std()
    if current_std > 0:
        features = (features - current_mean) / current_std * 2.77 - 3.91
    
    return features

# Load audio
print("Loading audio...")
audio, sr = load_audio('test_data/audio/librispeech-1995-1837-0001.wav')
print(f"Audio: {len(audio)} samples, {len(audio)/sr:.2f} seconds")

# Extract features
print("\nExtracting features...")
features = extract_features_simple(audio)
print(f"Features shape: {features.shape}")
print(f"Features stats: min={features.min():.2f}, max={features.max():.2f}, mean={features.mean():.2f}, std={features.std():.2f}")

# Compare with working features stats
print("\nWorking features stats: min=-10.73, max=6.68, mean=-3.91, std=2.77")

# Load model
print("\nLoading model...")
session = ort.InferenceSession('models/fastconformer_nemo_export/ctc_model.onnx')

# Prepare inputs
input_features = features.astype(np.float32)
input_length = np.array([features.shape[1]], dtype=np.int64)

# Run inference
print("\nRunning inference...")
outputs = session.run(None, {
    'processed_signal': np.expand_dims(input_features, 0),
    'processed_signal_length': input_length
})

log_probs = outputs[0]
encoded_lengths = outputs[1]

print(f"Output shape: {log_probs.shape}")
print(f"Encoded length: {encoded_lengths[0]}")

# Get predictions
predictions = np.argmax(log_probs[0], axis=-1)
print(f"\nFirst 30 predictions: {predictions[:30]}")

# Check if all blank
unique_tokens = np.unique(predictions)
print(f"Unique tokens: {unique_tokens}")

if len(unique_tokens) == 1 and unique_tokens[0] == 1024:
    print("\n❌ Model is outputting only blank tokens!")
    print("This indicates the features are still not correct.")
else:
    print("\n✅ Model is producing varied tokens!")
    
    # Load vocabulary
    vocab = []
    with open('models/fastconformer_nemo_export/tokens.txt', 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    
    # Simple CTC decode
    decoded_tokens = []
    prev_token = -1
    for i in range(encoded_lengths[0]):
        token = predictions[i]
        if token != prev_token and token < len(vocab):
            decoded_tokens.append(vocab[token])
        prev_token = token
    
    print(f"\nDecoded tokens (first 20): {decoded_tokens[:20]}")
    
    # Handle BPE
    text = ""
    for token in decoded_tokens:
        if len(token) >= 3 and token.startswith('▁'):
            if text:
                text += " "
            text += token[1:]  # Remove ▁
        else:
            text += token
    
    print(f"\nTranscription: '{text}'")
    print(f"Expected: 'it was the first great song of his life'")