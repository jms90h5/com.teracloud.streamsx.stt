#!/usr/bin/env python3
"""
Extract features exactly as the working test_nemo_correct.py does
"""
import numpy as np
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

# Load test audio
audio, sr = librosa.load("test_audio_16k.wav", sr=16000, mono=True)
print(f"Loaded audio: {audio.shape}, sample rate: {sr}")

# Extract features
log_mel = extract_log_mel_features(audio)
print(f"Feature shape: {log_mel.shape}")  # [n_mels, n_frames]

# Transpose to [time, features] for comparison
features_time_major = log_mel.T
print(f"Time-major shape: {features_time_major.shape}")

# Stats
print(f"\nFeature stats:")
print(f"  Min: {log_mel.min():.4f}, Max: {log_mel.max():.4f}")
print(f"  Mean: {log_mel.mean():.4f}, Std: {log_mel.std():.4f}")

# First frame (time-major)
print(f"\nFirst frame features (first 5):")
print(f"  {features_time_major[0, :5]}")

# Save in both formats
np.save("correct_features_mel_time.npy", log_mel)  # [mel, time] as model expects
np.save("correct_features_time_mel.npy", features_time_major)  # [time, mel] for comparison

# Also test with model
import onnxruntime as ort

session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")

# Model expects [batch, mel, time]
model_input = log_mel[np.newaxis, :, :].astype(np.float32)
length_input = np.array([log_mel.shape[1]], dtype=np.int64)  # number of time frames

print(f"\nModel input shape: {model_input.shape}")
print(f"Length input: {length_input}")

# Run inference
outputs = session.run(None, {
    "processed_signal": model_input,
    "processed_signal_length": length_input
})

predictions = np.argmax(outputs[0][0], axis=-1)
print(f"\nFirst 10 predictions: {predictions[:10]}")

# Save model input for C++ testing
model_input.tofile("correct_model_input.bin")
print(f"\nSaved model input to correct_model_input.bin ({model_input.size} floats)")