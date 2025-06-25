#!/usr/bin/env python3
"""
Save the exact working features from test_nemo_correct.py
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

# Load audio
audio_file = "opt/opt/test_data/audio/librispeech-1995-1837-0001.wav"
audio, sr = librosa.load(audio_file, sr=16000, mono=True)
print(f"Loaded audio: {audio.shape}, duration: {len(audio)/16000:.2f}s")

# Extract features exactly as test_nemo_correct.py does
features = extract_log_mel_features(
    audio,
    sr=16000,
    n_mels=80,
    n_fft=512,
    hop_length=160,
    win_length=400,
    dither=1e-5
)

print(f"Features shape: {features.shape}")
print(f"Features stats - Min: {features.min():.4f}, Max: {features.max():.4f}, Mean: {features.mean():.4f}")

# Save in both orientations
np.save("correct_python_features_mel_time.npy", features)  # [mel, time]
np.save("correct_python_features_time_mel.npy", features.T)  # [time, mel]

# Also save first 125 frames for comparison with C++
features_125 = features[:, :125]  # First 125 time frames
print(f"\nFirst 125 frames shape: {features_125.shape}")
print(f"First 5 values (mel-major): {features_125.flatten()[:5]}")

# Save binary for C++
features_125.tofile("correct_features_125.bin")
print(f"Saved {features_125.size} floats to correct_features_125.bin")