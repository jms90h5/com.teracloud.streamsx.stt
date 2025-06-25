#!/usr/bin/env python3
"""
Save Python features to binary file for comparison with C++
"""
import numpy as np
import librosa
import json

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

def main():
    # Configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    # Load audio
    audio_file = "opt/opt/test_data/audio/librispeech-1995-1837-0001.wav"
    audio, _ = librosa.load(audio_file, sr=config['sample_rate'], mono=True)
    
    print(f"Audio loaded: {len(audio)} samples")
    
    # Extract features (NO NORMALIZATION as per NeMo config)
    features = extract_log_mel_features(
        audio,
        sr=config['sample_rate'],
        n_mels=config['n_mels'],
        n_fft=config['preprocessor']['n_fft'],
        hop_length=int(config['preprocessor']['window_stride'] * config['sample_rate']),
        win_length=int(config['preprocessor']['window_size'] * config['sample_rate']),
        dither=1e-5  # As specified in NeMo config
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Features mean: {features.mean():.3f}, std: {features.std():.3f}")
    
    # First, save features in mel x time format (as extracted)
    features_mel_time = features  # [80, 874]
    
    # Then transpose to time x mel format (as C++ expects)
    features_time_mel = features.T  # [874, 80]
    
    # Save both formats
    with open("python_features_mel_time.bin", "wb") as f:
        features_mel_time.astype(np.float32).tofile(f)
    print(f"Saved features in [mel, time] format: {features_mel_time.shape}")
    
    with open("python_features_time_mel.bin", "wb") as f:
        features_time_mel.astype(np.float32).tofile(f)
    print(f"Saved features in [time, mel] format: {features_time_mel.shape}")
    
    # Also save as numpy for easier loading
    np.save("python_features_mel_time.npy", features_mel_time)
    np.save("python_features_time_mel.npy", features_time_mel)
    
    # Print first few values for comparison
    print(f"\nFirst 10 values (time x mel format):")
    flat_time_mel = features_time_mel.flatten()
    for i in range(10):
        print(f"  [{i}]: {flat_time_mel[i]:.6f}")

if __name__ == "__main__":
    main()