#!/usr/bin/env python3
"""
Test what preprocessing NeMo actually expects
"""
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import json

def load_audio_torch(filename, target_sr=16000):
    """Load audio using torchaudio (what NeMo uses)"""
    waveform, sample_rate = torchaudio.load(filename)
    
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform.squeeze().numpy()

def extract_fbank_features(audio, sample_rate=16000, n_mels=80, n_fft=512,
                          win_length=400, hop_length=160):
    """Extract log mel-filterbank features similar to NeMo"""
    # Convert to tensor
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Compute spectrogram
    spectrogram = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Convert to power spectrogram
    spectrogram = torch.abs(spectrogram) ** 2
    
    # Create mel filterbank
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=sample_rate/2,
        n_stft=n_fft//2 + 1
    )
    
    # Apply mel filterbank
    mel_spec = mel_scale(spectrogram)
    
    # Convert to log scale (NeMo uses natural log)
    log_mel = torch.log(mel_spec + 1e-10)
    
    # Transpose to [mels, time]
    log_mel = log_mel.transpose(0, 1)
    
    return log_mel.numpy()

def normalize_features(features, method='per_feature'):
    """Normalize features"""
    if method == 'per_feature':
        # Normalize each mel bin independently
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        return (features - mean) / (std + 1e-5)
    elif method == 'global':
        # Global normalization
        return (features - features.mean()) / (features.std() + 1e-5)
    else:
        return features

def main():
    # Configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    print("=== Testing NeMo-style preprocessing ===")
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Load vocabulary
    vocab = []
    with open("models/fastconformer_nemo_export/tokens.txt", 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    
    # Test audio
    audio_file = "test_data/audio/librispeech-1995-1837-0001.wav"
    print(f"\nLoading: {audio_file}")
    
    # Load audio with torchaudio
    audio = load_audio_torch(audio_file, target_sr=config['sample_rate'])
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Try different preprocessing approaches
    preprocessing_methods = [
        ("Raw log mel", lambda x: x),
        ("Per-feature norm", lambda x: normalize_features(x, 'per_feature')),
        ("Global norm", lambda x: normalize_features(x, 'global')),
        ("Scaled log mel", lambda x: x / 10.0),  # Some models use scaling
        ("Standardized [-1, 1]", lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1),
    ]
    
    for method_name, preprocess_fn in preprocessing_methods:
        print(f"\n--- Testing: {method_name} ---")
        
        # Extract features
        features = extract_fbank_features(
            audio,
            sample_rate=config['sample_rate'],
            n_mels=config['n_mels'],
            n_fft=config['preprocessor']['n_fft'],
            win_length=int(config['preprocessor']['window_size'] * config['sample_rate']),
            hop_length=int(config['preprocessor']['window_stride'] * config['sample_rate'])
        )
        
        # Apply preprocessing
        features = preprocess_fn(features)
        
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"Features mean: {features.mean():.3f}, std: {features.std():.3f}")
        
        # Prepare input
        processed_signal = features[np.newaxis, :, :].astype(np.float32)
        processed_signal_length = np.array([features.shape[1]], dtype=np.int64)
        
        # Run inference
        outputs = session.run(None, {
            'processed_signal': processed_signal,
            'processed_signal_length': processed_signal_length
        })
        
        log_probs, encoded_lengths = outputs
        
        # Decode
        predictions = np.argmax(log_probs[0], axis=-1)
        
        # Greedy CTC decode
        decoded = []
        prev_token = config['blank_id']
        for token in predictions:
            if token != config['blank_id'] and token != prev_token:
                decoded.append(token)
            prev_token = token
        
        # Convert to text
        text = ""
        for token_id in decoded:
            if token_id < len(vocab):
                text += vocab[token_id]
        
        # Count non-blank tokens
        non_blank = np.sum(predictions != config['blank_id'])
        print(f"Non-blank tokens: {non_blank}/{len(predictions)} ({non_blank/len(predictions)*100:.1f}%)")
        print(f"Transcription: '{text}'")
        
        if len(text) > 0:
            print("SUCCESS! Found working preprocessing method!")
            break
    
    print(f"\nExpected: 'it was the first great song'")

if __name__ == "__main__":
    main()