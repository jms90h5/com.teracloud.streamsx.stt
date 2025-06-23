#!/usr/bin/env python3
"""
Test NeMo FastConformer with proper mel-spectrogram extraction using librosa
"""
import numpy as np
import onnxruntime as ort
import json
import wave
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available. Install with: pip install librosa")

def load_audio_librosa(filename, sr=16000):
    """Load audio file using librosa"""
    audio, _ = librosa.load(filename, sr=sr, mono=True)
    return audio

def extract_mel_features_librosa(audio, sr=16000, n_mels=80, n_fft=512, 
                                hop_length=160, win_length=400):
    """Extract mel-spectrogram using librosa"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    
    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    return log_mel.astype(np.float32)

def load_vocabulary(vocab_path):
    """Load vocabulary"""
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    return vocab

def greedy_ctc_decode(log_probs, vocab, blank_id):
    """Greedy CTC decoding"""
    tokens = np.argmax(log_probs[0], axis=-1)
    
    decoded = []
    prev_token = blank_id
    
    for token in tokens:
        if token != blank_id and token != prev_token:
            decoded.append(token)
        prev_token = token
    
    # Convert to text
    text = ""
    for token_id in decoded:
        if token_id < len(vocab):
            text += vocab[token_id]
    
    return text, decoded

def main():
    if not HAS_LIBROSA:
        print("This test requires librosa. Please install it first.")
        return
    
    # Configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    print("=== NeMo FastConformer Test with Librosa ===")
    print(f"Sample rate: {config['sample_rate']} Hz")
    print(f"Features: {config['n_mels']} mel bins")
    print(f"FFT: {config['preprocessor']['n_fft']}")
    print(f"Window: {config['preprocessor']['window_size']}s ({int(config['preprocessor']['window_size'] * config['sample_rate'])} samples)")
    print(f"Stride: {config['preprocessor']['window_stride']}s ({int(config['preprocessor']['window_stride'] * config['sample_rate'])} samples)")
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Load vocabulary
    vocab = load_vocabulary("models/fastconformer_nemo_export/tokens.txt")
    print(f"\nVocabulary: {len(vocab)} tokens")
    
    # Test audio
    audio_file = "test_data/audio/librispeech-1995-1837-0001.wav"
    print(f"\nLoading: {audio_file}")
    
    # Load audio with librosa
    audio = load_audio_librosa(audio_file, sr=config['sample_rate'])
    duration = len(audio) / config['sample_rate']
    print(f"Audio: {len(audio)} samples, {duration:.2f} seconds")
    
    # Extract features
    print("\nExtracting mel-spectrogram...")
    features = extract_mel_features_librosa(
        audio,
        sr=config['sample_rate'],
        n_mels=config['n_mels'],
        n_fft=config['preprocessor']['n_fft'],
        hop_length=int(config['preprocessor']['window_stride'] * config['sample_rate']),
        win_length=int(config['preprocessor']['window_size'] * config['sample_rate'])
    )
    
    print(f"Features shape: {features.shape}")
    print(f"Features stats: mean={features.mean():.3f}, std={features.std():.3f}")
    
    # Prepare input
    processed_signal = features[np.newaxis, :, :]
    processed_signal_length = np.array([features.shape[1]], dtype=np.int64)
    
    # Run inference
    print("\nRunning inference...")
    outputs = session.run(None, {
        'processed_signal': processed_signal,
        'processed_signal_length': processed_signal_length
    })
    
    log_probs, encoded_lengths = outputs
    print(f"Output shape: {log_probs.shape}")
    print(f"Encoded length: {encoded_lengths[0]}")
    
    # Decode
    print("\nDecoding...")
    text, tokens = greedy_ctc_decode(log_probs, vocab, config['blank_id'])
    
    print(f"\nTranscription: '{text}'")
    print(f"Tokens decoded: {len(tokens)}")
    if tokens:
        print(f"First 10 tokens: {tokens[:10]}")
    
    # Analyze predictions
    predictions = np.argmax(log_probs[0], axis=-1)
    unique, counts = np.unique(predictions, return_counts=True)
    
    print(f"\nToken distribution:")
    non_blank_count = 0
    for token, count in zip(unique, counts):
        if count > 5:  # Only show frequent tokens
            is_blank = " (blank)" if token == config['blank_id'] else ""
            print(f"  Token {token}{is_blank}: {count} times ({count/len(predictions)*100:.1f}%)")
            if token != config['blank_id']:
                non_blank_count += count
    
    print(f"\nNon-blank tokens: {non_blank_count}/{len(predictions)} ({non_blank_count/len(predictions)*100:.1f}%)")
    
    # Expected output
    print(f"\nExpected: 'it was the first great song'")

if __name__ == "__main__":
    main()