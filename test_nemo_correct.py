#!/usr/bin/env python3
"""
Test NeMo FastConformer with correct preprocessing (no normalization)
"""
import numpy as np
import onnxruntime as ort
import json
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

def load_audio(filename, sr=16000):
    """Load audio file"""
    if HAS_LIBROSA:
        audio, _ = librosa.load(filename, sr=sr, mono=True)
    else:
        # Fallback to wave
        import wave
        with wave.open(filename, 'rb') as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            
            frames = wav.readframes(n_frames)
            
            if sampwidth == 2:
                audio = np.frombuffer(frames, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            audio = audio.astype(np.float32) / 32768.0
            
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
    
    return audio

def extract_log_mel_features(audio, sr=16000, n_mels=80, n_fft=512, 
                            hop_length=160, win_length=400, dither=1e-5):
    """Extract log mel features matching NeMo preprocessing"""
    
    # Add dither (small noise) as NeMo does
    if dither > 0:
        audio = audio + dither * np.random.randn(len(audio))
    
    if HAS_LIBROSA:
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
        
    else:
        # Simple fallback without librosa
        # This is approximate and won't give accurate results
        num_frames = (len(audio) - win_length) // hop_length + 1
        log_mel = np.zeros((n_mels, num_frames), dtype=np.float32)
        
        # Create Hann window
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / (win_length - 1))
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + win_length
            
            if end <= len(audio):
                # Apply window
                windowed = audio[start:end] * window
                
                # Simple energy computation per mel bin
                energy = np.sum(windowed ** 2)
                log_energy = np.log(energy + 1e-10)
                
                # Distribute across mel bins (very simplified)
                for j in range(n_mels):
                    # Create some variation across mel bins
                    mel_weight = np.exp(-0.5 * ((j - n_mels/2) / (n_mels/4)) ** 2)
                    log_mel[j, i] = log_energy + np.log(mel_weight + 1e-10)
    
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
    
    # Convert to text (handle BPE tokens)
    text = ""
    for token_id in decoded:
        if token_id < len(vocab):
            token = vocab[token_id]
            # Handle BPE tokens (remove ## or ▁)
            if token.startswith('##'):
                text += token[2:]
            elif token.startswith('▁'):
                text += ' ' + token[1:]
            else:
                text += token
    
    return text.strip(), decoded

def main():
    # Configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    print("=== NeMo FastConformer Test (Correct Preprocessing) ===")
    print(f"Using librosa: {HAS_LIBROSA}")
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Load vocabulary
    vocab = load_vocabulary("models/fastconformer_nemo_export/tokens.txt")
    print(f"Vocabulary: {len(vocab)} tokens")
    
    # Show some vocabulary entries
    print("\nFirst 10 vocabulary entries:")
    for i in range(min(10, len(vocab))):
        print(f"  {i}: '{vocab[i]}'")
    
    # Test audio
    audio_file = "test_data/audio/librispeech-1995-1837-0001.wav"
    print(f"\nLoading: {audio_file}")
    
    # Load audio
    audio = load_audio(audio_file, sr=config['sample_rate'])
    duration = len(audio) / config['sample_rate']
    print(f"Audio: {len(audio)} samples, {duration:.2f} seconds")
    
    # Extract features (NO NORMALIZATION as per NeMo config)
    print("\nExtracting log mel features (no normalization)...")
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
        print(f"Token IDs: {tokens[:20]}...")
        print(f"Tokens text: {[vocab[t] for t in tokens[:20]]}")
    
    # Analyze predictions
    predictions = np.argmax(log_probs[0], axis=-1)
    unique, counts = np.unique(predictions, return_counts=True)
    
    print(f"\nToken distribution:")
    for token, count in zip(unique, counts):
        if count > 5:
            is_blank = " (blank)" if token == config['blank_id'] else f" ('{vocab[token]}')"
            print(f"  Token {token}{is_blank}: {count} times")
    
    # Show frame-by-frame for first few frames
    print(f"\nFirst 10 frame predictions:")
    for i in range(min(10, len(predictions))):
        pred = predictions[i]
        conf = np.exp(log_probs[0, i, pred])
        token_str = vocab[pred] if pred < len(vocab) else "<blank>"
        print(f"  Frame {i}: {pred} ('{token_str}', conf={conf:.3f})")
    
    print(f"\nExpected: 'it was the first great song'")
    
    if not HAS_LIBROSA:
        print("\nNote: For accurate results, install librosa: pip install librosa")

if __name__ == "__main__":
    main()