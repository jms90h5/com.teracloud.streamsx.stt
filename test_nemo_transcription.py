#!/usr/bin/env python3
"""
Test NeMo FastConformer with actual transcription
"""
import numpy as np
import onnxruntime as ort
import json
import wave
import struct

def load_wav_file(filename):
    """Load WAV file and return audio samples"""
    with wave.open(filename, 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        
        # Read all frames
        frames = wav.readframes(n_frames)
        
        # Convert to numpy array
        if sampwidth == 2:  # 16-bit
            audio = np.frombuffer(frames, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        # Convert to float32 and normalize
        audio = audio.astype(np.float32) / 32768.0
        
        # Handle stereo -> mono
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        return audio, framerate

def extract_mel_features(audio, sample_rate=16000, n_mels=80, n_fft=512, 
                        hop_length=160, win_length=400):
    """Extract mel-spectrogram features (simplified version)"""
    # This is a simplified version - in production, use librosa or torchaudio
    # For now, create dummy features of the right shape
    num_frames = (len(audio) - win_length) // hop_length + 1
    features = np.random.randn(n_mels, num_frames).astype(np.float32)
    
    # Add some structure to make it more realistic
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        if end <= len(audio):
            # Simple energy-based features
            energy = np.sum(audio[start:end] ** 2)
            features[:, i] *= np.sqrt(energy + 1e-8)
    
    return features

def greedy_ctc_decode(log_probs, vocab, blank_id):
    """Simple greedy CTC decoding"""
    # Get most likely tokens at each time step
    tokens = np.argmax(log_probs[0], axis=-1)
    
    # Remove blanks and repeated tokens
    decoded = []
    prev_token = blank_id
    
    for token in tokens:
        if token != blank_id and token != prev_token:
            decoded.append(token)
        prev_token = token
    
    # Convert tokens to text
    text = ""
    for token_id in decoded:
        if token_id < len(vocab):
            text += vocab[token_id]
    
    return text, decoded

def load_vocabulary(vocab_path):
    """Load vocabulary from tokens.txt"""
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.strip())
    return vocab

def main():
    # Load configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    print("=== NeMo FastConformer Transcription Test ===")
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Load vocabulary
    vocab = load_vocabulary("models/fastconformer_nemo_export/tokens.txt")
    print(f"Loaded vocabulary: {len(vocab)} tokens")
    
    # Test audio file
    audio_file = "test_data/audio/librispeech-1995-1837-0001.wav"
    print(f"\nLoading audio: {audio_file}")
    print("Expected: 'it was the first great song'")
    
    # Load audio
    audio, sample_rate = load_wav_file(audio_file)
    duration = len(audio) / sample_rate
    print(f"Audio: {len(audio)} samples, {duration:.2f} seconds, {sample_rate} Hz")
    
    # Extract features
    print("\nExtracting features...")
    features = extract_mel_features(
        audio, 
        sample_rate=config['sample_rate'],
        n_mels=config['n_mels'],
        n_fft=config['preprocessor']['n_fft'],
        hop_length=int(config['preprocessor']['window_stride'] * config['sample_rate']),
        win_length=int(config['preprocessor']['window_size'] * config['sample_rate'])
    )
    print(f"Features shape: {features.shape}")
    
    # Prepare input
    processed_signal = features[np.newaxis, :, :]  # Add batch dimension
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
    print(f"Token IDs: {tokens[:20]}...")  # First 20 tokens
    
    # Show confidence
    probs = np.exp(log_probs[0])  # Convert log probs to probs
    max_probs = np.max(probs, axis=-1)
    avg_confidence = np.mean(max_probs[:encoded_lengths[0]])
    print(f"Average confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    main()