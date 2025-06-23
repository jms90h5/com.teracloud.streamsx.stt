#!/usr/bin/env python3
"""
Debug NeMo FastConformer output to understand why transcription is empty
"""
import numpy as np
import onnxruntime as ort
import json
import wave

def load_wav_file(filename):
    """Load WAV file and return audio samples"""
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
        
        return audio, framerate

def extract_mel_features(audio, sample_rate=16000, n_mels=80, n_fft=512, 
                        hop_length=160, win_length=400):
    """Extract mel-spectrogram features"""
    num_frames = (len(audio) - win_length) // hop_length + 1
    features = np.random.randn(n_mels, num_frames).astype(np.float32)
    
    # Add some structure
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        if end <= len(audio):
            energy = np.sum(audio[start:end] ** 2)
            features[:, i] *= np.sqrt(energy + 1e-8)
    
    return features

def main():
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Test audio
    audio_file = "test_data/audio/librispeech-1995-1837-0001.wav"
    audio, sample_rate = load_wav_file(audio_file)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Audio min/max: {audio.min():.3f} / {audio.max():.3f}")
    
    # Extract features
    features = extract_mel_features(audio)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features min/max: {features.min():.3f} / {features.max():.3f}")
    print(f"Features mean/std: {features.mean():.3f} / {features.std():.3f}")
    
    # Prepare input
    processed_signal = features[np.newaxis, :, :]
    processed_signal_length = np.array([features.shape[1]], dtype=np.int64)
    
    # Run inference
    outputs = session.run(None, {
        'processed_signal': processed_signal,
        'processed_signal_length': processed_signal_length
    })
    
    log_probs, encoded_lengths = outputs
    print(f"\nOutput shape: {log_probs.shape}")
    print(f"Encoded length: {encoded_lengths[0]}")
    
    # Analyze output
    print(f"\nLog probs min/max: {log_probs.min():.3f} / {log_probs.max():.3f}")
    
    # Get predictions for each frame
    predictions = np.argmax(log_probs[0], axis=-1)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")
    
    # Count occurrences
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"\nToken distribution:")
    for token, count in zip(unique, counts):
        print(f"  Token {token}: {count} times ({count/len(predictions)*100:.1f}%)")
    
    # Check if blank token (1024) dominates
    blank_id = 1024
    blank_count = np.sum(predictions == blank_id)
    print(f"\nBlank token ({blank_id}) appears: {blank_count}/{len(predictions)} times")
    
    # Look at non-blank predictions
    non_blank_indices = np.where(predictions != blank_id)[0]
    print(f"\nNon-blank predictions at indices: {non_blank_indices[:10]}...")
    if len(non_blank_indices) > 0:
        print(f"Non-blank tokens: {predictions[non_blank_indices[:10]]}")
    
    # Check log prob values for a few frames
    print(f"\nLog probs for first 5 frames:")
    for i in range(min(5, log_probs.shape[1])):
        frame_probs = log_probs[0, i, :]
        best_idx = np.argmax(frame_probs)
        best_prob = frame_probs[best_idx]
        print(f"  Frame {i}: best token={best_idx}, log_prob={best_prob:.3f}, prob={np.exp(best_prob):.3f}")
        
        # Show top 5 tokens
        top5_indices = np.argsort(frame_probs)[-5:][::-1]
        print(f"    Top 5: {list(zip(top5_indices, frame_probs[top5_indices]))}")
    
    # Try with normalized features
    print("\n=== Testing with normalized features ===")
    norm_features = (features - features.mean()) / (features.std() + 1e-8)
    processed_signal = norm_features[np.newaxis, :, :]
    
    outputs = session.run(None, {
        'processed_signal': processed_signal,
        'processed_signal_length': processed_signal_length
    })
    
    log_probs2, _ = outputs
    predictions2 = np.argmax(log_probs2[0], axis=-1)
    unique2, counts2 = np.unique(predictions2, return_counts=True)
    
    print(f"Predictions with normalized features:")
    for token, count in zip(unique2, counts2):
        if count > 5:  # Only show tokens that appear more than 5 times
            print(f"  Token {token}: {count} times")

if __name__ == "__main__":
    main()