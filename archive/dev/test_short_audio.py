#!/usr/bin/env python3
"""
Test with short audio clips to verify model accuracy
"""
import numpy as np
import onnxruntime as ort
import json
import wave
import os

# Import the working preprocessing from test_nemo_correct.py
from test_nemo_correct import load_audio, extract_log_mel_features, load_vocabulary, greedy_ctc_decode

def test_audio_file(session, vocab, config, audio_file, expected_text=None):
    """Test a single audio file"""
    print(f"\n{'='*60}")
    print(f"Testing: {audio_file}")
    if expected_text:
        print(f"Expected: '{expected_text}'")
    
    # Load audio
    audio = load_audio(audio_file, sr=config['sample_rate'])
    duration = len(audio) / config['sample_rate']
    print(f"Duration: {duration:.2f} seconds")
    
    # Extract features
    features = extract_log_mel_features(
        audio,
        sr=config['sample_rate'],
        n_mels=config['n_mels'],
        n_fft=config['preprocessor']['n_fft'],
        hop_length=int(config['preprocessor']['window_stride'] * config['sample_rate']),
        win_length=int(config['preprocessor']['window_size'] * config['sample_rate']),
        dither=1e-5
    )
    
    # Prepare input
    processed_signal = features[np.newaxis, :, :]
    processed_signal_length = np.array([features.shape[1]], dtype=np.int64)
    
    # Run inference
    outputs = session.run(None, {
        'processed_signal': processed_signal,
        'processed_signal_length': processed_signal_length
    })
    
    log_probs, encoded_lengths = outputs
    
    # Decode
    text, tokens = greedy_ctc_decode(log_probs, vocab, config['blank_id'])
    
    print(f"Transcription: '{text}'")
    
    # Calculate metrics if expected text provided
    if expected_text:
        # Simple word accuracy
        expected_words = expected_text.lower().split()
        actual_words = text.lower().split()
        
        # Find common words
        common = 0
        for word in actual_words:
            if word in expected_words:
                common += 1
        
        accuracy = common / max(len(expected_words), 1) * 100
        print(f"Word accuracy: {accuracy:.1f}% ({common}/{len(expected_words)} words)")
    
    return text

def main():
    # Load config
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Load vocabulary
    vocab = load_vocabulary("models/fastconformer_nemo_export/tokens.txt")
    
    print("=== NeMo FastConformer Audio Tests ===")
    
    # Test files with known/expected transcriptions
    test_cases = [
        # LibriSpeech files typically have transcriptions in their filenames or metadata
        ("test_data/audio/librispeech_3sec.wav", "it was the first great"),
        ("test_data/audio/test_16k.wav", None),  # Unknown transcription
        ("test_data/audio/2.wav", None),  # Short file
    ]
    
    results = []
    for audio_file, expected in test_cases:
        if os.path.exists(audio_file):
            result = test_audio_file(session, vocab, config, audio_file, expected)
            results.append((audio_file, result))
        else:
            print(f"\nSkipping {audio_file} (not found)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for audio_file, transcription in results:
        filename = os.path.basename(audio_file)
        print(f"{filename:30} -> '{transcription[:50]}{'...' if len(transcription) > 50 else ''}'")
    
    # Check README for expected transcriptions
    readme_path = "test_data/audio/README.txt"
    if os.path.exists(readme_path):
        print(f"\n{'='*60}")
        print("README content (may contain expected transcriptions):")
        with open(readme_path, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()