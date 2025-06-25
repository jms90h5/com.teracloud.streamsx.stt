#!/usr/bin/env python3
"""
Save exact features from Python implementation for comparison with C++
This will help us identify where the implementations diverge.
"""

import numpy as np
import soundfile as sf
import onnxruntime as ort
import json

def extract_log_mel_features(audio, sr=16000, n_mels=80, n_fft=512, 
                           hop_length=160, win_length=400, dither=0):
    """Extract log mel features matching NeMo's preprocessing exactly"""
    import librosa
    
    # Add dither (disabled for comparison)
    if dither > 0:
        audio = audio + dither * np.random.randn(len(audio))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=8000.0,
        power=2.0,
        norm=None,
        htk=False
    )
    
    # Natural log (not log10)
    log_mel = np.log(mel_spec + 1e-10)
    
    # Transpose to [time, features]
    log_mel = log_mel.T
    
    return log_mel

def main():
    # Load test audio
    audio_file = "opt/opt/test_data/audio/librispeech-1995-1837-0001.wav"
    audio, sr = sf.read(audio_file)
    print(f"Loaded audio: {len(audio)} samples at {sr} Hz")
    
    # Extract features
    features = extract_log_mel_features(audio)
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    
    # Save features for C++ comparison
    np.save("python_features.npy", features.astype(np.float32))
    print("Saved features to python_features.npy")
    
    # Also save as text for easier debugging
    with open("python_features.txt", "w") as f:
        f.write(f"Shape: {features.shape}\n")
        f.write(f"First 5 frames (first 10 values each):\n")
        for i in range(min(5, features.shape[0])):
            frame = features[i]
            values = " ".join([f"{v:.6f}" for v in frame[:10]])
            f.write(f"Frame {i}: {values}...\n")
    
    # Test with ONNX model
    print("\nTesting with ONNX model...")
    session = ort.InferenceSession("models/fastconformer_nemo_export/ctc_model.onnx")
    
    # Model expects [batch, features, time] (not [batch, time, features])
    # Transpose features from [time, features] to [features, time]
    features_transposed = features.T
    model_input = features_transposed[np.newaxis, :, :].astype(np.float32)
    input_length = np.array([features.shape[0]], dtype=np.int64)
    
    print(f"Model input shape: {model_input.shape}")
    print(f"Input length: {input_length}")
    
    # Run inference
    outputs = session.run(None, {
        "processed_signal": model_input,
        "processed_signal_length": input_length
    })
    
    logits = outputs[0]
    print(f"Output shape: {logits.shape}")
    
    # Save logits for comparison
    np.save("python_logits.npy", logits)
    
    # Get predictions
    predictions = np.argmax(logits[0], axis=-1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 10 predictions: {predictions[:10]}")
    print(f"Unique predictions: {np.unique(predictions)}")
    
    # Decode
    with open("models/fastconformer_nemo_export/tokens.txt", "r") as f:
        vocab = [line.strip() for line in f]
    
    # Simple CTC decode (remove blanks and duplicates)
    prev_token = -1
    decoded_tokens = []
    for token_id in predictions:
        if token_id != 1024 and token_id != prev_token:  # Not blank and not duplicate
            if token_id < len(vocab):
                decoded_tokens.append(vocab[token_id])
        prev_token = token_id
    
    text = "".join(decoded_tokens).replace("▁", " ").strip()
    print(f"\nTranscription: '{text}'")
    
    # Save all debug info
    debug_info = {
        "audio_samples": len(audio),
        "sample_rate": sr,
        "feature_shape": list(features.shape),
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
        "feature_mean": float(features.mean()),
        "model_input_shape": list(model_input.shape),
        "output_shape": list(logits.shape),
        "unique_predictions": list(map(int, np.unique(predictions))),
        "transcription": text
    }
    
    with open("python_debug_info.json", "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print("\nSaved all debug information for C++ comparison")

if __name__ == "__main__":
    main()