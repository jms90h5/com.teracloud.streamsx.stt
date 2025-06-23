#!/usr/bin/env python3
"""
Test the exported NeMo FastConformer model
"""
import numpy as np
import onnxruntime as ort
import json

def test_model():
    # Load configuration
    with open("models/fastconformer_nemo_export/config.json", 'r') as f:
        config = json.load(f)
    
    print("=== Testing NeMo FastConformer Model ===")
    print(f"Model path: {config['model_path']}")
    print(f"Vocabulary size: {config['vocabulary_size']}")
    print(f"Blank ID: {config['blank_id']}")
    print(f"Sample rate: {config['sample_rate']}")
    print(f"Features: {config['n_mels']} mel bins")
    
    # Load model
    model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
    session = ort.InferenceSession(model_path)
    
    # Get input information
    for input in session.get_inputs():
        print(f"\nInput: {input.name}")
        print(f"  Shape: {input.shape}")
        print(f"  Type: {input.type}")
    
    # Get output information
    for output in session.get_outputs():
        print(f"\nOutput: {output.name}")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.type}")
    
    # Test with different input sizes
    print("\n=== Testing dynamic input sizes ===")
    for num_frames in [50, 100, 200, 500, 1000]:
        try:
            # Create dummy input - mel spectrogram features
            processed_signal = np.random.randn(1, config['n_mels'], num_frames).astype(np.float32)
            processed_signal_length = np.array([num_frames], dtype=np.int64)
            
            # Run inference
            outputs = session.run(None, {
                'processed_signal': processed_signal,
                'processed_signal_length': processed_signal_length
            })
            
            log_probs, encoded_lengths = outputs
            print(f"✓ {num_frames} frames -> log_probs shape: {log_probs.shape}, encoded_len: {encoded_lengths}")
            
            # Check output dimensions
            assert log_probs.shape[0] == 1  # batch size
            assert log_probs.shape[2] == config['vocabulary_size'] + 1  # vocab + blank
            
        except Exception as e:
            print(f"✗ {num_frames} frames: {str(e)[:100]}")
    
    print("\n✅ Model testing complete!")

if __name__ == "__main__":
    test_model()