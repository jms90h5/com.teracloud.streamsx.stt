#!/usr/bin/env python3
"""
Test ONNX model inference to understand shape requirements
"""

import numpy as np
import onnxruntime as ort
import sys

def test_model(model_path):
    """Test model with different input shapes"""
    
    print(f"Testing model: {model_path}")
    
    # Create session
    session = ort.InferenceSession(model_path)
    
    # Get input info
    input_info = session.get_inputs()[0]
    print(f"\nInput name: {input_info.name}")
    print(f"Input shape: {input_info.shape}")
    print(f"Input type: {input_info.type}")
    
    # Try different sequence lengths
    test_lengths = [100, 125, 200, 250, 500, 1000]
    
    for seq_len in test_lengths:
        try:
            # Create dummy input [batch, seq_len, features]
            dummy_input = np.random.randn(1, seq_len, 80).astype(np.float32)
            
            print(f"\nTesting with sequence length {seq_len}...")
            print(f"Input shape: {dummy_input.shape}")
            
            # Run inference
            outputs = session.run(None, {input_info.name: dummy_input})
            
            print(f"✓ Success! Output shape: {outputs[0].shape}")
            print(f"  Subsampling factor: {seq_len / outputs[0].shape[1]:.1f}")
            
        except Exception as e:
            print(f"✗ Failed with error: {str(e)}")
            # Try to extract the expected shape from error
            if "requested shape" in str(e):
                print(f"  Error details: {str(e).split('requested shape:')[1].split()[0] if 'requested shape:' in str(e) else 'N/A'}")

def main():
    if len(sys.argv) < 2:
        model_path = "models/fastconformer_ctc_export/model.onnx"
    else:
        model_path = sys.argv[1]
    
    test_model(model_path)

if __name__ == "__main__":
    main()