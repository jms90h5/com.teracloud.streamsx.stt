#!/usr/bin/env python3
"""
Test what scale of features the NeMo model expects
"""

import numpy as np
import torch
import onnxruntime as ort

# Create test features with different scales
def test_feature_scales():
    model_path = "models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx"
    
    print("Testing feature scales for NeMo model...")
    
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Test different feature scales
    scales = [
        ("Raw log mel (-23 to 5)", -23.0, 5.0),
        ("Normalized (-1 to 1)", -1.0, 1.0),
        ("Positive log mel (0 to 30)", 0.0, 30.0),
        ("Large positive (0 to 100)", 0.0, 100.0),
        ("Zero mean unit var", -3.0, 3.0)
    ]
    
    for scale_name, min_val, max_val in scales:
        # Create features with this scale
        features = np.random.uniform(min_val, max_val, (1, 160, 80)).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: features})
        logits = outputs[0]
        
        # Get predictions
        predictions = np.argmax(logits[0], axis=-1)
        unique_preds = np.unique(predictions)
        
        print(f"\n{scale_name} [{min_val:.1f}, {max_val:.1f}]:")
        print(f"  Output shape: {logits.shape}")
        print(f"  Logits range: [{np.min(logits):.3f}, {np.max(logits):.3f}]")
        print(f"  Unique predictions: {unique_preds[:10]}...")
        print(f"  Most common token: {predictions[0]} (repeated {np.sum(predictions == predictions[0])} times)")

if __name__ == "__main__":
    test_feature_scales()