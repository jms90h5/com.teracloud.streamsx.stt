#!/usr/bin/env python3
"""
Simple test to find the correct input size for the model
"""
import numpy as np
import onnxruntime as ort

# Test the model
model_path = "models/fastconformer_ctc_export/model.onnx"
session = ort.InferenceSession(model_path)

# Get input info
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Try multiples of 4 (subsampling factor)
for frames in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    try:
        dummy_input = np.random.randn(1, frames, 80).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"✓ {frames} frames -> output shape: {outputs[0].shape}")
    except Exception as e:
        if "requested shape" in str(e):
            # Extract the expected shape
            error_msg = str(e)
            if "{125,8,64}" in error_msg:
                print(f"✗ {frames} frames: Model expects fixed 125 frames after subsampling")
            else:
                print(f"✗ {frames} frames: {error_msg.split('requested shape:')[1].split()[0] if 'requested shape:' in error_msg else 'Error'}")
        else:
            print(f"✗ {frames} frames: {str(e)[:100]}")