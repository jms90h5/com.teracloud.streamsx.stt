#!/usr/bin/env python3
"""Check model input/output names and shapes"""
import onnxruntime as ort

model_path = "models/fastconformer_nemo_export/ctc_model.onnx"
session = ort.InferenceSession(model_path)

print("=== Model Input/Output Information ===")
print(f"Model: {model_path}")

print("\nInputs:")
for i, input_info in enumerate(session.get_inputs()):
    print(f"  {i}: {input_info.name}")
    print(f"     Shape: {input_info.shape}")
    print(f"     Type: {input_info.type}")

print("\nOutputs:")
for i, output_info in enumerate(session.get_outputs()):
    print(f"  {i}: {output_info.name}")
    print(f"     Shape: {output_info.shape}")
    print(f"     Type: {output_info.type}")