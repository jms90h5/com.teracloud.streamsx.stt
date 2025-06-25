#!/usr/bin/env python3
"""
Check ONNX model input/output shapes to understand requirements
"""

import onnx
import sys
import os

def analyze_onnx_model(model_path):
    """Analyze ONNX model to understand shape requirements"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return
        
    print(f"\nAnalyzing model: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    # Load model
    model = onnx.load(model_path)
    
    # Analyze inputs
    print("\n=== Model Inputs ===")
    for input in model.graph.input:
        print(f"\nInput: {input.name}")
        shape = []
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(f"  Shape: {shape}")
        print(f"  Type: {input.type.tensor_type.elem_type}")
    
    # Analyze outputs  
    print("\n=== Model Outputs ===")
    for output in model.graph.output:
        print(f"\nOutput: {output.name}")
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        print(f"  Shape: {shape}")
        print(f"  Type: {output.type.tensor_type.elem_type}")
    
    # Find reshape operations that might cause issues
    print("\n=== Reshape Operations ===")
    reshape_count = 0
    for node in model.graph.node:
        if node.op_type == "Reshape":
            reshape_count += 1
            print(f"\nReshape {reshape_count}: {node.name}")
            # Try to find shape info
            for attr in node.attribute:
                if attr.name == "shape":
                    print(f"  Target shape: {attr.ints}")
    
    print(f"\nTotal reshape operations: {reshape_count}")
    
    # Look for attention-related nodes
    print("\n=== Attention Layers ===")
    attention_count = 0
    for node in model.graph.node:
        if "self_attn" in node.name or "attention" in node.name.lower():
            attention_count += 1
            if attention_count <= 5:  # Show first 5
                print(f"  {node.name} ({node.op_type})")
    
    print(f"\nTotal attention-related nodes: {attention_count}")
    
    # Check for subsampling info
    print("\n=== Subsampling Info ===")
    for node in model.graph.node:
        if "subsample" in node.name.lower() or "downsample" in node.name.lower():
            print(f"  {node.name} ({node.op_type})")

def main():
    # Models to check
    models = [
        "opt/models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx",
        "opt/models/fastconformer_ctc_export/model.onnx"
    ]
    
    # Add command line argument if provided
    if len(sys.argv) > 1:
        models = [sys.argv[1]]
    
    for model_path in models:
        if os.path.exists(model_path):
            analyze_onnx_model(model_path)
        else:
            print(f"\nSkipping (not found): {model_path}")

if __name__ == "__main__":
    main()