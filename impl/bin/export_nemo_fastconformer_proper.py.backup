#!/usr/bin/env python3
"""
Proper export of NVIDIA FastConformer model using NeMo's built-in export functionality
"""

import torch
import sys
import os
from nemo.collections.asr.models import EncDecCTCModelBPE

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_nemo_fastconformer_proper.py <path_to_nemo_model>")
        sys.exit(1)
    
    nemo_path = sys.argv[1]
    if not os.path.exists(nemo_path):
        print(f"Error: Model file not found: {nemo_path}")
        sys.exit(1)
    
    print(f"Loading NeMo model from: {nemo_path}")
    
    # Load the model using NeMo
    model = EncDecCTCModelBPE.restore_from(nemo_path)
    model.eval()
    
    # Set up output path
    output_dir = "fastconformer_export_proper"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.onnx")
    
    print(f"\nModel info:")
    print(f"  Sample rate: {model.cfg.sample_rate}")
    print(f"  Vocabulary size: {len(model.tokenizer.vocab)}")
    print(f"  Model type: {type(model).__name__}")
    
    # Create example input
    # The model expects preprocessed features, not raw audio
    # Check the model's preprocessor settings
    print(f"\nPreprocessor info:")
    print(f"  Features: {model.cfg.preprocessor.features}")
    print(f"  Sample rate: {model.cfg.preprocessor.sample_rate}")
    print(f"  Feature type: {model.cfg.preprocessor._target_}")
    
    # For mel-spectrogram features, we need [batch, features, time]
    batch_size = 1
    n_mels = model.cfg.preprocessor.features  # Usually 80
    time_steps = 200  # About 2 seconds of audio after feature extraction
    
    input_signal = torch.randn(batch_size, n_mels, time_steps)
    input_signal_length = torch.tensor([time_steps])
    
    # Export to ONNX using NeMo's built-in export
    print(f"\nExporting to ONNX: {output_path}")
    model.export(
        output_path,
        input_example=[input_signal, input_signal_length],
        verbose=True,
        check_trace=False,  # Skip trace checking for now
        onnx_opset_version=14
    )
    
    print(f"\n✅ Export complete!")
    print(f"Model exported to: {output_path}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    model.tokenizer.save_tokenizer(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    # Save model config
    import json
    config_path = os.path.join(output_dir, "config.json")
    config = {
        "sample_rate": model.cfg.sample_rate,
        "labels": model.cfg.labels,
        "normalize": "per_feature",
        "model_path": output_path,
        "tokenizer_dir": tokenizer_path
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

if __name__ == "__main__":
    main()