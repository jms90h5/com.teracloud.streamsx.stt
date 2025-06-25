#!/usr/bin/env python3
"""
Export NVIDIA FastConformer model with preprocessing included
"""

import torch
import sys
import os
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_fastconformer_with_preprocessing.py <path_to_nemo_model>")
        sys.exit(1)
    
    nemo_path = sys.argv[1]
    if not os.path.exists(nemo_path):
        print(f"Error: Model file not found: {nemo_path}")
        sys.exit(1)
    
    print(f"Loading NeMo model from: {nemo_path}")
    
    # Load the model
    model = EncDecHybridRNNTCTCBPEModel.restore_from(nemo_path)
    model.eval()
    
    # Create output directory
    output_dir = "fastconformer_export_full"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nModel info:")
    print(f"  Sample rate: {model.cfg.sample_rate}")
    print(f"  Vocabulary size: {len(model.tokenizer.vocab)}")
    print(f"  Model type: {type(model).__name__}")
    
    # Export only the CTC decoder part for now
    # Create a wrapper that extracts just the CTC output
    class CTCWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, processed_signal, processed_signal_length):
            # Get encoder output
            encoded, encoded_len = self.model.encoder(
                audio_signal=processed_signal, 
                length=processed_signal_length
            )
            
            # Get CTC output
            log_probs = self.model.ctc_decoder(encoder_output=encoded)
            
            return log_probs, encoded_len
    
    # Create the wrapper
    ctc_model = CTCWrapper(model)
    ctc_model.eval()
    
    # Create example input - preprocessed features
    batch_size = 1
    n_mels = 80  # Standard mel features
    time_steps = 200  # About 2 seconds of features
    
    processed_signal = torch.randn(batch_size, n_mels, time_steps)
    processed_signal_length = torch.tensor([time_steps])
    
    # Export to ONNX
    output_path = os.path.join(output_dir, "ctc_model.onnx")
    print(f"\nExporting CTC model to: {output_path}")
    
    torch.onnx.export(
        ctc_model,
        (processed_signal, processed_signal_length),
        output_path,
        input_names=['processed_signal', 'processed_signal_length'],
        output_names=['log_probs', 'encoded_lengths'],
        dynamic_axes={
            'processed_signal': {0: 'batch', 2: 'time'},
            'processed_signal_length': {0: 'batch'},
            'log_probs': {0: 'batch', 1: 'time'},
            'encoded_lengths': {0: 'batch'}
        },
        opset_version=14,
        verbose=True,
        do_constant_folding=True
    )
    
    print(f"\n✅ CTC model exported!")
    
    # Save tokenizer - NeMo uses SentencePiece tokenizer
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save the tokenizer model file if it exists
    if hasattr(model.tokenizer, 'tokenizer') and hasattr(model.tokenizer.tokenizer, 'save'):
        tokenizer_model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        model.tokenizer.tokenizer.save(tokenizer_model_path)
        print(f"Tokenizer model saved to: {tokenizer_model_path}")
    
    # Copy the tokenizer model from the extracted directory
    import shutil
    import glob
    extracted_dir = "extracted_fastconformer"
    if os.path.exists(extracted_dir):
        for sp_model in glob.glob(os.path.join(extracted_dir, "*tokenizer.model")):
            shutil.copy2(sp_model, os.path.join(tokenizer_dir, "tokenizer.model"))
            print(f"Copied tokenizer model from: {sp_model}")
    
    # Save vocab as tokens.txt for compatibility
    vocab_path = os.path.join(output_dir, "tokens.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in model.tokenizer.vocab:
            f.write(token + '\n')
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Save configuration
    import json
    config_path = os.path.join(output_dir, "config.json")
    config = {
        "sample_rate": model.cfg.sample_rate,
        "n_mels": n_mels,
        "vocabulary_size": len(model.tokenizer.vocab),
        "blank_id": len(model.tokenizer.vocab),  # Blank is usually the last token
        "model_path": output_path,
        "tokenizer_dir": tokenizer_dir,
        "tokens_path": vocab_path,
        "preprocessor": {
            "sample_rate": model.cfg.preprocessor.sample_rate,
            "features": model.cfg.preprocessor.features,
            "n_fft": model.cfg.preprocessor.n_fft,
            "window_size": model.cfg.preprocessor.window_size,
            "window_stride": model.cfg.preprocessor.window_stride,
            "window": model.cfg.preprocessor.window
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    print(f"\n✅ Export complete! All files saved to: {output_dir}")

if __name__ == "__main__":
    main()