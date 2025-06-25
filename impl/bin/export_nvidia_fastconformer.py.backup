#!/usr/bin/env python3
"""
Export NVIDIA FastConformer Hybrid Large Streaming Multi model to ONNX
This script specifically handles the nvidia/stt_en_fastconformer_hybrid_large_streaming_multi model
"""

import torch
import onnx
import os
import sys
import tarfile
import yaml
import json
from pathlib import Path

def extract_nemo_model(nemo_path, extract_dir):
    """Extract .nemo file contents"""
    print(f"Extracting {nemo_path} to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Use filter='data' to avoid the security warning in Python 3.9+
    with tarfile.open(nemo_path, 'r') as tar:
        tar.extractall(extract_dir, filter='data')
    
    return extract_dir

def load_model_config(extract_dir):
    """Load model configuration from extracted .nemo"""
    config_path = os.path.join(extract_dir, "model_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nModel Configuration:")
    print(f"  Model type: {config.get('model_type', 'Unknown')}")
    print(f"  Architecture: {config['encoder'].get('_target_', 'Unknown')}")
    
    # Extract key parameters
    encoder_config = config['encoder']
    print(f"\nEncoder Configuration:")
    print(f"  Features: {encoder_config.get('feat_in', 80)}")
    print(f"  Model dimension: {encoder_config.get('d_model', 512)}")
    print(f"  Layers: {encoder_config.get('n_layers', 17)}")
    print(f"  Attention heads: {encoder_config.get('n_heads', 8)}")
    
    # For CTC models
    if 'decoder' in config:
        decoder_config = config['decoder']
        print(f"\nDecoder Configuration:")
        print(f"  Vocabulary size: {decoder_config.get('num_classes', 1024)}")
    
    return config

def create_ctc_model_wrapper(config, checkpoint_path):
    """Create a wrapper for the CTC part of the hybrid model"""
    
    class CTCModelWrapper(torch.nn.Module):
        def __init__(self, encoder, decoder_ctc):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder_ctc
            
        def forward(self, audio_signal, length):
            # Encoder forward pass
            encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
            
            # CTC decoder forward pass
            # Conv1d expects [batch, channels, time] but encoder output is [batch, time, channels]
            encoded_transposed = encoded.transpose(1, 2)  # [B, C, T]
            log_probs = self.decoder(encoded_transposed)  # [B, vocab, T]
            log_probs = log_probs.transpose(1, 2)  # [B, T, vocab] for CTC loss
            
            return log_probs, encoded_len
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Create encoder
    from nemo.collections.asr.modules import ConformerEncoder
    encoder_config = config['encoder']
    
    # Adjust config for the specific model
    encoder = ConformerEncoder(
        feat_in=encoder_config.get('feat_in', 80),
        n_layers=encoder_config.get('n_layers', 17),
        d_model=encoder_config.get('d_model', 512),
        feat_out=-1,
        subsampling=encoder_config.get('subsampling', 'dw_striding'),
        subsampling_factor=encoder_config.get('subsampling_factor', 8),
        subsampling_conv_channels=encoder_config.get('subsampling_conv_channels', 256),
        causal_downsampling=encoder_config.get('causal_downsampling', True),
        ff_expansion_factor=encoder_config.get('ff_expansion_factor', 4),
        self_attention_model=encoder_config.get('self_attention_model', 'rel_pos'),
        n_heads=encoder_config.get('n_heads', 8),
        conv_kernel_size=encoder_config.get('conv_kernel_size', 9),  # Changed from 31 to 9
        conv_context_size=encoder_config.get('conv_context_size', 'causal'),
        dropout=encoder_config.get('dropout', 0.1),
        dropout_pre_encoder=encoder_config.get('dropout_pre_encoder', 0.1),
        dropout_emb=encoder_config.get('dropout_emb', 0.0),
        dropout_att=encoder_config.get('dropout_att', 0.1),
        pos_emb_max_len=encoder_config.get('pos_emb_max_len', 5000)
    )
    
    # Create CTC decoder - it's a 1D convolution in this model
    vocab_size = config['decoder'].get('vocab_size', 1024) + 1  # +1 for blank token
    decoder_ctc = torch.nn.Conv1d(encoder.d_model, vocab_size, kernel_size=1)
    
    # Load weights
    encoder_state = {}
    decoder_state = {}
    
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            encoder_state[k.replace('encoder.', '')] = v
        elif k.startswith('ctc_decoder.') or k.startswith('decoder.'):
            key = k.replace('ctc_decoder.', '').replace('decoder.', '')
            if 'decoder_layers.0.' in key:
                key = key.replace('decoder_layers.0.', '')
            decoder_state[key] = v
    
    encoder.load_state_dict(encoder_state, strict=False)
    if 'weight' in decoder_state:
        decoder_ctc.weight.data = decoder_state['weight']
    if 'bias' in decoder_state:
        decoder_ctc.bias.data = decoder_state['bias']
    
    model = CTCModelWrapper(encoder, decoder_ctc)
    model.eval()
    
    return model

def export_to_onnx(model, output_path, vocab_size):
    """Export model to ONNX with dynamic axes"""
    
    # Create dummy inputs
    batch_size = 1
    audio_len = 16000  # 1 second of audio
    audio_signal = torch.randn(batch_size, 80, audio_len // 80)  # [B, features, time]
    length = torch.tensor([audio_len // 80], dtype=torch.int32)
    
    print(f"\nExporting to ONNX with input shape: {audio_signal.shape}")
    
    # Export with dynamic axes
    dynamic_axes = {
        'audio_signal': {0: 'batch', 2: 'time'},
        'length': {0: 'batch'},
        'log_probs': {0: 'batch', 1: 'time'},
        'encoded_len': {0: 'batch'}
    }
    
    torch.onnx.export(
        model,
        (audio_signal, length),
        output_path,
        input_names=['audio_signal', 'length'],
        output_names=['log_probs', 'encoded_len'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\n✅ Model exported successfully to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Print model info
    print("\nONNX Model Information:")
    print(f"  Inputs:")
    for input in onnx_model.graph.input:
        print(f"    - {input.name}: {[d.dim_param if d.dim_param else d.dim_value for d in input.type.tensor_type.shape.dim]}")
    print(f"  Outputs:")
    for output in onnx_model.graph.output:
        print(f"    - {output.name}: {[d.dim_param if d.dim_param else d.dim_value for d in output.type.tensor_type.shape.dim]}")

def export_tokenizer(extract_dir, output_dir):
    """Export tokenizer/vocabulary"""
    # Look for tokenizer files with hash prefixes
    import glob
    
    # Find vocab.txt file (with or without hash prefix)
    vocab_files = glob.glob(os.path.join(extract_dir, "*vocab.txt"))
    if not vocab_files:
        vocab_files = glob.glob(os.path.join(extract_dir, "*tokenizer.vocab"))
    
    if vocab_files:
        src_path = vocab_files[0]
        dst_path = os.path.join(output_dir, 'tokens.txt')
        
        with open(src_path, 'r', encoding='utf-8') as f:
            tokens = f.read()
        
        # Ensure proper formatting
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(tokens)
        
        print(f"\n✅ Exported tokenizer from: {os.path.basename(src_path)}")
        print(f"   Saved to: {dst_path}")
        
        # Count tokens
        num_tokens = len(tokens.strip().split('\n'))
        print(f"   Vocabulary size: {num_tokens}")
        
        # Also copy the sentencepiece model if it exists
        sp_models = glob.glob(os.path.join(extract_dir, "*tokenizer.model"))
        if sp_models:
            sp_src = sp_models[0]
            sp_dst = os.path.join(output_dir, 'tokenizer.model')
            import shutil
            shutil.copy2(sp_src, sp_dst)
            print(f"   Also copied SentencePiece model to: {sp_dst}")
        
        return num_tokens
    
    print("\n⚠️  No tokenizer file found, creating default")
    return 1024

def main():
    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python export_nvidia_fastconformer.py <path_to_nemo_model>")
        print("Expected model: stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
        sys.exit(1)
    
    nemo_path = sys.argv[1]
    
    if not os.path.exists(nemo_path):
        print(f"Error: Model file not found: {nemo_path}")
        sys.exit(1)
    
    # Check if it's the correct model
    if "fastconformer_hybrid_large_streaming" not in os.path.basename(nemo_path):
        print(f"Warning: Expected nvidia/stt_en_fastconformer_hybrid_large_streaming_multi model")
        print(f"Got: {os.path.basename(nemo_path)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Set up paths
    base_dir = os.path.dirname(nemo_path)
    extract_dir = os.path.join(base_dir, "extracted_fastconformer")
    output_dir = os.path.join(base_dir, "fastconformer_ctc_onnx")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract .nemo file
    extract_nemo_model(nemo_path, extract_dir)
    
    # Load configuration
    config = load_model_config(extract_dir)
    
    # Export tokenizer
    vocab_size = export_tokenizer(extract_dir, output_dir)
    
    # Create model wrapper
    checkpoint_path = os.path.join(extract_dir, "model_weights.ckpt")
    model = create_ctc_model_wrapper(config, checkpoint_path)
    
    # Export to ONNX
    output_path = os.path.join(output_dir, "fastconformer_ctc.onnx")
    export_to_onnx(model, output_path, vocab_size)
    
    # Save configuration
    config_output = {
        'model_type': 'FastConformer-CTC',
        'sample_rate': 16000,
        'features': 80,
        'subsampling_factor': 8,
        'vocab_size': vocab_size,
        'normalize': config.get('normalize', 'per_feature'),
        'model_path': output_path,
        'tokens_path': os.path.join(output_dir, 'tokens.txt')
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_output, f, indent=2)
    
    print(f"\n✅ Export complete! Files saved to: {output_dir}")
    print("\nTo use this model:")
    print(f"  Model: {output_path}")
    print(f"  Tokens: {os.path.join(output_dir, 'tokens.txt')}")
    print(f"  Config: {os.path.join(output_dir, 'config.json')}")

if __name__ == "__main__":
    main()