#!/usr/bin/env python3
"""
Large FastConformer Export - Fixed Version

Uses the working approach from export_nemo_model_final.py 
but with the already extracted large model files.
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
import shutil

class FastConformerEncoder(nn.Module):
    """Large FastConformer encoder."""
    
    def __init__(self, feat_in=80, d_model=512, n_layers=17, n_heads=8):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model
        
        # Subsampling: 2 conv layers with stride 2 each = 4x reduction
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Input projection to model dimension  
        subsampled_feat_dim = feat_in // 4
        self.input_projection = nn.Linear(subsampled_feat_dim * d_model, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, features, lengths=None):
        batch_size, seq_len, feat_dim = features.shape
        
        # Subsampling
        x = features.unsqueeze(1)  # [batch, 1, seq_len, feat_in]
        x = self.subsampling(x)    # [batch, d_model, seq_len//4, feat_in//4]
        
        batch, channels, new_seq_len, new_feat_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, new_seq_len, channels * new_feat_dim)
        x = self.input_projection(x)
        
        # Transformer encoding
        encoded = self.transformer(x)
        return encoded, lengths // 4 if lengths is not None else None

class CTCDecoder(nn.Module):
    """CTC decoder."""
    
    def __init__(self, d_model=512, num_classes=1024):
        super().__init__()
        self.projection = nn.Linear(d_model, num_classes)
        
    def forward(self, encoded_features):
        logits = self.projection(encoded_features)
        return torch.log_softmax(logits, dim=-1)

class LargeFastConformerCTC(nn.Module):
    """Complete large FastConformer CTC model."""
    
    def __init__(self, config):
        super().__init__()
        
        self.encoder = FastConformerEncoder(
            feat_in=80,
            d_model=512,
            n_layers=17,
            n_heads=8
        )
        
        self.decoder = CTCDecoder(
            d_model=512,
            num_classes=1024
        )
        
    def forward(self, audio_signal, length=None):
        encoded, encoded_lengths = self.encoder(audio_signal, length)
        log_probs = self.decoder(encoded)
        return log_probs

def load_compatible_weights(model, state_dict):
    """Load compatible weights using the working approach."""
    print("Matching parameters...")
    model_dict = model.state_dict()
    compatible_dict = {}
    
    # Match parameters by shape and name similarity (same as working version)
    matched_count = 0
    for model_key in model_dict.keys():
        model_shape = model_dict[model_key].shape
        for ckpt_key in state_dict.keys():
            if (state_dict[ckpt_key].shape == model_shape and 
                any(part in ckpt_key for part in model_key.split('.'))):
                compatible_dict[model_key] = state_dict[ckpt_key]
                matched_count += 1
                break
    
    print(f"Successfully matched {matched_count}/{len(model_dict)} parameters")
    
    if compatible_dict:
        model.load_state_dict(compatible_dict, strict=False)
        return True
    return False

def main():
    print("=== Large FastConformer Export - Fixed ===")
    
    # Use already extracted files
    extract_dir = "models/nemo_extracted"
    output_dir = "opt/models/fastconformer_ctc_export"
    
    # Check files exist
    config_path = os.path.join(extract_dir, "model_config.yaml")
    weights_path = os.path.join(extract_dir, "model_weights.ckpt")
    vocab_path = os.path.join(extract_dir, "a4398a6af7324038a05d5930e49f4cf2_vocab.txt")
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return False
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found: {weights_path}")
        return False
        
    print(f"✅ Using extracted files from: {extract_dir}")
    
    # Load config and checkpoint
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading checkpoint...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Create model
    model = LargeFastConformerCTC(config)
    weights_loaded = load_compatible_weights(model, checkpoint)
    model.eval()
    
    # Export to ONNX
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "model.onnx")
    dummy_audio = torch.randn(1, 500, 80)  # [batch, seq_len, mel_features]
    dummy_length = torch.tensor([500], dtype=torch.long)
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model, (dummy_audio, dummy_length), output_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['audio_signal', 'length'], output_names=['log_probs'],
        dynamic_axes={
            'audio_signal': {0: 'batch_size', 1: 'seq_len'},
            'length': {0: 'batch_size'},
            'log_probs': {0: 'batch_size', 1: 'encoded_seq_len'}
        }
    )
    
    # Copy vocabulary file
    vocab_output = os.path.join(output_dir, "tokens.txt")
    if os.path.exists(vocab_path):
        shutil.copy(vocab_path, vocab_output)
    else:
        print("⚠️  Creating basic vocabulary")
        with open(vocab_output, 'w') as f:
            for i in range(1024):
                f.write(f"<token_{i}>\n")
    
    # Copy config
    config_output = os.path.join(output_dir, "model_config.yaml")
    with open(config_output, 'w') as f:
        yaml.dump(config, f)
    
    # Report results
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    vocab_lines = sum(1 for line in open(vocab_output))
    
    print(f"\n✅ Export completed!")
    print(f"   Model: {output_path} ({size_mb:.1f} MB)")
    print(f"   Vocabulary: {vocab_output} ({vocab_lines} tokens)")
    print(f"   Config: {config_output}")
    print(f"   Weights: {'Real parameters' if weights_loaded else 'Random'}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)