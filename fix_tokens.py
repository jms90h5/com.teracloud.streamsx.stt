#!/usr/bin/env python3
"""Extract tokens using the direct tokenizer approach"""

import os
import nemo.collections.asr as nemo_asr

# Load model
print("Loading model...")
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")

# Get tokenizer
tokenizer = model.tokenizer

print(f"Tokenizer type: {type(tokenizer)}")
print(f"Tokenizer has {tokenizer.tokenizer.get_piece_size()} pieces")

# Write tokens
output_path = "opt/models/fastconformer_ctc_export/tokens.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for i in range(tokenizer.tokenizer.get_piece_size()):
        piece = tokenizer.tokenizer.id_to_piece(i)
        f.write(f"{piece}\n")

print(f"✅ Wrote {tokenizer.tokenizer.get_piece_size()} tokens to {output_path}")

# Verify
with open(output_path, 'r') as f:
    lines = f.readlines()
    print(f"First 10 tokens: {[line.strip() for line in lines[:10]]}")
    print(f"Last token: {lines[-1].strip()}")
    print(f"Blank token ID (usually 1024): {lines[1024].strip() if len(lines) > 1024 else 'N/A'}")