#!/usr/bin/env python3

"""
Fix tokens.txt format to include token IDs for NeMo CTC model
"""

def fix_tokens_file():
    tokens_path = "/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/opt/models/fastconformer_ctc_export/tokens.txt"
    output_path = "/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/opt/models/fastconformer_ctc_export/tokens_with_ids.txt"
    
    with open(tokens_path, 'r') as f:
        tokens = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(tokens)} tokens")
    
    # Write tokens with IDs
    with open(output_path, 'w') as f:
        for i, token in enumerate(tokens):
            f.write(f"{token} {i}\n")
    
    print(f"Wrote tokens with IDs to {output_path}")
    print(f"First few tokens: {tokens[:5]}")
    print(f"Last few tokens: {tokens[-5:]}")

if __name__ == "__main__":
    fix_tokens_file()