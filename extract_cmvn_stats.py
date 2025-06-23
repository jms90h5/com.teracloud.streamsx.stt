#!/usr/bin/env python3
"""
Extract CMVN statistics from NeMo model
This should give us the normalization stats needed for feature extraction
"""

import os
import sys
import tarfile
import yaml
import torch
import json

def extract_cmvn_from_nemo(nemo_path):
    """Extract CMVN stats from .nemo file"""
    
    # .nemo files are tar archives
    print(f"Opening {nemo_path}...")
    
    with tarfile.open(nemo_path, 'r') as tar:
        # List contents
        print("\nContents of .nemo file:")
        for member in tar.getmembers():
            print(f"  {member.name}")
            
        # Look for model config
        config_member = None
        for member in tar.getmembers():
            if 'model_config.yaml' in member.name:
                config_member = member
                break
                
        if config_member:
            print(f"\nExtracting config from {config_member.name}")
            config_file = tar.extractfile(config_member)
            config = yaml.safe_load(config_file)
            
            # Look for preprocessor/normalization config
            if 'preprocessor' in config:
                print("\nPreprocessor config:")
                preprocessor = config['preprocessor']
                for key, value in preprocessor.items():
                    if 'norm' in key.lower() or 'cmvn' in key.lower() or 'stats' in key.lower():
                        print(f"  {key}: {value}")
                        
            # Check for normalize settings
            if 'normalize' in config:
                print(f"\nNormalize setting: {config['normalize']}")
                
        # Look for actual CMVN stats files
        stats_files = []
        for member in tar.getmembers():
            if 'cmvn' in member.name.lower() or 'stats' in member.name.lower():
                stats_files.append(member)
                
        if stats_files:
            print("\nFound stats files:")
            for stats_file in stats_files:
                print(f"  {stats_file.name}")
                # Extract and show content
                content = tar.extractfile(stats_file).read()
                print(f"    Size: {len(content)} bytes")
                
        # Try to load the model weights to find normalization
        weights_member = None
        for member in tar.getmembers():
            if member.name.endswith('.ckpt'):
                weights_member = member
                break
                
        if weights_member:
            print(f"\nFound weights: {weights_member.name}")
            # Note: Loading full weights might be memory intensive
            print("  (Skipping weight loading to avoid memory issues)")

if __name__ == "__main__":
    # Use the actual NeMo model file
    nemo_file = "/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/models/nemo_cache_aware_conformer/models--nvidia--stt_en_fastconformer_hybrid_large_streaming_multi/blobs/8db5289d5238aca839b84f5afdd66eb1aa64413feb9c2a915940b291012aff8b"
    
    if os.path.exists(nemo_file):
        extract_cmvn_from_nemo(nemo_file)
    else:
        print(f"Error: {nemo_file} not found")