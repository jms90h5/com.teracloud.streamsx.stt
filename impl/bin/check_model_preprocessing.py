#!/usr/bin/env python3
"""
Check what preprocessing the FastConformer model expects
"""

import tarfile
import yaml

nemo_file = "/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/models/nemo_cache_aware_conformer/models--nvidia--stt_en_fastconformer_hybrid_large_streaming_multi/blobs/8db5289d5238aca839b84f5afdd66eb1aa64413feb9c2a915940b291012aff8b"

with tarfile.open(nemo_file, 'r') as tar:
    config_file = tar.extractfile('./model_config.yaml')
    config = yaml.safe_load(config_file)
    
    print("Preprocessor configuration:")
    if 'preprocessor' in config:
        for key, value in config['preprocessor'].items():
            print(f"  {key}: {value}")
    
    print("\nAudio preprocessing:")
    for key in ['sample_rate', 'window_size', 'window_stride', 'window', 
                'features', 'n_fft', 'n_mels', 'n_mfcc', 'dither', 'pad_to']:
        if key in config.get('preprocessor', {}):
            print(f"  {key}: {config['preprocessor'][key]}")
            
    print("\nModel expects:")
    print(f"  Normalization: {config['preprocessor'].get('normalize', 'None')}")
    print(f"  Features: {config['preprocessor'].get('features', 'Unknown')}")
    print(f"  Sample rate: {config['preprocessor'].get('sample_rate', 'Unknown')}")
    print(f"  Window size: {config['preprocessor'].get('window_size', 'Unknown')}")
    print(f"  Window stride: {config['preprocessor'].get('window_stride', 'Unknown')}")
    print(f"  Feature dimension: {config['preprocessor'].get('n_mels', 'Unknown')}")