#!/usr/bin/env python3
"""
Simple download of NVIDIA NeMo FastConformer Hybrid streaming model checkpoint.
If huggingface_hub is missing, installs it automatically.
"""
import os
import sys
import subprocess

def install_hf_hub():
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface-hub'], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_model():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('huggingface_hub not found; installing...')
        if not install_hf_hub():
            print('Failed to install huggingface-hub'); return None
        from huggingface_hub import hf_hub_download

    repo = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    filename = 'stt_en_fastconformer_hybrid_large_streaming_multi.nemo'
    model_dir = 'models/nemo_cache_aware_conformer'
    os.makedirs(model_dir, exist_ok=True)
    print(f'Downloading {filename} from {repo}...')
    path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=model_dir)
    print(f'✓ Downloaded model to: {path}')
    return path

def main():
    print('=== Download NeMo Hybrid CTC+RNNT Model ===')
    path = download_model()
    if not path:
        sys.exit(1)
    print('\nModel ready for export:')
    print(f'  {path}')

if __name__=='__main__':
    main()