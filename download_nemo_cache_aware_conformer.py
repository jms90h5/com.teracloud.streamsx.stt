#!/usr/bin/env python3
"""
Download and install the NVIDIA NeMo Cache-Aware Streaming FastConformer model
in CTC-only mode, including minimal ASR dependencies.
Model: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
114M parameters, cache-aware streaming support via cache tensors.
"""
import os
import sys
import subprocess

def install_nemo_toolkit_asr():
    """Install only the ASR subset of Nvidia NeMo toolkit via pip."""
    print("\nInstalling NeMo ASR toolkit (nemo_toolkit[asr])...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install',
            'nemo_toolkit[asr]', '--upgrade'
        ], check=True)
        print("✓ NeMo ASR toolkit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install NeMo ASR toolkit: {e}")
        return False

def check_nemo_installed():
    """Check if nemo_toolkit[asr] is available."""
    try:
        import nemo.collections.asr  # noqa: F401
        return True
    except ImportError:
        return False

def download_nemo_model():
    """Download the NeMo FastConformer streaming model checkpoint."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        # Ensure sentencepiece tokenizers are available before installing Nemo
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'sentencepiece'
        ], check=True)
        if not install_nemo_toolkit_asr():
            sys.exit(1)
        import nemo.collections.asr as nemo_asr

    model_name = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    print(f"\nDownloading pretrained model checkpoint: {model_name}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name)
    
    out_dir = 'models/nemo_cache_aware_conformer'
    os.makedirs(out_dir, exist_ok=True)
    nemo_path = os.path.join(out_dir, 'fastconformer_hybrid_streaming.nemo')
    model.save_to(nemo_path)
    print(f"✓ Model checkpoint saved to: {nemo_path}\n")
    return nemo_path

def main():
    print('=== NeMo Cache-Aware Streaming FastConformer Setup ===')
    nemopath = download_nemo_model()
    print('Model ready for export or further processing:')
    print(f'  {nemopath}')

if __name__ == '__main__':
    main()