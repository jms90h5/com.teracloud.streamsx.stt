#!/usr/bin/env python3
"""
Working NeMo export using compatibility patches
Based on successful legacy implementation from streamsx.stt.other
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv",
    category=RuntimeWarning,
    module="pydub.utils"
)

import sys

# WORKING SOLUTION: Patch huggingface_hub compatibility issues  
try:
    import huggingface_hub
    print(f"Found huggingface_hub version: {huggingface_hub.__version__}")
    
    # Check version and apply patches
    hf_ver = tuple(map(int, huggingface_hub.__version__.split('.')[:2]))
    
    if hf_ver >= (0, 20):
        print("Detected huggingface_hub >= 0.20.0, applying compatibility patches...")
        
        # Patch 1: Stub out list_repo_tree if missing (needed by transformers)
        if not hasattr(huggingface_hub, 'list_repo_tree'):
            print("  - Adding list_repo_tree stub")
            huggingface_hub.list_repo_tree = lambda *args, **kwargs: []
        
        # Patch 2: Add OfflineModeIsEnabled if missing
        try:
            import huggingface_hub.utils as hf_utils
            if not hasattr(hf_utils, 'OfflineModeIsEnabled'):
                print("  - Adding OfflineModeIsEnabled stub")
                hf_utils.OfflineModeIsEnabled = lambda *args, **kwargs: False
        except ImportError:
            pass
        
        # Patch 3: Create a dummy ModelFilter class if NeMo needs it
        if not hasattr(huggingface_hub, 'ModelFilter'):
            print("  - Adding ModelFilter stub for NeMo compatibility")
            class DummyModelFilter:
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, *args, **kwargs):
                    return True
            huggingface_hub.ModelFilter = DummyModelFilter
            
    print("✅ Compatibility patches applied successfully")
    
except ImportError:
    print("❌ huggingface_hub not found, please install it first")
    sys.exit(1)

# Now try to import NeMo
try:
    print("Importing NeMo ASR...")
    import nemo.collections.asr as nemo_asr
    print("✅ NeMo imported successfully!")
except ImportError as e:
    print(f"❌ NeMo import failed: {e}")
    print("\nTroubleshooting suggestions:")
    print("1. Try: pip install huggingface_hub==0.19.4")
    print("2. Or: pip install transformers==4.21.0")
    print("3. Check NEMO_PYTHON_DEPENDENCIES.md for detailed setup")
    sys.exit(1)

import os

def export_ctc_model():
    print("\n=== Exporting NeMo FastConformer Model in CTC Mode ===")
    
    # Create output directory
    output_dir = "models/fastconformer_ctc_export"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading NeMo FastConformer model...")
    
    # Try to use cached model first, fallback to download
    model_path = "models/nemo_cache_aware_conformer/models--nvidia--stt_en_fastconformer_hybrid_large_streaming_multi/snapshots/ae98143333690bd7ced4bc8ec16769bcb8918374/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    
    if os.path.exists(model_path):
        print(f"Using cached model: {model_path}")
        m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    else:
        print("Downloading model from HuggingFace...")
        m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
        )
    
    print("Setting CTC export configuration...")
    # CRITICAL: Export as CTC only, not hybrid RNNT
    m.set_export_config({'decoder_type': 'ctc'})
    
    # Additional CTC configuration (from legacy working code)
    m.change_decoding_strategy(decoder_type='ctc')
    
    print("Exporting to ONNX...")
    output_path = f"{output_dir}/model.onnx"
    m.export(output_path, check_trace=False)
    
    print("Generating tokens file...")
    # For hybrid models, vocabulary is accessed through multiple possible paths
    try:
        # Try multiple ways to access vocabulary (from legacy working code)
        if hasattr(m, 'tokenizer') and hasattr(m.tokenizer, 'vocab'):
            vocab = m.tokenizer.vocab
            vocab_list = [vocab.id_to_piece(i) for i in range(vocab.vocab_size())]
        elif hasattr(m, 'decoder') and hasattr(m.decoder, 'vocabulary'):
            vocab_list = m.decoder.vocabulary
        elif hasattr(m, 'ctc_decoder') and hasattr(m.ctc_decoder, 'vocabulary'):
            vocab_list = m.ctc_decoder.vocabulary
        else:
            # Fallback: use the tokenizer directly
            vocab_list = [m.tokenizer.ids_to_tokens([i])[0] for i in range(m.tokenizer.vocab_size)]
        
        with open(f'{output_dir}/tokens.txt', 'w') as f:
            for i, s in enumerate(vocab_list):
                f.write(f"{s} {i}\n")
            f.write(f"<blk> {i+1}\n")
        
        print("✅ CTC export completed successfully")
        print(f"Model: {output_path}")
        print(f"Tokens: {output_dir}/tokens.txt")
        
        # Print some model info
        print(f"\nModel Information:")
        print(f"Vocabulary size: {len(vocab_list)}")
        print(f"Sample tokens: {vocab_list[:10]}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate tokens file: {e}")
        print("Model export completed but tokens file generation failed.")
        print("You may need to generate tokens manually or use a different approach.")
        return False

if __name__ == "__main__":
    try:
        success = export_ctc_model()
        if success:
            print("\n🎉 Export completed successfully!")
            print("Next steps:")
            print("1. Test the exported model with: python test_ctc_export.py")
            print("2. Build C++ implementation using the CTC model")
        else:
            print("\n⚠️ Export completed with warnings")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\nFallback options:")
        print("1. Use pre-exported model: models/fastconformer_nemo_export/ctc_model.onnx")
        print("2. Try Docker approach as documented in NEMO_PYTHON_DEPENDENCIES.md")
        sys.exit(1)