#!/usr/bin/env python3
# Suppress pydub ffmpeg/avconv warning if ffmpeg CLI is not installed
import warnings
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv",
    category=RuntimeWarning,
    module="pydub.utils"
)
import sys

# PEP 604 union syntax (e.g., str | List[str]) requires Python >= 3.10
if sys.version_info < (3, 10):
    sys.stderr.write(
        "ERROR: Python 3.10 or newer is required to run this export script.\n"
        "Please invoke with python3.10 (not python3.9).\n"
    )
    sys.exit(1)
"""
Properly export NeMo FastConformer model in CTC mode
"""
import sys
try:
    import huggingface_hub
    # Pin to <0.20.0 for ModelFilter support; transformers needs list_repo_tree & offline-mode API
    hf_ver = tuple(map(int, huggingface_hub.__version__.split('.')[:2]))
    if hf_ver >= (0, 20):
        sys.stderr.write(
            f"ERROR: huggingface_hub version {huggingface_hub.__version__} is not supported.\n"
            "Please install huggingface_hub<0.20.0, e.g.:\n"
            "  pip install huggingface_hub==0.19.4\n"
        )
        sys.exit(1)
    # Stub out APIs removed/added across Hub versions for Transformers compatibility
    if not hasattr(huggingface_hub, 'list_repo_tree'):
        huggingface_hub.list_repo_tree = lambda *args, **kwargs: []
    # Also patch offline-mode helper inside huggingface_hub.utils
    try:
        import huggingface_hub.utils as hf_utils
        if not hasattr(hf_utils, 'OfflineModeIsEnabled'):
            hf_utils.OfflineModeIsEnabled = lambda *args, **kwargs: False
    except ImportError:
        pass
except ImportError:
    sys.stderr.write("ERROR: huggingface_hub is required for model export.\n")
    sys.exit(1)

try:
    import nemo.collections.asr as nemo_asr
except ImportError as e:
    msg = str(e)
    if 'list_repo_tree' in msg or 'OfflineModeIsEnabled' in msg:
        sys.stderr.write(
            "ERROR: Incompatible huggingface_hub for transformers import.\n"
            "NeMo export requires both ModelFilter and list_repo_tree/OfflineModeIsEnabled,\n"
            "which are not available together in any single huggingface_hub version.\n"
            "Please refer to NEMO_PYTHON_DEPENDENCIES.md for setup guidance.\n"
        )
        sys.exit(1)
    raise
import os

def export_ctc_model():
    print("=== Exporting NeMo FastConformer Model in CTC Mode ===")
    
    # Create output directory
    output_dir = "models/fastconformer_ctc_export"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading NeMo FastConformer model...")
    model_path = "models/nemo_cache_aware_conformer/models--nvidia--stt_en_fastconformer_hybrid_large_streaming_multi/snapshots/ae98143333690bd7ced4bc8ec16769bcb8918374/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    
    # Load the model
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)
    
    print("Setting CTC export configuration...")
    # CRITICAL: Export as CTC only, not hybrid RNNT
    m.set_export_config({'decoder_type': 'ctc'})
    
    print("Exporting to ONNX...")
    output_path = f"{output_dir}/model.onnx"
    m.export(output_path)
    
    print("Generating tokens file...")
    # For hybrid models, vocabulary is accessed through the tokenizer
    try:
        # Try multiple ways to access vocabulary
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
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate tokens file: {e}")
        print("Model export completed but tokens file generation failed.")
        print("You may need to generate tokens manually or use a different approach.")

if __name__ == "__main__":
    export_ctc_model()