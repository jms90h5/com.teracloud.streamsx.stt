#!/usr/bin/env python3
"""Test if NeMo is installed correctly"""

try:
    import nemo
    print(f"✓ NeMo version: {nemo.__version__}")
    
    import nemo.collections.asr as nemo_asr
    print("✓ NeMo ASR module loaded")
    
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    print("\n✅ NeMo installation successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install missing dependencies")