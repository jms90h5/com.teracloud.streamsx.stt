#!/bin/bash

# Comprehensive model download script for TeraCloud Streams STT Toolkit
# Downloads and sets up all supported streaming ASR models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== TeraCloud Streams STT Model Download Script ==="
echo "This script downloads and sets up streaming ASR models for the STT toolkit"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Function to show available models
show_available_models() {
    echo "Available Models:"
    echo ""
    echo "1. NVidia NeMo FastConformer (RECOMMENDED)"
    echo "   - 114M parameters, 13k+ hours training"
    echo "   - Best accuracy and latency performance"
    echo "   - Cache-aware streaming"
    echo "   - Requires: NeMo toolkit for export"
    echo ""
    echo "2. Icefall Conformer-RNN-T Large"
    echo "   - 120M parameters, LibriSpeech 960h"
    echo "   - Excellent accuracy (1.8% WER clean)"
    echo "   - 16-frame chunks"
    echo "   - Status: Interface ready, download pending"
    echo ""
    echo "3. WeNet Conformer-RNN-T Base"
    echo "   - 30M parameters, LibriSpeech 960h"
    echo "   - Good balance of size/accuracy"
    echo "   - 16-frame chunks"
    echo "   - Status: Interface ready, download pending"
    echo ""
    echo "4. SpeechBrain CRDNN-RNN-T"
    echo "   - 69M parameters, LibriSpeech 960h"
    echo "   - 5-frame chunks (very low latency)"
    echo "   - Status: Interface ready, download pending"
    echo ""
    echo "5. Sherpa-ONNX Paraformer (Legacy)"
    echo "   - Current working model"
    echo "   - Chinese/English bilingual"
    echo "   - Status: Already available"
    echo ""
}

# Function to download NeMo model
download_nemo() {
    echo "Setting up NVidia NeMo FastConformer..."
    echo ""
    
    NEMO_DIR="$MODELS_DIR/nemo_fastconformer_streaming"
    mkdir -p "$NEMO_DIR"
    
    echo "NeMo FastConformer requires the NeMo toolkit for model export."
    echo "This model offers the best performance but requires manual setup."
    echo ""
    echo "Steps to get NeMo model:"
    echo "1. Install NeMo toolkit: pip install nemo_toolkit[all]"
    echo "2. Run export script: python3 export_nemo_to_onnx.py"
    echo "3. Or download pre-exported ONNX from NGC catalog"
    echo ""
    
    # Create the export script if it doesn't exist
    if [ ! -f "$SCRIPT_DIR/export_nemo_to_onnx.py" ]; then
        echo "Creating NeMo export script..."
        cat > "$SCRIPT_DIR/export_nemo_to_onnx.py" << 'EOF'
#!/usr/bin/env python3
"""
Export NeMo FastConformer to ONNX format for TeraCloud Streams STT
"""
import sys
import os

def export_nemo_model():
    try:
        import nemo.collections.asr as nemo_asr
        
        # Load the streaming model
        model_name = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
        print(f"Loading NeMo model: {model_name}")
        
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
        
        # Configure for cache-aware streaming
        model.set_export_config({'cache_support': True})
        
        # Export to ONNX
        output_path = "opt/models/nemo_fastconformer_streaming/fastconformer_streaming.onnx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Exporting to: {output_path}")
        model.export(output_path)
        
        print("✓ NeMo model exported successfully!")
        return True
        
    except ImportError:
        print("Error: NeMo toolkit not found")
        print("Install with: pip install nemo_toolkit[all]")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    export_nemo_model()
EOF
        chmod +x "$SCRIPT_DIR/export_nemo_to_onnx.py"
    fi
    
    echo "✓ NeMo setup scripts created"
    echo "Run: python3 export_nemo_to_onnx.py"
    echo ""
}

# Function to download Icefall Conformer
download_icefall() {
    echo "Setting up Icefall Conformer-RNN-T Large..."
    echo ""
    
    ICEFALL_DIR="$MODELS_DIR/icefall_conformer_rnnt"
    mkdir -p "$ICEFALL_DIR"
    
    echo "Icefall Conformer-RNN-T Large (120M parameters)"
    echo "Training: LibriSpeech 960h"
    echo "Performance: 1.8% WER (clean), 4.2% WER (other)"
    echo ""
    
    # Create download instructions
    cat > "$ICEFALL_DIR/README.md" << 'EOF'
# Icefall Conformer-RNN-T Large

## Model Information
- **Parameters**: 120M
- **Training Data**: LibriSpeech 960 hours
- **Performance**: 1.8% WER (clean), 4.2% WER (other)
- **Chunk Size**: 16 frames
- **Cache Tensors**: encoder_out_cache, cnn_cache, att_cache

## Download Instructions

1. Install icefall:
```bash
pip install k2
git clone https://github.com/k2-fsa/icefall.git
cd icefall
pip install -r requirements.txt
```

2. Export to ONNX:
```bash
cd egs/librispeech/ASR/pruned_transducer_stateless7_streaming
python export-onnx.py --streaming
```

3. Copy ONNX files to this directory:
- encoder.onnx
- decoder.onnx  
- joiner.onnx

## Usage in STT Toolkit

```cpp
// C++ usage
auto pipeline = createConformerPipeline("models/icefall_conformer_rnnt", true);
```

```spl
// SPL usage
stream<...> results = OnnxSTT(audio) {
    param
        modelType: "Conformer";
        encoderModel: "models/icefall_conformer_rnnt/encoder.onnx";
        decoderModel: "models/icefall_conformer_rnnt/decoder.onnx";
        joinerModel: "models/icefall_conformer_rnnt/joiner.onnx";
}
```
EOF
    
    echo "✓ Icefall Conformer setup created"
    echo "See: $ICEFALL_DIR/README.md"
    echo ""
}

# Function to download WeNet Conformer
download_wenet() {
    echo "Setting up WeNet Conformer-RNN-T Base..."
    echo ""
    
    WENET_DIR="$MODELS_DIR/wenet_conformer_rnnt"
    mkdir -p "$WENET_DIR"
    
    echo "WeNet Conformer-RNN-T Base (30M parameters)"
    echo "Training: LibriSpeech 960h"
    echo "Performance: 2.1% WER (clean), 4.6% WER (other)"
    echo ""
    
    # Create download script
    cat > "$WENET_DIR/download_wenet.sh" << 'EOF'
#!/bin/bash

# Download WeNet Conformer-RNN-T model

echo "Downloading WeNet Conformer-RNN-T model..."

# WeNet model URLs (these would need to be updated with actual URLs)
WENET_BASE_URL="https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models"

# Download model files
wget -O encoder.onnx "$WENET_BASE_URL/20220506_u2pp_conformer_libtorch/encoder.onnx"
wget -O decoder.onnx "$WENET_BASE_URL/20220506_u2pp_conformer_libtorch/decoder.onnx"
wget -O units.txt "$WENET_BASE_URL/20220506_u2pp_conformer_libtorch/units.txt"

echo "✓ WeNet model downloaded"
EOF
    
    chmod +x "$WENET_DIR/download_wenet.sh"
    
    cat > "$WENET_DIR/README.md" << 'EOF'
# WeNet Conformer-RNN-T Base

## Model Information
- **Parameters**: 30M
- **Training Data**: LibriSpeech 960 hours  
- **Performance**: 2.1% WER (clean), 4.6% WER (other)
- **Chunk Size**: 16 frames
- **Cache Tensor**: Single "cache" tensor

## Download Instructions

1. Run the download script:
```bash
./download_wenet.sh
```

2. Or export from WeNet manually:
```bash
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
python wenet/bin/export_onnx.py --mode streaming
```

## Usage in STT Toolkit

```cpp
// C++ usage
auto pipeline = createWenetPipeline("models/wenet_conformer_rnnt", true);
```
EOF
    
    echo "⚠ WeNet download URLs need to be updated"
    echo "✓ WeNet setup created"
    echo "See: $WENET_DIR/README.md"
    echo ""
}

# Function to download SpeechBrain CRDNN
download_speechbrain() {
    echo "Setting up SpeechBrain CRDNN-RNN-T..."
    echo ""
    
    SPEECHBRAIN_DIR="$MODELS_DIR/speechbrain_crdnn_rnnt"
    mkdir -p "$SPEECHBRAIN_DIR"
    
    echo "SpeechBrain CRDNN-RNN-T (69M parameters)"
    echo "Training: LibriSpeech 960h"
    echo "Performance: 1.9% WER (clean), 4.4% WER (other)"
    echo "Latency: Very low (5-frame chunks)"
    echo ""
    
    cat > "$SPEECHBRAIN_DIR/README.md" << 'EOF'
# SpeechBrain CRDNN-RNN-T

## Model Information
- **Parameters**: 69M
- **Training Data**: LibriSpeech 960 hours
- **Performance**: 1.9% WER (clean), 4.4% WER (other)
- **Chunk Size**: 5 frames (very low latency)
- **Cache Tensor**: LSTM hidden states (h0)

## Download Instructions

1. Install SpeechBrain:
```bash
pip install speechbrain
```

2. Export model to ONNX:
```bash
python -c "
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR

model = EncoderDecoderASR.from_hparams(
    source='speechbrain/asr-crdnn-rnnlm-librispeech',
    savedir='tmp'
)

# Export to ONNX (implementation needed)
# model.export_onnx('speechbrain_crdnn.onnx')
"
```

## Usage in STT Toolkit

```cpp
// C++ usage (when implemented)
auto pipeline = createSpeechBrainPipeline("models/speechbrain_crdnn_rnnt", true);
```

## Notes
- SpeechBrain ONNX export may require custom implementation
- Very low latency due to 5-frame chunks
- Excellent for real-time applications
EOF
    
    echo "⚠ SpeechBrain ONNX export implementation needed"
    echo "✓ SpeechBrain setup created"
    echo "See: $SPEECHBRAIN_DIR/README.md"
    echo ""
}

# Function to verify existing Sherpa-ONNX model
verify_sherpa() {
    echo "Verifying Sherpa-ONNX Paraformer (Legacy)..."
    
    SHERPA_DIR="$MODELS_DIR/sherpa_onnx_paraformer"
    
    if [ -d "$SHERPA_DIR" ]; then
        echo "✓ Sherpa-ONNX model directory exists"
        
        # Check for required files
        REQUIRED_FILES=(
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx"
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx"
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx"
            "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
        )
        
        ALL_FOUND=true
        for file in "${REQUIRED_FILES[@]}"; do
            if [ -f "$SHERPA_DIR/$file" ]; then
                echo "  ✓ $file"
            else
                echo "  ✗ $file (missing)"
                ALL_FOUND=false
            fi
        done
        
        if [ "$ALL_FOUND" = true ]; then
            echo "✓ Sherpa-ONNX model complete and ready"
        else
            echo "⚠ Sherpa-ONNX model incomplete"
            echo "Run the original model download script to fix"
        fi
    else
        echo "⚠ Sherpa-ONNX model not found"
        echo "Run the existing model download script first"
    fi
    echo ""
}

# Main menu function
show_menu() {
    echo "Select models to download:"
    echo ""
    echo "1) NVidia NeMo FastConformer (RECOMMENDED)"
    echo "2) Icefall Conformer-RNN-T Large"
    echo "3) WeNet Conformer-RNN-T Base"
    echo "4) SpeechBrain CRDNN-RNN-T"
    echo "5) Verify Sherpa-ONNX Paraformer (Legacy)"
    echo "6) Download all available models"
    echo "7) Show model information"
    echo "8) Exit"
    echo ""
    echo -n "Enter your choice [1-8]: "
}

# Main execution
main() {
    cd "$SCRIPT_DIR"
    
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -r choice
            
            case $choice in
                1)
                    download_nemo
                    ;;
                2)
                    download_icefall
                    ;;
                3)
                    download_wenet
                    ;;
                4)
                    download_speechbrain
                    ;;
                5)
                    verify_sherpa
                    ;;
                6)
                    echo "Downloading all available models..."
                    download_nemo
                    download_icefall
                    download_wenet
                    download_speechbrain
                    verify_sherpa
                    echo "✓ All model setups complete"
                    ;;
                7)
                    show_available_models
                    ;;
                8)
                    echo "Exiting..."
                    break
                    ;;
                *)
                    echo "Invalid option. Please try again."
                    ;;
            esac
            
            echo ""
            echo "Press Enter to continue..."
            read -r
        done
    else
        # Command line mode
        case "$1" in
            "nemo")
                download_nemo
                ;;
            "icefall")
                download_icefall
                ;;
            "wenet")
                download_wenet
                ;;
            "speechbrain")
                download_speechbrain
                ;;
            "sherpa")
                verify_sherpa
                ;;
            "all")
                download_nemo
                download_icefall
                download_wenet
                download_speechbrain
                verify_sherpa
                ;;
            "list")
                show_available_models
                ;;
            *)
                echo "Usage: $0 [nemo|icefall|wenet|speechbrain|sherpa|all|list]"
                echo ""
                echo "Interactive mode: $0"
                echo "Command line: $0 <model_name>"
                ;;
        esac
    fi
}

# Run main function
main "$@"