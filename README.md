# TeraCloud Streams Speech-to-Text Toolkit

A production-ready speech-to-text toolkit for Teracloud Streams that provides real-time transcription using state-of-the-art models including NVIDIA NeMo FastConformer and ONNX-based architectures.

## ⚠️ Status: PARTIALLY WORKING (June 23, 2025)

**IMPORTANT**: Previous "FULLY WORKING" status was incorrect. See `CRITICAL_FINDINGS_2025_06_23.md` for actual state.

### What Works:
- ✅ **Model inference** - ONNX Runtime correctly processes features when provided
- ✅ **Python testing** - `test_working_input_again.py` produces correct transcriptions
- ✅ **CTC decoding** - Properly handles BPE tokens and produces text

### What Needs Fixing:
- ❌ **C++ feature extraction** - Produces wrong feature statistics
- ❌ **Wrong model in tests** - Using 2.6MB model instead of 471MB model
- ❌ **SPL samples** - All produce empty transcript files

## Key Features

- **NVIDIA NeMo FastConformer Support** - State-of-the-art accuracy with streaming capability
- **Pure C++ Implementation** - No Python dependencies at runtime
- **ONNX Runtime Integration** - Efficient inference on CPU and GPU
- **Multiple Model Support** - Extensible architecture for different ASR models
- **Streaming-Ready** - Designed for real-time audio processing
- **SPL Operator Interface** - Easy integration with Streams applications

## Quick Start

### Prerequisites
- Teracloud Streams 7.2.0.1+
- C++14 compatible compiler
- ONNX Runtime 1.16.3 (included)
- Python 3.10+ (required for NeMo export scripts)
- FFmpeg CLI (for audio I/O via pydub/ffmpeg backend)
- FFmpeg CLI (for audio I/O via pydub/ffmpeg backend)
  ```bash
  # On Rocky Linux 9, enable EPEL and RPM Fusion to get ffmpeg:
  sudo dnf install -y epel-release
  # Some EPEL packages require CodeReady Builder (CRB)
  sudo dnf config-manager --set-enabled crb
  sudo dnf install -y dnf-plugins-core
  # Enable RPM Fusion (free & nonfree) for EL9
  sudo rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm
  sudo rpm -Uvh https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-9.noarch.rpm
  sudo dnf install -y ffmpeg
  ```

### Build and Test

```bash
# 1. Build the implementation library
cd impl
make clean && make

# 2. Test the standalone implementation
cd ..
./test_onnx_stt_operator models/fastconformer_nemo_export/ctc_model.onnx \
                         models/fastconformer_nemo_export/tokens.txt \
                         test_data/audio/librispeech_3sec.wav

# Expected output:
# Text: it was the first great song of his life
# Confidence: 0.917
# Real-time factor: 0.199
```

### Use in Streams Application

```spl
use com.teracloud.streamsx.stt::*;

composite SpeechTranscription {
    graph
        stream<blob audioChunk, uint64 audioTimestamp> Audio = 
            FileAudioSource() {
                param filename: "speech.wav";
            }
        
        stream<rstring text, boolean isFinal, float64 confidence> Result = 
            OnnxSTT(Audio) {
                param
                    encoderModel: "models/fastconformer_nemo_export/ctc_model.onnx";
                    vocabFile: "models/fastconformer_nemo_export/tokens.txt";
                    cmvnFile: "none";
                    modelType: "NEMO_CTC";
                    blankId: 1024;
            }
}
```

## Supported Models

### NVIDIA NeMo FastConformer CTC
- **Model**: `nvidia/stt_en_fastconformer_ctc_large`
- **Architecture**: Conformer with CTC head
- **Performance**: 0.2 RTF on CPU
- **Accuracy**: State-of-the-art for streaming ASR

### Export NeMo Model
```python
# Use provided export script
python export_fastconformer_with_preprocessing.py

# Or manually:
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_fastconformer_ctc_large")
model.export("ctc_model.onnx")
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Stream   │────▶│  OnnxSTT         │────▶│  Transcription  │
│  (16kHz mono)   │     │  SPL Operator    │     │  Stream         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ OnnxSTTInterface │
                        └──────────────────┘
                               │
                        ┌──────┴───────────┐
                        ▼                  ▼
                 ┌──────────────┐   ┌──────────────┐
                 │ NeMoCTCModel │   │ CacheAware   │
                 │              │   │ Conformer    │
                 └──────────────┘   └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ ImprovedFbank │
                 │ (Feature Ext) │
                 └──────────────┘
```

## Directory Structure

```
com.teracloud.streamsx.stt/
├── impl/                          # C++ implementation
│   ├── include/                   # Header files
│   ├── src/                       # Source files
│   └── lib/                       # Built libraries
├── com.teracloud.streamsx.stt/    # SPL operators
│   ├── OnnxSTT/                   # Main STT operator
│   └── FileAudioSource.spl        # Audio file reader
├── models/                        # Model files
│   └── fastconformer_nemo_export/ # NeMo model export
├── samples/                       # Example applications
│   └── NeMoCTCSample.spl         # NeMo example
├── test_data/                     # Test audio files
└── docs/                         # Documentation
```

## Performance

Benchmarked on Intel Xeon CPU with 4 threads:

| Model | Audio Length | Processing Time | RTF | Memory |
|-------|-------------|-----------------|-----|---------|
| NeMo FastConformer CTC | 3s | 600ms | 0.20 | 500MB |
| NeMo FastConformer CTC | 8.7s | 1700ms | 0.19 | 500MB |

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Quick Start](QUICK_START_NEXT_SESSION.md)
- [Operator Guide](OPERATOR_GUIDE.md)
- [Implementation Guide](docs/IMPLEMENTATION.md)
- [NeMo Integration Workflow](docs/NEMO_WORKFLOW.md)
- [Testing Strategy](docs/TESTING.md)
- [Update Plan](docs/STREAMS_OPERATOR_UPDATE_PLAN.md)
- [Scripts Overview](docs/SCRIPTS_OVERVIEW.md)
- [Archived Developer Tools](archive/dev/)
- [Archived Reference Docs](archive/docs/)

## Known Limitations

- CPU inference only (GPU support planned)
- English models only (multilingual support planned)
- Batch size 1 (batch processing planned)

## Contributing

1. Follow the existing code style
2. Add unit tests for new features
3. Update documentation
4. Test with provided audio samples

## License

Teracloud Proprietary - See LICENSE file for details

## Acknowledgments

- NVIDIA NeMo team for the excellent FastConformer models
- ONNX Runtime team for the inference framework
- Kaldi team for feature extraction algorithms