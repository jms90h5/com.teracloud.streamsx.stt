# TeraCloud Streams Speech-to-Text Toolkit

A production-ready speech-to-text toolkit for Teracloud Streams that provides real-time transcription using state-of-the-art models including NVIDIA NeMo FastConformer and ONNX-based architectures.

## Status

**Current State**: Toolkit reorganized to follow Teracloud Streams standards. See `doc/MIGRATION_GUIDE.md` for recent changes.

**Working Components**:
- ✅ **Python model export** - `export_model_ctc_patched.py` handles NeMo dependency conflicts
- ✅ **C++ implementation** - Builds successfully with new directory structure
- ✅ **ONNX Runtime integration** - Inference pipeline functional
- ✅ **Automatic setup** - New `setup.sh` script for easy onboarding

**Known Issues**: See `doc/CRITICAL_FINDINGS_2025_06_23.md` for technical details.

**Working Verification**: To verify the model works correctly:
```bash
# This should produce: "it was the first great sorrow of his life..."
python3 test_working_input_again.py
```

## Key Features

- **NVIDIA NeMo FastConformer Support** - State-of-the-art accuracy with streaming capability
- **Pure C++ Implementation** - No Python dependencies at runtime
- **ONNX Runtime Integration** - Efficient inference on CPU and GPU
- **Multiple Model Support** - Extensible architecture for different ASR models
- **Streaming-Ready** - Designed for real-time audio processing
- **SPL Operator Interface** - Easy integration with Streams applications

## Quick Start

### 1. Clone and Setup (One Command!)

```bash
# Clone the repository
git clone <repository-url>
cd com.teracloud.streamsx.stt

# Run automatic setup - handles everything!
./setup.sh

# Activate development environment  
source .envrc
```

**That's it!** The setup script automatically:
- ✅ Checks all prerequisites
- ✅ Creates Python virtual environment
- ✅ Builds C++ implementation
- ✅ Sets up development aliases
- ✅ Verifies the installation

**Development Aliases**: After running `source .envrc`, you get:
- `stt-build` - Build C++ implementation
- `stt-test` - Run verification tests
- `stt-python` - Activate Python environment
- `stt-samples` - Check sample status

### 2. Export a Model (Required for Transcription)

```bash
# Install NeMo toolkit (one-time, large download ~2-3GB)
pip install -r doc/requirements_nemo.txt

# Export the working model (uses the proven script that handles dependency conflicts)
python impl/bin/export_model_ctc_patched.py

# This creates: opt/models/fastconformer_ctc_export/model.onnx (~44MB)
```

### 3. Test Everything Works

```bash
# Verify setup
stt-test

# Build and test samples
cd samples && make BasicNeMoDemo
```

### Prerequisites

The `setup.sh` script checks for these automatically:
- **Python 3.11+** with venv support
- **C++14 compiler** (g++)
- **Teracloud Streams 7.2.0.1+**
- **ONNX Runtime** (included in toolkit)
- **FFmpeg** (for audio processing):
  ```bash
  # Rocky Linux 9 / RHEL 9:
  sudo dnf install -y epel-release
  sudo dnf config-manager --set-enabled crb
  sudo rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm
  sudo dnf install -y ffmpeg
  
  # Ubuntu:
  sudo apt update && sudo apt install ffmpeg
  ```

### Troubleshooting Setup

**"Python 3.11 not found"**:
```bash
# Rocky Linux / RHEL:
sudo dnf install python3.11 python3.11-venv python3.11-pip

# Ubuntu:
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**"g++ not found"**:
```bash
# Rocky Linux / RHEL:
sudo dnf groupinstall 'Development Tools'

# Ubuntu:
sudo apt install build-essential
```

**"STREAMS_INSTALL not set"**:
```bash
# Source Streams environment first:
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh

# Then run setup:
./setup.sh
```

**C++ Compilation Fails with `NO_EXCEPTION` error**:
*   **Symptom**: The build fails with a compiler error similar to `error: expected unqualified-id before 'noexcept'` pointing to the ONNX Runtime headers.
*   **Cause**: ONNX Runtime's headers define a macro `NO_EXCEPTION` which conflicts with identifiers used in the Teracloud Streams SDK headers.
*   **Solution**: The toolkit now uses an isolation header `impl/include/onnx_wrapper.hpp`. This file includes the necessary ONNX headers and then immediately undefines the conflicting macro. If you are adding new C++ files that require the ONNX API, include `onnx_wrapper.hpp` instead of including `<onnxruntime_cxx_api.h>` directly.



### Manual Setup (Alternative)

```bash
# 1. Setup Python environment
./impl/setup_python_env.sh

# 2. Build the implementation library
cd impl
make clean && make

# 3. Test the implementation (requires model)
cd ..
source ./activate_python.sh
python impl/bin/export_model_ctc_patched.py  # Export model first
./test/verify_nemo_setup.sh                  # Verify setup
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
                    encoderModel: "opt/models/fastconformer_nemo_export/ctc_model.onnx";
                    vocabFile: "opt/models/fastconformer_nemo_export/tokens.txt";
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

### Python Environment Setup

The toolkit includes Python helper scripts for model export and utilities. First set up the Python environment:

```bash
# Setup Python virtual environment (one time)
./impl/setup_python_env.sh

# Activate environment for script usage
source ./activate_python.sh

# Install NeMo for model export (optional, large download)
pip install -r doc/requirements_nemo.txt
```

### Export NeMo Model
```bash
# Activate Python environment
source ./activate_python.sh

# Use provided export script
python impl/bin/export_model_ctc_patched.py

# Or other export scripts
python impl/bin/export_fastconformer_with_preprocessing.py
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

Following official Teracloud Streams Toolkit Development Guide standards:

```
com.teracloud.streamsx.stt/
├── info.xml                       # Toolkit metadata (recommended)
├── toolkit.xml                    # Toolkit index (auto-generated)
├── activate_python.sh             # Quick Python environment activation
├── etc/                          # Configuration files
│   └── requirements_nemo.txt     # Python dependencies
├── lib/                          # Support libraries
│   └── onnxruntime/              # ONNX Runtime (was deps/)
├── opt/                          # Optional components
│   ├── models/                   # Model files (was models/)
│   │   └── fastconformer_nemo_export/
│   └── test_data/                # Test audio files (was test_data/)
├── impl/                         # Implementation directory
│   ├── bin/                      # Implementation scripts
│   │   ├── *.py                  # Python helper scripts
│   │   └── *.sh                  # Shell scripts
│   ├── venv/                     # Python virtual environment
│   ├── include/                  # Header files
│   ├── src/                      # C++ source files
│   └── lib/                      # Built libraries
├── doc/                          # Documentation (was *.md files)
├── samples/                      # Example applications
│   └── com.teracloud.streamsx.stt.sample/
├── test/                         # Test framework
└── com.teracloud.streamsx.stt/   # SPL operators (toolkit namespace)
    ├── OnnxSTT/                  # Main STT operator
    └── FileAudioSource/          # Audio file reader
```

## Performance

Benchmarked on Intel Xeon CPU with 4 threads:

| Model | Audio Length | Processing Time | RTF | Memory |
|-------|-------------|-----------------|-----|---------|
| NeMo FastConformer CTC | 3s | 600ms | 0.20 | 500MB |
| NeMo FastConformer CTC | 8.7s | 1700ms | 0.19 | 500MB |

## Documentation

- [Architecture](doc/ARCHITECTURE.md)
- [Quick Start](doc/QUICK_START_NEXT_SESSION.md)
- [Operator Guide](doc/OPERATOR_GUIDE.md)
- [Implementation Guide](doc/IMPLEMENTATION.md)
- [NeMo Integration Workflow](doc/NEMO_WORKFLOW.md)
- [Testing Strategy](doc/TESTING.md)
- [Update Plan](doc/STREAMS_OPERATOR_UPDATE_PLAN.md)
- [Scripts Overview](doc/SCRIPTS_OVERVIEW.md)
- [Migration to Standard Structure](doc/REVISED_DIRECTORY_STRUCTURE.md)
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