# Test Directory

This directory contains test utilities and verification scripts for the TeraCloud Streams STT toolkit.

## Current Test Files

### **Active Test Programs**

#### `test_real_nemo_fixed.cpp`
- **Purpose**: Comprehensive standalone C++ test for NeMo CTC model
- **Features**: Command-line options, WAV file support, performance benchmarking
- **Model**: Uses current `models/fastconformer_ctc_export/model.onnx`
- **Status**: ✅ **Recommended** - Most complete test program

#### `test_nemo_ctc_simple.cpp`
- **Purpose**: Basic functionality test for NeMoCTCImpl class
- **Features**: Simple initialization and transcription test
- **Model**: Uses current `models/fastconformer_ctc_export/model.onnx`
- **Status**: ✅ **Useful** for basic testing

### **Verification Scripts**

#### `verify_nemo_setup.sh`
- **Purpose**: Comprehensive system verification script
- **Checks**: ONNX Runtime, C++ library, models, Python dependencies
- **Status**: ✅ **Updated** with current model paths
- **Usage**: `./verify_nemo_setup.sh`

## Building and Running Tests

### Prerequisites

1. **Build the toolkit first:**
   ```bash
   cd ../impl && make clean && make
   ```

2. **Export the NeMo model:**
   ```bash
   cd .. && python export_model_ctc_patched.py
   ```

3. **Verify setup:**
   ```bash
   cd test && ./verify_nemo_setup.sh
   ```

### Build Test Programs

#### Simple CTC Test
```bash
cd test
g++ -std=c++14 -O2 -I../impl/include -I../deps/onnxruntime/include \
    test_nemo_ctc_simple.cpp ../impl/lib/libs2t_impl.so \
    -L../deps/onnxruntime/lib -lonnxruntime -ldl \
    -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../deps/onnxruntime/lib' \
    -o test_nemo_ctc_simple

# Run
./test_nemo_ctc_simple
```

#### Comprehensive Fixed Test
```bash
cd test
g++ -std=c++14 -O2 -I../impl/include -I../deps/onnxruntime/include \
    test_real_nemo_fixed.cpp ../impl/lib/libs2t_impl.so \
    -L../deps/onnxruntime/lib -lonnxruntime -ldl \
    -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../deps/onnxruntime/lib' \
    -o test_real_nemo_fixed

# Run with options
./test_real_nemo_fixed --help
./test_real_nemo_fixed --verbose
```

### Library Path Requirements

**Important**: Always set library paths when running tests:
```bash
export LD_LIBRARY_PATH=$PWD/../deps/onnxruntime/lib:$PWD/../impl/lib:$LD_LIBRARY_PATH
```

Or use the rpath settings shown in the build commands above.

## Model Requirements

### Current Working Model Structure
```
models/fastconformer_ctc_export/
├── model.onnx          # Main CTC model (44MB)
└── tokens.txt          # Vocabulary file (1023 tokens + blank)
```

### Model Export
If models are missing, export them using:
```bash
cd .. && python export_model_ctc_patched.py
```

This script handles the NeMo/huggingface_hub dependency conflicts automatically.

## Test Data

Tests can use:
- **Audio files**: Place in `../test_data/audio/`
- **Raw audio**: 16kHz, 16-bit, mono format
- **WAV files**: Automatically converted by test programs

Example test file: `../test_data/audio/librispeech-1995-1837-0001.wav`

## Expected Output

### Successful Test Run
```
=== Simple NeMo CTC Test ===
Initializing model...

Model info:
Model loaded successfully
Input shape: [1, 80, -1]
Output shape: [1, -1, 1025]
Vocabulary size: 1025

Transcribing 2 seconds of silence...
Result: ''
```

### Common Issues

1. **"Failed to initialize model"**
   - Check model path: `../models/fastconformer_ctc_export/model.onnx`
   - Ensure ONNX Runtime is installed: `../setup_onnx_runtime.sh`

2. **"libonnxruntime.so: cannot open shared object file"**
   - Set LD_LIBRARY_PATH or use rpath in build command
   - Check: `ls ../deps/onnxruntime/lib/libonnxruntime.so`

3. **"Vocabulary file missing"**
   - Ensure tokens.txt exists: `../models/fastconformer_ctc_export/tokens.txt`
   - Re-export model if needed

## Archived Files

Obsolete test files have been moved to `../archive/test_scripts/`:
- `test_real_nemo.cpp` - Used old model paths
- `test_nemo_improvements.cpp` - Missing dependencies
- `test_nemo_standalone.cpp` - Superseded by test_real_nemo_fixed.cpp
- `run_nemo_test.sh` - Used old paths

These are preserved for historical reference but should not be used.

## Development Notes

- Tests are designed to work with the current CTC model export workflow
- All paths are relative to the test directory
- Tests focus on C++ API validation, not SPL integration
- For SPL testing, use the samples in `../samples/`

## Quick Verification

Run this command to verify everything is working:
```bash
cd test && ./verify_nemo_setup.sh && echo "✅ Ready for testing"
```

This will check all dependencies and provide specific guidance for any missing components.