# SPL Applications Guide

This guide covers building and running the SPL sample applications in the STT toolkit.

## Prerequisites

1. Complete the toolkit setup using either `./setup.sh` or `./setup_external_venv.sh`
2. Export the NeMo model (see main README.md)
3. Ensure Streams environment is sourced

## Available SPL Samples

All samples are located in the `samples/` directory:

1. **SimpleTest** - Basic test of NeMo STT functionality
2. **BasicSTTExample** - Example with file output
3. **NeMoCTCSample** - Demonstrates CTC model processing

All samples process the same test audio (`audio/librispeech_3sec.wav`) and should produce:
```
"it was the first great"
```

## Building SPL Applications

Navigate to the samples directory:
```bash
cd samples
```

Build an application using the Streams compiler:
```bash
# Build SimpleTest
sc -M SimpleTest -t ../

# Build BasicSTTExample  
sc -M BasicSTTExample -t ../

# Build NeMoCTCSample
sc -M NeMoCTCSample -t ../
```

The `-t ../` flag tells the compiler to use the parent directory as a toolkit path.

## Running SPL Applications

### Standalone Mode

Run the compiled application in standalone mode:
```bash
./output/bin/standalone -d .
```

**Important**: The `-d .` flag sets the data directory to the current directory, which is required for the FileAudioSource operator to find the audio files.

### Expected Output

For SimpleTest:
```
SimpleTest Transcription: it was the first great
```

For BasicSTTExample:
```
Transcription: it was the first great
```
Also creates `transcription_output.txt` with: `{transcription="it was the first great"}`

For NeMoCTCSample:
```
=== NeMo CTC Transcription ===
Text: it was the first great
Expected: 'it was the first great'
=============================
```

## Technical Notes

### Library Naming
The SPL operators expect a library named `libnemo_ctc_impl.so`, but the makefile creates `libs2t_impl.so`. The setup scripts automatically create a symlink to resolve this:
```bash
cd impl/lib
ln -sf libs2t_impl.so libnemo_ctc_impl.so
```

### Tensor Dimension Fix
The NeMo model expects input tensors in [batch, features, time] format, but the original audio processing created [batch, time, features]. This has been fixed in `impl/include/NeMoCTCImpl.cpp` with a transpose operation.

### Model Path Configuration
The samples use environment variables for model paths with fallbacks:
- `STREAMS_STT_MODEL_PATH` - Path to ONNX model file
- `STREAMS_STT_TOKENS_PATH` - Path to vocabulary tokens file

You can set these before running, or the applications will use default paths.

## Troubleshooting

### "No data directory specified" Error
Always use the `-d .` flag when running standalone applications.

### Model Not Found
Ensure you've exported the model using:
```bash
source ./activate_python.sh
python impl/bin/export_model_ctc_patched.py
```

### Wrong Transcription Output
If you get different output than "it was the first great", ensure:
1. The C++ library has been rebuilt with the tensor transpose fix
2. The symlink `libnemo_ctc_impl.so` exists and points to the updated library
3. Rebuild the SPL application after any C++ changes

### Compilation Errors
If you get "Lorem" related errors during SPL compilation:
1. Ensure Python venv is outside the toolkit directory (use `setup_external_venv.sh`)
2. Or delete files with spaces: `find . -name "* *" -type f -delete`

## Integration with Streams Jobs

To submit as a Streams job instead of running standalone:
```bash
streamtool submitjob output/SimpleTest.sab
```

Note: Job submission requires proper model path configuration in the Streams instance.