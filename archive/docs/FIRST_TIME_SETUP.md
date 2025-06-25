# First-Time Setup Guide

## For Users Cloning the Repository

### Automatic Setup (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd com.teracloud.streamsx.stt
   ```

2. **Run automatic setup:**
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - ✅ Check system dependencies (Python 3.11, g++, Streams)
   - ✅ Create Python virtual environment at `impl/venv/`
   - ✅ Install essential packages (onnx, onnxruntime, numpy, scipy)
   - ✅ Build C++ implementation library
   - ✅ Create convenience scripts and aliases
   - ✅ Run verification tests

3. **Activate development environment:**
   ```bash
   source .envrc
   ```
   
   This gives you convenient aliases:
   - `stt-build` - Build C++ implementation
   - `stt-test` - Run verification tests
   - `stt-python` - Activate Python environment
   - `stt-samples` - Check sample status

### What the Setup Script Does

#### System Requirements Check:
- Python 3.11 with venv support
- C++ compiler (g++)
- Teracloud Streams environment (warns if not set)

#### Python Environment:
- Creates `impl/venv/` with Python 3.11
- Installs essential packages for model scripts
- NeMo toolkit installation is optional (large download)

#### C++ Build:
- Compiles the STT implementation library
- Links with ONNX Runtime at `lib/onnxruntime/`
- Creates `impl/lib/libs2t_impl.so`

#### Convenience Setup:
- Creates `.envrc` with development aliases
- Sets up library paths for testing
- Provides quick commands for common tasks

### Model Export (Separate Step)

After initial setup, you'll need to export a model:

```bash
# Activate environment
source .envrc

# Install NeMo (large download, ~2-3GB)
pip install -r doc/requirements_nemo.txt

# Export the working CTC model (confirmed working script)
python impl/bin/export_model_ctc_patched.py

# This creates: opt/models/fastconformer_ctc_export/model.onnx
```

### Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# 1. Python environment
./impl/setup_python_env.sh

# 2. Build C++ implementation  
cd impl && make && cd ..

# 3. Verify setup
./test/verify_nemo_setup.sh
```

### Troubleshooting

#### "Python 3.11 not found"
```bash
# Rocky Linux / RHEL
sudo dnf install python3.11 python3.11-venv python3.11-pip

# Ubuntu
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### "g++ not found"
```bash
# Rocky Linux / RHEL
sudo dnf groupinstall 'Development Tools'

# Ubuntu
sudo apt install build-essential
```

#### "STREAMS_INSTALL not set"
```bash
# Source Streams environment first
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh

# Then run setup
./setup.sh
```

#### Build failures
- Check that ONNX Runtime is present: `ls lib/onnxruntime/lib/`
- Verify paths in Makefiles match new structure
- Run verification: `./test/verify_nemo_setup.sh`

### What You Get

After successful setup:

**Development Environment:**
- Python 3.11 venv with essential packages
- C++ STT implementation library
- ONNX Runtime integration
- Convenient development aliases

**Ready to Use:**
- Model export scripts: `impl/bin/export_model_ctc_patched.py` (working)
- Sample applications: `samples/` directory
- Test framework: `test/` directory
- Documentation: `doc/` directory

**Next Steps:**
1. Export a model using the working script
2. Build and test sample applications
3. Integrate with your Streams applications

The automatic setup handles all the complexity of virtual environments, library paths, and dependencies that users previously had to figure out manually.