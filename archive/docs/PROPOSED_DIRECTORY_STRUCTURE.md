# Proposed Directory Reorganization

## Current Issues
- 50+ Python scripts in root directory
- Multiple binary/data files cluttering root
- Mixed documentation files
- No clear separation between core toolkit and utilities

## Proposed Structure

```
com.teracloud.streamsx.stt/
│
├── README.md                    # Main documentation (keep in root)
├── ARCHITECTURE.md              # System architecture (keep in root)
├── OPERATOR_GUIDE.md            # SPL operator guide (keep in root)
├── toolkit.xml                  # Toolkit descriptor (required in root)
├── info.xml                     # Toolkit metadata (required in root)
├── Makefile                     # Main build file (keep in root)
├── requirements.txt             # Python dependencies (keep in root)
├── .gitignore
│
├── com.teracloud.streamsx.stt/  # SPL operators (required location)
│   ├── FileAudioSource.spl
│   ├── NeMoSTT/
│   └── OnnxSTT/
│
├── impl/                        # C++ implementation (required location)
│   ├── Makefile
│   ├── include/
│   ├── src/
│   └── lib/
│
├── deps/                        # External dependencies
│   └── onnxruntime/
│
├── models/                      # Model files (keep as-is)
│   ├── fastconformer_ctc_export/
│   └── README.md
│
├── samples/                     # SPL sample applications
│   ├── README.md
│   ├── Makefile
│   └── *.spl files
│
├── test/                        # Test suite
│   ├── README.md
│   ├── *.cpp test files
│   └── verify_nemo_setup.sh
│
├── python/                      # NEW: Python utilities
│   ├── README.md
│   ├── export/                  # Model export scripts
│   │   ├── export_model_ctc_patched.py
│   │   ├── export_model_ctc.py
│   │   └── ... other export scripts
│   ├── download/                # Model download scripts
│   │   ├── download_nemo_cache_aware_conformer.py
│   │   ├── download_sherpa_onnx_model.sh
│   │   └── download_silero_vad.sh
│   ├── analysis/                # Feature analysis/debug scripts
│   │   ├── check_model_shapes.py
│   │   ├── extract_features.py
│   │   └── ... other analysis scripts
│   └── requirements/            # Python environment setup
│       ├── requirements.txt
│       └── requirements_nemo.txt
│
├── scripts/                     # NEW: Shell scripts
│   ├── setup_onnx_runtime.sh
│   ├── setup_kaldi_fbank.sh
│   └── get_transcription.sh
│
├── docs/                        # Existing documentation directory
│   ├── QUICK_START.md
│   ├── development/            # Development-specific docs
│   └── solutions/              # Solution guides
│
├── archive/                     # Historical/development artifacts
│   ├── dev/
│   ├── docs/
│   └── test_scripts/
│
└── .work/                       # NEW: Working directory (gitignored)
    ├── test_outputs/           # Test outputs
    ├── debug_files/            # Debug artifacts (.bin, .npy files)
    └── venv/                   # Virtual environments
```

## Cleanup Actions

### 1. Move Python Scripts to `python/`
```bash
# Export scripts
mv export_*.py python/export/

# Download scripts  
mv download_*.py python/download/

# Analysis/debug scripts
mv check_*.py extract_*.py validate_*.py python/analysis/
mv save_*.py create_*.py find_*.py python/analysis/
```

### 2. Move Shell Scripts to `scripts/`
```bash
mv setup_*.sh download_*.sh scripts/
mv get_transcription.sh scripts/
```

### 3. Clean Up Debug/Test Artifacts
```bash
# Move to working directory (gitignored)
mkdir -p .work/debug_files
mv *.bin *.npy .work/debug_files/

# Move Python virtual environments
mv venv_* .work/
```

### 4. Organize Documentation
```bash
# Move quick start to docs
mv QUICK_START_NEXT_SESSION.md docs/QUICK_START.md

# Move development-specific docs
mv CRITICAL_FINDINGS_*.md docs/development/
mv MAJOR_BREAKTHROUGH_*.md docs/development/

# Move solution guides
mv *_SOLUTIONS.md *_DEPENDENCIES.md docs/solutions/
```

### 5. Update .gitignore
```gitignore
# Working directory
.work/
*.bin
*.npy
python_debug_info.json

# Virtual environments
venv*/
.venv/

# Build outputs
output/
data/
*.sab
*.adl

# Generated files
cpp_features*
python_features*
*_transcript_*.txt
```

## Benefits

1. **Clear Separation**: Core C++ toolkit vs Python utilities
2. **Easier Navigation**: Related files grouped together
3. **Cleaner Root**: Only essential files in root directory
4. **Better Git History**: Debug artifacts in gitignored directory
5. **User-Friendly**: Clear where to find what

## Migration Script

```bash
#!/bin/bash
# migrate_structure.sh

# Create new directories
mkdir -p python/{export,download,analysis,requirements}
mkdir -p scripts
mkdir -p .work/{test_outputs,debug_files,venv}
mkdir -p docs/{development,solutions}

# Move Python scripts
echo "Moving Python scripts..."
mv export_*.py python/export/ 2>/dev/null || true
mv download_*.py python/download/ 2>/dev/null || true
mv check_*.py extract_*.py validate_*.py save_*.py create_*.py find_*.py python/analysis/ 2>/dev/null || true

# Move shell scripts
echo "Moving shell scripts..."
mv setup_*.sh download_*.sh get_transcription.sh scripts/ 2>/dev/null || true

# Move debug artifacts
echo "Moving debug artifacts..."
mv *.bin *.npy python_debug_info.json .work/debug_files/ 2>/dev/null || true

# Move virtual environments
echo "Moving virtual environments..."
mv venv_* .work/ 2>/dev/null || true

# Update documentation
echo "Organizing documentation..."
mv QUICK_START_NEXT_SESSION.md docs/QUICK_START.md 2>/dev/null || true
mv CRITICAL_FINDINGS_*.md MAJOR_BREAKTHROUGH_*.md docs/development/ 2>/dev/null || true
mv *_SOLUTIONS.md *_DEPENDENCIES.md *_UPGRADE_*.md docs/solutions/ 2>/dev/null || true

# Create README files
echo "Creating README files..."

# Python README
cat > python/README.md << 'EOF'
# Python Utilities

This directory contains Python scripts for model management and analysis.

## Subdirectories

- **export/** - Model export scripts (NeMo to ONNX conversion)
- **download/** - Model and resource download utilities
- **analysis/** - Feature extraction and debugging tools
- **requirements/** - Python environment setup files

## Quick Start

1. Set up Python environment:
   ```bash
   python3.11 -m venv ../.work/venv
   source ../.work/venv/bin/activate
   pip install -r requirements/requirements.txt
   ```

2. Export a model:
   ```bash
   python export/export_model_ctc_patched.py
   ```

See individual subdirectories for detailed documentation.
EOF

echo "Migration complete!"
```

## Questions for Consideration

1. Should we keep `models_backup_*` directories or move to archive?
2. Should generated SPL files (*.spl.broken) be removed or archived?
3. Should `output/` directories be gitignored entirely?
4. Do you want Python scripts further categorized by function?