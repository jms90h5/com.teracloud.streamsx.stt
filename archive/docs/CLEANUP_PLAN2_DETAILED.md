# Cleanup Plan Phase 2 - Detailed Implementation Guide

**Date**: June 24, 2025  
**Status**: Ready for implementation  
**Principle**: Conservative approach - preserve user functionality, archive development artifacts  
**Updated from**: Original cleanup plan with lessons learned from dependency resolution work

## Overview

This plan provides detailed steps for cleaning up the STT toolkit while preserving all user-facing functionality and maintaining development history through archival. The approach is more conservative than the original plan, keeping anything potentially useful for toolkit users.

## Key Changes from Original Plan

1. **Archive instead of delete** - Preserve development history
2. **Keep more user utilities** - Scripts that help with setup and testing
3. **Preserve critical documentation** - Including recent breakthrough findings
4. **Conservative file assessment** - When in doubt, keep the file

## Phase 1: Directory Structure Preparation

### Create Archive Directories
```bash
cd /home/streamsadmin/workspace/teracloud/toolkits/com.teracloud.streamsx.stt
mkdir -p archive/dev_scripts
mkdir -p archive/docs/historical
mkdir -p archive/test_outputs
mkdir -p archive/build_artifacts
```

## Phase 2: Archive Development Scripts

### Python Development Scripts to Archive
Move these to `archive/dev_scripts/` with git mv to preserve history:

**Rationale**: These are one-time debugging, exploration, or superseded scripts not needed by toolkit users

```bash
# Model export attempts (superseded by export_model_ctc_patched.py)
git mv export_model_ctc.py archive/dev_scripts/                    # Original with strict checks
git mv export_model_ctc_py39.py archive/dev_scripts/               # Python 3.9 attempt
git mv export_fastconformer_simple.py archive/dev_scripts/         # Early export exploration
git mv export_nemo_cache_aware_onnx.py archive/dev_scripts/        # Cache-aware attempt
git mv convert_nemo_to_onnx_minimal.py archive/dev_scripts/        # Minimal conversion
git mv convert_extracted_to_onnx.py archive/dev_scripts/           # One-time conversion

# Debugging and analysis tools
git mv check_onnx_model.py archive/dev_scripts/                    # ONNX inspection
git mv decode_transcript.py archive/dev_scripts/                   # Transcript decoding
git mv final_nemo_direct_inference.py archive/dev_scripts/         # Direct inference test
git mv extract_nemo_model.py archive/dev_scripts/                  # Model extraction

# One-time fixes and explorations
git mv comprehensive_nemo_fix.py archive/dev_scripts/              # Comprehensive fix attempt
git mv fix_all_union_syntax.py archive/dev_scripts/                # Python 3.10 syntax fixes
git mv fix_nemo_python39.py archive/dev_scripts/                   # Python 3.9 compatibility
git mv generate_real_transcript.py archive/dev_scripts/            # Test data generation
```

### Scripts to KEEP in Root (User-Facing Utilities)
**Rationale**: These help users set up, export, and test the toolkit

```bash
# KEEP THESE - User utilities:
export_model_ctc_patched.py                # WORKING export solution with patches
download_nemo_cache_aware_conformer.py     # Downloads required NeMo model  
download_nemo_simple.py                    # Alternative download method
test_nemo_simple.py                        # Simple test for exported models
wav_to_raw.py                              # Audio format conversion utility
setup_onnx_runtime.sh                      # ONNX setup for users
download_sherpa_onnx_model.sh              # Model download utility
```

### Create Archive Scripts README
```bash
cat > archive/dev_scripts/README.md << 'EOF'
# Development Scripts Archive

This directory contains scripts used during the development and debugging of the STT toolkit.
These are preserved for historical reference but are not needed for normal toolkit usage.

## Script Categories

### Model Export Evolution
- `export_model_ctc.py` - Original export with strict version checks (superseded)
- `export_model_ctc_py39.py` - Python 3.9 compatibility attempt (superseded by 3.11)
- `export_fastconformer_simple.py` - Early export exploration
- `export_nemo_cache_aware_onnx.py` - Cache-aware export attempt
- `convert_nemo_to_onnx_minimal.py` - Minimal conversion approach
- `convert_extracted_to_onnx.py` - One-time extraction-based conversion

### Debugging and Analysis Tools
- `check_onnx_model.py` - ONNX model inspection utility
- `decode_transcript.py` - Transcript decoding and analysis
- `final_nemo_direct_inference.py` - Direct inference testing
- `extract_nemo_model.py` - NeMo model extraction utility
- `generate_real_transcript.py` - Test data generation

### Problem Resolution
- `comprehensive_nemo_fix.py` - Comprehensive dependency fix attempt
- `fix_all_union_syntax.py` - Python 3.10 union syntax fixes
- `fix_nemo_python39.py` - Python 3.9 compatibility fixes

## Current Working Solutions (in root directory)
- `/export_model_ctc_patched.py` - Working export with dynamic dependency patches
- `/NEMO_EXPORT_SOLUTIONS.md` - Complete export guide and solutions
- `/download_nemo_cache_aware_conformer.py` - Model download utility

## Development Timeline
These scripts represent the evolution of solutions from initial exploration to working implementations.
The working solutions have been promoted to the root directory for user access.
EOF
```

## Phase 3: Archive Historical Documentation

### Documentation Files to Archive
Move these to `archive/docs/historical/` to preserve development history:

**Rationale**: These are progress tracking, status updates, and planning documents that served their purpose but aren't needed by users

```bash
# Planning and integration documents
git mv NEMO_INTEGRATION_PLAN.md archive/docs/historical/           # Initial integration planning
git mv NEMO_INTEGRATION_STATUS.md archive/docs/historical/         # Integration progress tracking
git mv NEMO_WORK_PLAN.md archive/docs/historical/                  # Detailed work planning
git mv NEMO_RESTORATION_PLAN.md archive/docs/historical/           # Historical recovery plan

# Implementation tracking
git mv NEMO_CACHE_AWARE_IMPLEMENTATION.md archive/docs/historical/ # Implementation details
git mv IMPLEMENTATION_COMPLETE.md archive/docs/historical/         # Completion status
git mv IMPLEMENTATION_SUMMARY.md archive/docs/historical/          # Implementation summary
git mv OPERATOR_UPDATES_COMPLETE.md archive/docs/historical/       # Operator update status

# Status updates and progress tracking
git mv STATUS_UPDATE_2025_06_23_FINAL.md archive/docs/historical/  # Final status update
git mv WORK_PROGRESS_TRACKING.md archive/docs/historical/          # Detailed progress tracking
git mv WORKING_SAMPLE_PATTERN_FINAL.md archive/docs/historical/    # Development pattern notes
git mv HONEST_CURRENT_STATE_JUNE_17_2025.md archive/docs/historical/ # Historical status

# Problem resolution documentation
git mv FIX_NEMO_MODEL.md archive/docs/historical/                  # Historical fix approaches
git mv FIX_SUMMARY_2025_06_23.md archive/docs/historical/          # Fix summary
git mv FEATURE_EXTRACTION_ROOT_CAUSE.md archive/docs/historical/   # Debugging analysis
git mv FASTCONFORMER_TEST_GUIDE.md archive/docs/historical/        # Historical test guide

# Discovery and breakthrough documentation
git mv MAJOR_BREAKTHROUGH_SUMMARY.md archive/docs/historical/      # Key discoveries
git mv REAL_MODEL_SUCCESS.md archive/docs/historical/              # Success documentation
git mv SOLUTION_SUMMARY.md archive/docs/historical/                # Solution summary (superseded)

# Sample and testing plans
git mv STREAMS_SAMPLES_FIX_PLAN.md archive/docs/historical/        # Sample fixing plan
git mv STREAMS_SAMPLE_RESULTS.md archive/docs/historical/          # Test results
git mv STREAMS_SAMPLE_EXECUTION_PLAN.md archive/docs/historical/   # Execution planning
git mv SYSTEMATIC_TESTING_PLAN.md archive/docs/historical/         # Testing strategy
git mv RESTORE_WORKING_SAMPLES_PLAN.md archive/docs/historical/    # Restoration planning

# Cleanup and maintenance
git mv FAKE_DATA_REMOVAL_SESSION.md archive/docs/historical/       # Development cleanup session
git mv CURRENT_STATE_TRUTH.md archive/docs/historical/             # Historical state assessment
git mv CLEANUP_PLAN.md archive/docs/historical/                    # Original cleanup plan
```

### Documentation to KEEP in Root (User-Facing)
**Rationale**: These provide essential information for toolkit users

```bash
# KEEP THESE - Essential user documentation:
README.md                           # Main documentation and entry point
ARCHITECTURE.md                     # System design reference
OPERATOR_GUIDE.md                   # SPL operator usage guide
QUICK_START_NEXT_SESSION.md         # Quick start guide
README_MODELS.md                    # Model information and requirements

# KEEP THESE - Critical technical guides:
NEMO_EXPORT_SOLUTIONS.md            # Comprehensive export solution guide
NEMO_PYTHON_DEPENDENCIES.md         # Dependency conflict explanation
PYTHON_UPGRADE_2025_06_24.md        # Python 3.11 upgrade instructions
CRITICAL_FINDINGS_2025_06_23.md     # Working model identification
Claude.md                           # Development guidelines
CLEANUP_PLAN_UPDATED.md             # This cleanup plan
```

### Create Historical Documentation Index
```bash
cat > archive/docs/historical/INDEX.md << 'EOF'
# Historical Documentation Index

This directory contains development documentation created during the STT toolkit implementation.
These documents trace the evolution of the project and problem-solving process from initial planning through final implementation.

## Document Categories

### Planning and Strategy Documents
- `NEMO_INTEGRATION_PLAN.md` - Initial integration planning and approach
- `NEMO_WORK_PLAN.md` - Detailed work breakdown and timeline
- `STREAMS_SAMPLE_EXECUTION_PLAN.md` - Sample implementation strategy
- `SYSTEMATIC_TESTING_PLAN.md` - Comprehensive testing approach

### Progress Tracking and Status Updates
- `NEMO_INTEGRATION_STATUS.md` - Ongoing integration progress
- `STATUS_UPDATE_2025_06_23_FINAL.md` - Final status report
- `WORK_PROGRESS_TRACKING.md` - Detailed progress tracking
- `HONEST_CURRENT_STATE_JUNE_17_2025.md` - Mid-project status assessment

### Implementation Documentation
- `NEMO_CACHE_AWARE_IMPLEMENTATION.md` - Cache-aware streaming implementation
- `IMPLEMENTATION_COMPLETE.md` - Implementation completion report
- `IMPLEMENTATION_SUMMARY.md` - High-level implementation overview
- `OPERATOR_UPDATES_COMPLETE.md` - SPL operator update completion

### Problem Resolution and Debugging
- `FIX_NEMO_MODEL.md` - Model fixing approaches and attempts
- `FEATURE_EXTRACTION_ROOT_CAUSE.md` - Feature extraction debugging analysis
- `FIX_SUMMARY_2025_06_23.md` - Summary of fixes implemented
- `FASTCONFORMER_TEST_GUIDE.md` - Testing procedures and debugging

### Discovery and Breakthrough Documentation
- `MAJOR_BREAKTHROUGH_SUMMARY.md` - Key technical discoveries
- `REAL_MODEL_SUCCESS.md` - Successful model implementation
- `SOLUTION_SUMMARY.md` - Solution documentation (superseded by current guides)
- `WORKING_SAMPLE_PATTERN_FINAL.md` - Working implementation patterns

### Sample and Testing Documentation
- `STREAMS_SAMPLES_FIX_PLAN.md` - Sample application fixing strategy
- `STREAMS_SAMPLE_RESULTS.md` - Sample execution results
- `RESTORE_WORKING_SAMPLES_PLAN.md` - Sample restoration procedures

### Maintenance and Cleanup
- `FAKE_DATA_REMOVAL_SESSION.md` - Development artifact cleanup
- `CURRENT_STATE_TRUTH.md` - Project state assessment
- `CLEANUP_PLAN.md` - Original cleanup strategy

## Development Timeline Context

These documents span from June 17-24, 2025, covering:
1. Initial NeMo integration challenges
2. Dependency conflict discovery and resolution
3. Model export breakthrough with dynamic patching
4. Final implementation and documentation

## Current User Documentation

For current toolkit usage, see root directory:
- `/README.md` - Main documentation
- `/NEMO_EXPORT_SOLUTIONS.md` - Export solutions and dependency fixes
- `/ARCHITECTURE.md` - System architecture
- `/OPERATOR_GUIDE.md` - SPL operator reference

## Development Lessons Learned

Key insights preserved in these documents:
1. Huggingface/NeMo circular dependency resolution
2. Dynamic patching approach for compatibility
3. Python 3.11 requirement for modern syntax
4. CTC vs RNNT export complexity differences
5. Importance of conservative cleanup strategies
EOF
```

## Phase 4: Clean Build Artifacts and Generated Files

### Remove Compiled Files
```bash
# Remove all object files
find . -name "*.o" -type f -delete

# Archive compiled test executables before removal
mkdir -p archive/build_artifacts
mv test/test_real_nemo archive/build_artifacts/ 2>/dev/null || true
mv test/test_nemo_improvements archive/build_artifacts/ 2>/dev/null || true
mv test/test_nemo_standalone archive/build_artifacts/ 2>/dev/null || true

# Clean implementation directory
cd impl && make clean && cd ..

# Clean sample builds
cd samples && make clean && cd ..
```

### Remove Generated Runtime Files
```bash
# Move transcription results to archive
mv transcription_results archive/test_outputs/ 2>/dev/null || true

# Remove other generated files
rm -f ibm_culture_transcript.txt
rm -f get_transcription.sh

# Remove virtual environments (large, regeneratable)
rm -rf venv_nemo/
rm -rf venv_nemo_py311/

# Create note about removed venvs
cat > PYTHON_ENVIRONMENT_SETUP.md << 'EOF'
# Python Environment Setup

Virtual environments (venv_nemo, venv_nemo_py311) have been removed to save space.
To recreate the working Python 3.11 environment:

```bash
python3.11 -m venv venv_nemo_py311
source venv_nemo_py311/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

For detailed setup instructions including dependency resolution, see:
- NEMO_EXPORT_SOLUTIONS.md
- PYTHON_UPGRADE_2025_06_24.md
- NEMO_PYTHON_DEPENDENCIES.md
EOF
```

## Phase 5: Update .gitignore

### Add Comprehensive Exclusions
```bash
cat >> .gitignore << 'EOF'

# Virtual environments
venv*/
.venv/
env/
ENV/
.env

# Build artifacts
*.o
*.so
*.a
*.dylib
*.pyc
*.pyo
__pycache__/
*.egg-info/

# Generated outputs
transcription_results/
*.transcript.txt
*.transcript.json
test_output/
output/

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~
.sublime-*

# OS files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
*.log
*.backup
*.bak

# Model downloads and caches (large files)
models/*/download/
models/*/cache/
models/*/.cache/
*.nemo
*.onnx.large

# Python cache and compiled
*.py[cod]
*$py.class
*.so
.Python
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Test outputs
test_results/
coverage/
.coverage
.pytest_cache/
.tox/

# Documentation builds
docs/_build/
site/
EOF
```

## Phase 6: Final Organization and Structure

### Root Directory Structure After Cleanup
```
com.teracloud.streamsx.stt/
├── README.md                        # Main documentation
├── ARCHITECTURE.md                  # System design
├── OPERATOR_GUIDE.md                # SPL operator reference
├── QUICK_START_NEXT_SESSION.md      # Quick start guide
├── README_MODELS.md                 # Model information
├── NEMO_EXPORT_SOLUTIONS.md         # Export solutions (comprehensive)
├── NEMO_PYTHON_DEPENDENCIES.md      # Dependency issues and solutions
├── PYTHON_UPGRADE_2025_06_24.md     # Python 3.11 upgrade guide
├── CRITICAL_FINDINGS_2025_06_23.md  # Working model identification
├── Claude.md                        # Development guidelines
├── CLEANUP_PLAN_UPDATED.md          # Updated cleanup plan
├── PYTHON_ENVIRONMENT_SETUP.md      # Environment setup notes
├── toolkit.xml                      # Toolkit descriptor
├── info.xml                         # Toolkit metadata
├── Makefile                         # Build configuration
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git exclusions
│
├── User Scripts (in root):
│   ├── export_model_ctc_patched.py          # Working export solution
│   ├── download_nemo_cache_aware_conformer.py # Model download
│   ├── download_nemo_simple.py              # Alternative download
│   ├── test_nemo_simple.py                  # Simple testing
│   ├── wav_to_raw.py                        # Audio conversion
│   ├── setup_onnx_runtime.sh                # ONNX setup
│   └── download_sherpa_onnx_model.sh        # Model download utility
│
├── Core Directories:
│   ├── com.teracloud.streamsx.stt/  # SPL operators
│   ├── impl/                        # C++ implementation
│   ├── deps/                        # Dependencies
│   ├── models/                      # Model files
│   ├── samples/                     # Sample applications
│   ├── test/                        # Test suite
│   ├── test_data/                   # Test audio files
│   └── docs/                        # Additional documentation
│
└── Archive Structure:
    └── archive/
        ├── dev_scripts/              # Development scripts
        │   ├── README.md             # Script documentation
        │   └── [development_scripts] # Archived scripts
        ├── docs/
        │   └── historical/           # Historical documentation
        │       ├── INDEX.md          # Documentation index
        │       └── [historical_docs] # Progress and planning docs
        ├── test_outputs/             # Archived test results
        └── build_artifacts/          # Archived compiled files
```

## Phase 7: Verification and Testing

### Complete Verification Checklist

After completing all phases:

- [ ] **Build Verification**:
  ```bash
  cd impl && make clean && make
  ```

- [ ] **Toolkit Generation**:
  ```bash
  spl-make-toolkit -i . --no-mixed-mode -m
  ```

- [ ] **Sample Compilation**:
  ```bash
  cd samples && make all
  ```

- [ ] **Export Functionality**:
  ```bash
  python export_model_ctc_patched.py
  ```

- [ ] **Model Download**:
  ```bash
  python download_nemo_cache_aware_conformer.py
  ```

- [ ] **Documentation Integrity**:
  - [ ] All README links work
  - [ ] Quick start guide is accurate
  - [ ] Architecture documentation reflects current state

- [ ] **Archive Accessibility**:
  - [ ] Archive directories browsable
  - [ ] Archive README files informative
  - [ ] Historical progression traceable

- [ ] **Git Repository Health**:
  - [ ] No untracked files that should be committed
  - [ ] .gitignore excludes appropriate files
  - [ ] Repository size reasonable

## Phase 8: Commit Strategy

### Phased Commits for Clean History

1. **Commit 1**: Archive development scripts
   ```bash
   git add archive/dev_scripts/
   git commit -m "Archive development scripts with preservation of history

   Move debugging and exploration scripts to archive/dev_scripts/:
   - Model export attempts superseded by export_model_ctc_patched.py
   - Debugging utilities not needed for normal toolkit use
   - One-time fixes and compatibility attempts
   
   All scripts preserved with git mv to maintain development history."
   ```

2. **Commit 2**: Archive historical documentation
   ```bash
   git add archive/docs/
   git commit -m "Archive historical development documentation

   Move progress tracking and planning docs to archive/docs/historical/:
   - Status updates and progress tracking
   - Problem resolution documentation
   - Implementation planning and strategy
   
   Current user documentation remains in root directory."
   ```

3. **Commit 3**: Clean build artifacts and update .gitignore
   ```bash
   git add .gitignore PYTHON_ENVIRONMENT_SETUP.md
   git commit -m "Clean build artifacts and improve .gitignore

   - Remove compiled binaries and object files
   - Archive transcription results
   - Remove large virtual environments (recreatable)
   - Update .gitignore to prevent future artifacts"
   ```

## Success Criteria

### Quantitative Measures
- File count reduced while maintaining functionality
- Repository size optimized
- Build times maintained or improved
- All tests continue to pass

### Qualitative Measures
- Clear separation between user-facing and development artifacts
- Intuitive file organization for new users
- Preserved development history for maintainers
- Comprehensive documentation for all use cases

This plan provides a thorough, conservative approach to cleaning up the toolkit while preserving all valuable materials and maintaining full functionality for users.