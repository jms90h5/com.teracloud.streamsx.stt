# STT Toolkit Samples

This directory contains sample applications for the TeraCloud Streams Speech-to-Text toolkit.

## Current Status

**⚠️ IMPORTANT**: These samples may not currently compile due to known issues with ONNX Runtime header conflicts. See `../doc/ARCHITECTURE.md` for detailed information about current status and ongoing fixes.

## Available Samples

### Production Examples

#### BasicSTTExample
- **Purpose**: Clean, simple C++/ONNX demonstration
- **Location**: `BasicExample/BasicSTTExample.spl`
- **Technology**: NeMoSTT operator (simplified interface)
- **Requirements**: NeMo FastConformer model, test audio data

#### SimpleTest  
- **Purpose**: Minimal working C++/ONNX example template
- **Location**: `SimpleTest.spl`
- **Technology**: NeMoSTT operator (entry-level)
- **Requirements**: Basic NeMo model setup

#### WorkingNeMoRealtime
- **Purpose**: Production-ready real-time transcription  
- **Location**: `WorkingNeMoRealtime.spl`
- **Technology**: NeMoSTT operator with confidence scoring
- **Requirements**: FastConformer model, real-time audio processing

#### NeMoCTCSample
- **Purpose**: Advanced C++/ONNX configuration using OnnxSTT
- **Location**: `NeMoCTCSample.spl` 
- **Technology**: OnnxSTT operator (full configuration control)
- **Requirements**: NeMo CTC model, advanced setup

## Quick Start

1. **Check build environment**:
   ```bash
   make status
   ```

2. **Build samples** (may fail due to known issues):
   ```bash
   make all
   ```

3. **Get help**:
   ```bash
   make help
   ```

## Prerequisites

- TeraCloud Streams 7.2.0.1+
- Indexed STT toolkit
- NeMo FastConformer model (see toolkit documentation)
- Test audio data

## Troubleshooting

If compilation fails with ONNX Runtime header errors:
1. Check `../doc/ARCHITECTURE.md` for current status
2. See the "Known Issues and Troubleshooting" section
3. Consider using the working Python verification instead

## Directory Structure

```
samples/
├── BasicExample/           # Simple demonstration
│   └── BasicSTTExample.spl
├── SimpleTest.spl          # Minimal template
├── WorkingNeMoRealtime.spl # Real-time processing
├── NeMoCTCSample.spl       # Advanced OnnxSTT configuration
├── data/                   # Test data (when available)
├── output/                 # Build artifacts (created by make)
├── Makefile               # Build system
└── README.md              # This file
```

## Technology Overview

**All samples use pure C++/ONNX integration** via two main operators:
- **NeMoSTT**: Simplified interface for basic usage
- **OnnxSTT**: Advanced configuration with full control

**No Python integration samples are currently provided** - all processing happens in C++ for performance.

## Notes

- Samples are provided for reference and testing
- Current compilation issues are being actively addressed
- See main toolkit documentation for working verification methods
- Build artifacts are automatically cleaned with `make clean`