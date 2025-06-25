#!/bin/bash

# Setup script for kaldi-native-fbank
# This provides high-quality feature extraction for speech recognition

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPS_DIR="${SCRIPT_DIR}/deps"
KALDI_FBANK_DIR="${DEPS_DIR}/kaldi-native-fbank"

echo "Setting up kaldi-native-fbank for feature extraction..."

# Create deps directory
mkdir -p "${DEPS_DIR}"

# Option 1: Download pre-built release (faster)
download_prebuilt() {
    echo "Downloading pre-built kaldi-native-fbank..."
    cd "${DEPS_DIR}"
    
    # Download latest release
    KALDI_VERSION="1.19.2"
    DOWNLOAD_URL="https://github.com/csukuangfj/kaldi-native-fbank/releases/download/v${KALDI_VERSION}/kaldi-native-fbank-${KALDI_VERSION}-linux-x64-shared.tar.bz2"
    
    wget -q --show-progress "${DOWNLOAD_URL}" -O kaldi-native-fbank.tar.bz2
    
    # Extract
    echo "Extracting kaldi-native-fbank..."
    tar -xjf kaldi-native-fbank.tar.bz2
    mv kaldi-native-fbank-${KALDI_VERSION}-linux-x64-shared kaldi-native-fbank
    rm kaldi-native-fbank.tar.bz2
    
    echo "Pre-built kaldi-native-fbank installed successfully!"
}

# Option 2: Build from source (more flexible)
build_from_source() {
    echo "Building kaldi-native-fbank from source..."
    cd "${DEPS_DIR}"
    
    # Clone repository
    if [ ! -d "kaldi-native-fbank-src" ]; then
        git clone https://github.com/csukuangfj/kaldi-native-fbank.git kaldi-native-fbank-src
    fi
    
    cd kaldi-native-fbank-src
    
    # Build
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="${KALDI_FBANK_DIR}" \
          -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF \
          -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
          ..
    
    make -j$(nproc)
    make install
    
    echo "kaldi-native-fbank built and installed successfully!"
}

# Check if already installed
if [ -d "${KALDI_FBANK_DIR}" ]; then
    echo "kaldi-native-fbank already exists in ${KALDI_FBANK_DIR}"
    echo "Remove it first if you want to reinstall."
    exit 0
fi

# Use pre-built by default (faster)
if [ "$1" == "--build-from-source" ]; then
    build_from_source
else
    download_prebuilt
fi

# Create a simple test program to verify installation
cat > "${DEPS_DIR}/test_kaldi_fbank.cpp" << 'EOF'
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include <iostream>
#include <vector>

int main() {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = 16000;
    opts.mel_opts.num_bins = 80;
    
    knf::OnlineFbank fbank(opts);
    
    // Test with dummy audio
    std::vector<float> audio(16000, 0.0f);  // 1 second of silence
    fbank.AcceptWaveform(16000, audio.data(), audio.size());
    
    int num_frames_ready = fbank.NumFramesReady();
    std::cout << "Fbank test successful! Frames ready: " << num_frames_ready << std::endl;
    
    return 0;
}
EOF

echo ""
echo "Setup complete! kaldi-native-fbank is installed in: ${KALDI_FBANK_DIR}"
echo ""
echo "To use in your build:"
echo "  export KALDI_NATIVE_FBANK_DIR=${KALDI_FBANK_DIR}"
echo ""
echo "Include directories:"
echo "  ${KALDI_FBANK_DIR}/include"
echo ""
echo "Library directories:"
echo "  ${KALDI_FBANK_DIR}/lib"
echo ""