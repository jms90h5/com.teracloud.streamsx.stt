#!/bin/bash
# Setup ONNX Runtime locally for the WeNet toolkit

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEPS_DIR="${SCRIPT_DIR}/deps"
ONNX_VERSION="1.16.3"

echo "Setting up ONNX Runtime v${ONNX_VERSION} locally..."

# Create dependencies directory
mkdir -p "${DEPS_DIR}"
cd "${DEPS_DIR}"

# Check if already installed
if [ -f "onnxruntime/lib/libonnxruntime.so" ]; then
    echo "ONNX Runtime already installed in ${DEPS_DIR}/onnxruntime"
    exit 0
fi

# Download ONNX Runtime for Linux x64
echo "Downloading ONNX Runtime..."
wget -q --show-progress "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"

# Extract
echo "Extracting..."
tar -xzf "onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
mv "onnxruntime-linux-x64-${ONNX_VERSION}" onnxruntime
rm "onnxruntime-linux-x64-${ONNX_VERSION}.tgz"

# Create environment setup script
cat > "${SCRIPT_DIR}/setup_env.sh" << 'EOF'
#!/bin/bash
# Source this file to set up ONNX Runtime environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ONNX_ROOT="${SCRIPT_DIR}/deps/onnxruntime"

export LD_LIBRARY_PATH="${ONNX_ROOT}/lib:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ONNX_ROOT}:${CMAKE_PREFIX_PATH}"
export ONNXRUNTIME_ROOT_PATH="${ONNX_ROOT}"

echo "ONNX Runtime environment configured:"
echo "  ONNXRUNTIME_ROOT_PATH=${ONNXRUNTIME_ROOT_PATH}"
echo "  LD_LIBRARY_PATH includes ${ONNX_ROOT}/lib"
EOF

chmod +x "${SCRIPT_DIR}/setup_env.sh"

# Test installation
echo ""
echo "Testing ONNX Runtime installation..."
cat > test_onnx.cpp << 'EOF'
#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "ONNX Runtime initialized successfully!" << std::endl;
        std::cout << "API Version: " << ORT_API_VERSION << std::endl;
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
EOF

# Try to compile test
echo "Compiling test program..."
if g++ -std=c++14 test_onnx.cpp \
    -I"${DEPS_DIR}/onnxruntime/include" \
    -I"${DEPS_DIR}/onnxruntime/include/onnxruntime/core/session" \
    -L"${DEPS_DIR}/onnxruntime/lib" \
    -lonnxruntime \
    -Wl,-rpath,"${DEPS_DIR}/onnxruntime/lib" \
    -o test_onnx 2>/dev/null; then
    
    echo "Running test..."
    if ./test_onnx; then
        echo ""
        echo "✓ ONNX Runtime successfully installed!"
        echo ""
        echo "To use ONNX Runtime, run:"
        echo "  source ${SCRIPT_DIR}/setup_env.sh"
        echo ""
        echo "Then build with:"
        echo "  cd ${SCRIPT_DIR}/build"
        echo "  cmake -DCMAKE_BUILD_TYPE=Release .."
        echo "  make"
    else
        echo "Test execution failed"
    fi
    rm -f test_onnx test_onnx.cpp
else
    echo "Test compilation failed - this is expected if g++ is not available"
    echo "ONNX Runtime files are installed in: ${DEPS_DIR}/onnxruntime"
    rm -f test_onnx.cpp
fi

echo ""
echo "Installation complete!"