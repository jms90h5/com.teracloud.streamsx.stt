# Test Directory

This directory contains test files and scripts that were moved from the main codebase to ensure production cleanliness.

## Test Files

- `test_real_nemo.cpp` - Comprehensive NeMo model testing
- `test_nemo_improvements.cpp` - Feature extraction testing  
- `test_nemo_standalone.cpp` - Standalone NeMo testing
- `run_nemo_test.sh` - Test execution script
- `verify_nemo_setup.sh` - Setup verification script

## Building Tests

```bash
# From main directory
cd test
g++ -std=c++14 -I ../impl/include -I ../deps/onnxruntime/include \
    -L ../impl/lib -L ../deps/onnxruntime/lib \
    -lonnxruntime -ldl test_real_nemo.cpp ../impl/lib/libs2t_impl.so \
    -o test_real_nemo

# Run with proper library paths
export LD_LIBRARY_PATH=$PWD/../deps/onnxruntime/lib:$PWD/../impl/lib:$LD_LIBRARY_PATH
./test_real_nemo
```

## Note

These test files are for development and verification purposes only. 
They should not be included in production deployments.