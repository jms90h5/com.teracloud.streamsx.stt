#include <iostream>

// Define NO_EXCEPTION before including ONNX headers to simulate the conflict
#define NO_EXCEPTION

#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Test successful" << std::endl;
    return 0;
}