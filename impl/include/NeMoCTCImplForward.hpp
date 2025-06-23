#pragma once

#include <vector>
#include <string>
#include <memory>

// Forward declaration to avoid including ONNX headers
class NeMoCTCImpl;

// Factory function
std::unique_ptr<NeMoCTCImpl> createNeMoCTCImpl();