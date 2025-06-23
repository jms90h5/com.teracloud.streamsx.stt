#pragma once

// Isolated ONNX Runtime includes to prevent macro conflicts with SPL/Streams
// This prevents NO_EXCEPTION macro from conflicting with SPL-generated code

// Include ONNX headers with macro renamed to avoid conflicts
#define NO_EXCEPTION ONNX_NO_EXCEPTION
#include <onnxruntime_cxx_api.h>
#undef NO_EXCEPTION

// Create type aliases to use ONNX types without namespace pollution
namespace OnnxIsolated {
    using Session = Ort::Session;
    using SessionOptions = Ort::SessionOptions;
    using MemoryInfo = Ort::MemoryInfo;
    using Value = Ort::Value;
    using AllocatorWithDefaultOptions = Ort::AllocatorWithDefaultOptions;
    using Env = Ort::Env;
    using RunOptions = Ort::RunOptions;
}