#pragma once

// Isolated ONNX Runtime includes to prevent macro conflicts with SPL/Streams
// This prevents NO_EXCEPTION macro from conflicting with SPL-generated code

// Include ONNX headers with macro isolation to prevent conflicts
#ifdef NO_EXCEPTION
#define SAVE_NO_EXCEPTION NO_EXCEPTION
#undef NO_EXCEPTION
#endif

#include <onnxruntime_cxx_api.h>

#ifdef SAVE_NO_EXCEPTION
#define NO_EXCEPTION SAVE_NO_EXCEPTION
#undef SAVE_NO_EXCEPTION
#endif

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