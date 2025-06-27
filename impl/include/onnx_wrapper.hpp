#pragma once

// Isolate ONNX Runtime headers to prevent macro conflicts (e.g., NO_EXCEPTION)
#include <onnxruntime_cxx_api.h>

// Undefine the conflicting macro after including the header.
// This prevents it from affecting other headers.
#undef NO_EXCEPTION