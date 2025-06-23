# ACTUAL CURRENT STATE AND DETAILED FIX PLAN

**Date**: 2025-06-17  
**Status**: COMPILATION FAILURE - FUNDAMENTAL C++ HEADER CONFLICT  
**Truth**: Previous claims of "working" samples were FALSE

## 🚨 ACTUAL CURRENT STATE

### What Actually Works
✅ **Model Export**: 43.6MB ONNX model successfully created from large NeMo checkpoint  
✅ **SPL Syntax**: Samples compile through SPL phase with correct schemas (`rstring transcription`)  
✅ **Operator Definition**: NeMoSTT operator XML is properly defined  
✅ **Directory Structure**: Proper Streams sample pattern identified and implemented  

### What is BROKEN (and was always broken)
❌ **C++ Compilation**: Fundamental header conflict prevents any compilation  
❌ **Runtime**: Has never actually run due to compilation failure  
❌ **Previous Claims**: All documentation claiming "working" status was false  

### The Core Issue
```
/deps/onnxruntime/include/onnxruntime_c_api.h:133:22: error: expected unqualified-id before 'noexcept'
  133 | #define NO_EXCEPTION noexcept
```

**Root Cause**: ONNX Runtime 1.16.3 headers define `NO_EXCEPTION` as a macro, which conflicts with some other code that expects it to be an identifier.

## 📋 DETAILED FIX PLAN

### Phase 1: IDENTIFY THE EXACT CONFLICT SOURCE
**Timeline**: 30 minutes  
**Approach**: Systematic investigation using compiler preprocessing

#### Step 1.1: Generate Preprocessed Output
```bash
cd samples/output/BasicNeMoTest/build
make VERBOSE=1 2>&1 | grep "g++ .*Transcription.cpp" | head -1 > compile_cmd.txt
# Extract the exact compilation command
# Add -E flag to generate preprocessed output only
g++ -E [original_flags] src/operator/Transcription.cpp > preprocessed.cpp 2>preprocess_errors.txt
```

#### Step 1.2: Locate Conflict Point
```bash
# Find where NO_EXCEPTION is defined vs where it's used as identifier
grep -n "NO_EXCEPTION" preprocessed.cpp | head -20
# Look for the specific line 133 mentioned in error
sed -n '130,140p' preprocessed.cpp
```

#### Step 1.3: Identify Conflicting Headers
- **Resource**: Compiler preprocessor output
- **Method**: Trace back through `#include` stack to find what's trying to use `NO_EXCEPTION` as identifier
- **Expected**: Some Streams SDK header or SPL-generated code conflicting with ONNX

### Phase 2: IMPLEMENT TARGETED HEADER FIX
**Timeline**: 45 minutes  
**Approach**: Isolate ONNX headers from conflicting code

#### Step 2.1: Create Proper Isolation
Based on findings from Phase 1, implement one of:

**Option A: Namespace Isolation**
```cpp
// onnx_isolated.hpp
namespace onnx_isolated {
    #include <onnxruntime_cxx_api.h>
}
using OrtSession = onnx_isolated::Ort::Session;
// etc.
```

**Option B: Macro Redefinition**
```cpp
// Before including ONNX
#define NO_EXCEPTION_SAVE NO_EXCEPTION
#undef NO_EXCEPTION
#include <onnxruntime_cxx_api.h>
#define NO_EXCEPTION NO_EXCEPTION_SAVE
```

**Option C: Forward Declarations**
```cpp
// Avoid including ONNX headers in main files
// Use pimpl pattern with forward declarations
```

#### Step 2.2: Update All Implementation Files
- **Files to modify**: `NeMoCTCImpl.hpp`, `NeMoCTCImpl.cpp`, and any others including ONNX
- **Resource**: Generated operator code in `samples/output/BasicNeMoTest/src/`
- **Method**: Replace direct ONNX includes with isolated wrapper

### Phase 3: VERIFY COMPILATION SUCCESS
**Timeline**: 15 minutes  
**Approach**: Clean build test

#### Step 3.1: Clean Build
```bash
cd samples
rm -rf output/BasicNeMoTest
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh
sc -a -t .. -M com.teracloud.streamsx.stt.sample::BasicNeMoTest --output-directory output/BasicNeMoTest
```

#### Step 3.2: Verify Success Criteria
- ✅ No compilation errors
- ✅ .sab file generated
- ✅ All libraries linked properly

### Phase 4: RUNTIME VALIDATION
**Timeline**: 30 minutes  
**Approach**: Actual execution test

#### Step 4.1: Job Submission
```bash
streamtool submitjob output/BasicNeMoTest/com.teracloud.streamsx.stt.sample.BasicNeMoTest.sab
```

#### Step 4.2: Verify Actual Output
- **Expected**: `"it was the first great song of his life..."`
- **Method**: Check job logs and output files
- **Resource**: `streamtool getlog` and generated transcript files

## 🔍 RESOURCES FOR IDENTIFICATION

### Primary Resources
1. **Compiler Preprocessor**: `g++ -E` to see exact macro expansions
2. **ONNX Runtime Documentation**: https://onnxruntime.ai/docs/api/c/
3. **Streams SPL Compiler Generated Code**: `samples/output/BasicNeMoTest/src/operator/`
4. **ONNX Runtime Headers**: `deps/onnxruntime/include/onnxruntime_c_api.h` lines 130-140

### Secondary Resources
1. **Streams Toolkit Development Guide**: https://doc.streams.teracloud.com/index.html
2. **C++ Header Conflict Resolution Patterns**: Standard techniques for macro conflicts
3. **Working ONNX Runtime Examples**: Look for similar integrations that avoid this conflict

### Code Inspection Points
1. **Generated Operator Code**: What headers does Streams generate that conflict?
2. **SPL Compiler Output**: What macros/identifiers does it generate?
3. **ONNX Header Chain**: What other headers pull in conflicting definitions?

## 🎯 SUCCESS CRITERIA

### Compilation Success
- [ ] Clean C++ compilation with no macro conflicts
- [ ] Successful linking of all libraries
- [ ] Generated .sab file exists and is valid

### Runtime Success  
- [ ] Job submits without errors
- [ ] Audio processing occurs without crashes
- [ ] Actual transcription output generated
- [ ] Output matches expected: `"it was the first great song of his life..."`

### Documentation Accuracy
- [ ] No false claims about "working" status
- [ ] Accurate documentation of what actually works vs. what's broken
- [ ] Clear reproduction steps for any claimed functionality

## ⚠️ COMMITMENT TO HONESTY

I will NOT claim anything is "working" unless:
1. It compiles cleanly without errors
2. It runs successfully in the runtime environment  
3. It produces the expected output
4. I have personally verified all three steps

No more false documentation. No more wasted time on rediscovering the same broken state.

The goal is to fix the actual compilation issue, not to create more misleading documentation about non-functional code.