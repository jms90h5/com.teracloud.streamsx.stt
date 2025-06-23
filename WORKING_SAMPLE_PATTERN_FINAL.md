# Working Sample Pattern - FINAL DOCUMENTATION

**Date**: 2025-06-17  
**Status**: VERIFIED FROM WORKING CHECKPOINT  
**Source**: `com.teracloud.streamss.stt.checkpoint_06_02_2025.tar.gz`

## 🎯 CRITICAL FINDINGS

### ✅ WORKING SPL PATTERN (Verified from June 2nd checkpoint)

```spl
use com.teracloud.streamsx.stt::*;

composite WorkingBasicTest {
    graph
        // Audio source
        stream<blob audioChunk, uint64 audioTimestamp> AudioStream = FileAudioSource() {
            param
                filename: "../test_data/audio/librispeech-1995-1837-0001.wav";
                blockSize: 16384u;
                sampleRate: 16000;
                bitsPerSample: 16;
                channelCount: 1;
        }
        
        // CRITICAL: Output schema is "rstring transcription", NOT "rstring text"
        stream<rstring transcription> Transcription = NeMoSTT(AudioStream) {
            param
                modelPath: "../models/fastconformer_ctc_export/model.onnx";
                tokensPath: "../models/fastconformer_ctc_export/tokens.txt";
        }
        
        // Display
        () as Display = Custom(Transcription) {
            logic
                onTuple Transcription: {
                    printStringLn("Transcription: " + transcription);
                }
        }
}
```

### 🚨 CRITICAL REQUIREMENTS

1. **Output Schema**: `stream<rstring transcription>` (the C++ implementation outputs `transcription`, not `text`)
2. **No Namespace**: SPL files in samples/ directory must have NO namespace declaration
3. **Composite Name**: Must match filename (e.g., `WorkingBasicTest.spl` → `composite WorkingBasicTest`)
4. **File Paths**: Relative paths from samples/ directory:
   - Model: `"../models/fastconformer_ctc_export/model.onnx"`
   - Tokens: `"../models/fastconformer_ctc_export/tokens.txt"`
   - Audio: `"../test_data/audio/librispeech-1995-1837-0001.wav"`

### ✅ VERIFIED WORKING COMPONENTS

1. **ONNX Model**: ✅ `models/fastconformer_ctc_export/model.onnx` (43.6MB)
2. **Vocabulary**: ✅ `models/fastconformer_ctc_export/tokens.txt` (1023 tokens)
3. **NeMoSTT Operator**: ✅ Implemented in C++ with libnemo_ctc_impl.so
4. **FileAudioSource**: ✅ Available as composite operator
5. **Test Audio**: ✅ `test_data/audio/librispeech-1995-1837-0001.wav`

### 🎯 EXPECTED OUTPUT

**Test File**: `librispeech-1995-1837-0001.wav`  
**Expected Transcription**: `"it was the first great song of his life..."`

### 🚫 BROKEN SAMPLES IDENTIFIED

**Problem**: Later versions (June 3rd+) introduced SPL ternary operator syntax that is INVALID:

```spl
// ❌ BROKEN SYNTAX (causes compilation errors)
stream<blob> ProcessedAudio = 
    $enableThrottling ? 
        Throttle(AudioStream) { ... } :
        AudioStream;
```

**Solution**: Use the June 2nd checkpoint version which has clean, working syntax.

### 📁 WORKING CHECKPOINT LOCATION

```bash
/homes/jsharpe/teracloud/toolkits/com.teracloud.streamss.stt.checkpoint_06_02_2025.tar.gz
```

**Extract working samples**:
```bash
tar -xf com.teracloud.streamss.stt.checkpoint_06_02_2025.tar.gz \
    com.teracloud.streamsx.stt/samples/BasicNeMoDemo.spl
```

### 🔧 COMPILATION COMMAND

```bash
cd samples/
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh
sc -a -t .. -M WorkingBasicTest --output-directory output/WorkingBasicTest
```

### 🚀 JOB SUBMISSION

```bash
streamtool submitjob output/WorkingBasicTest/WorkingBasicTest.sab
```

### 📊 SUCCESS CRITERIA

- ✅ Clean compilation with no syntax errors
- ✅ Job runs without ONNX Runtime errors  
- ✅ Output contains: `"it was the first great song of his life..."`
- ✅ Processing time should be ~30x real-time performance

---

## 🔄 TROUBLESHOOTING PATTERN

**If samples don't work**:

1. **STOP making random changes**
2. **Extract from working checkpoint**: `com.teracloud.streamss.stt.checkpoint_06_02_2025.tar.gz`
3. **Use exact pattern above** - no modifications
4. **Check output schema**: Must be `rstring transcription`
5. **Verify file paths**: All relative from samples/ directory

**Common Mistakes to AVOID**:
- ❌ Using `rstring text` instead of `rstring transcription`
- ❌ Adding namespace declarations in samples/ directory
- ❌ Using ternary operator syntax (not supported in SPL)
- ❌ Wrong file paths for model/tokens/audio
- ❌ Making random syntax changes without checking documentation

---

**REMEMBER**: The June 2nd checkpoint was working. Use it as the source of truth, not later broken versions or random syntax experiments.