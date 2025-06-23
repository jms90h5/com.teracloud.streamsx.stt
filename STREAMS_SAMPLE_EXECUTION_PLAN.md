# Detailed Plan: Running Teracloud Streams STT Sample Applications

**Goal**: Demonstrate the Teracloud Streams STT toolkit sample applications running and producing correct transcription output.

## Current Status

### Completed Work
- ✅ NeMo FastConformer CTC model exported to ONNX
- ✅ Standalone C++ test program produces correct output: "it was the first great song of his life"
- ✅ OnnxSTT operator updated to support NeMo CTC models
- ✅ SPL sample applications created (NeMoCTCSample.spl, BasicNeMoTest.spl)
- ✅ All compilation warnings fixed
- ✅ Documentation updated

### What Needs to Be Done
Build and run the Streams applications to prove they work correctly.

## Execution Plan

### Phase 1: Build the SPL Toolkit
1. **Build the toolkit using Streams compiler**
   ```bash
   cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt
   make toolkit
   ```
   - This will compile all operators and create the toolkit index

2. **Verify toolkit build**
   - Check for `toolkit.xml` updates
   - Verify operator compilation succeeded
   - Check for any build errors

### Phase 2: Compile SPL Sample Applications

1. **Compile NeMoCTCSample.spl**
   ```bash
   cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/samples
   sc -M NeMoCTCSample -t ../
   ```
   - This compiles the SPL application
   - `-M` specifies the main composite
   - `-t` specifies the toolkit path

2. **Compile BasicNeMoTest.spl**
   ```bash
   cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/samples/com.teracloud.streamsx.stt.sample
   sc -M BasicNeMoTest -t ../../
   ```

### Phase 3: Run Applications and Verify Output

1. **Run NeMoCTCSample**
   ```bash
   cd output/NeMoCTCSample/bin
   ./standalone
   ```
   
   **Expected Output**:
   ```
   === NeMo CTC Transcription ===
   Text: it was the first great
   Is Final: true
   Confidence: 0.917
   Expected: 'it was the first great'
   =============================
   ```
   
   **Verification**:
   - Check console output matches expected
   - Verify `nemo_ctc_output.txt` created with transcription

2. **Run BasicNeMoTest**
   ```bash
   cd output/BasicNeMoTest/bin
   ./standalone
   ```
   
   **Expected Output**:
   ```
   BasicNeMoTest Transcription: it was the first great song of his life
   ```
   
   **Verification**:
   - Check console output shows full transcription
   - Verify `BasicNeMoTest_transcript_*.txt` created

### Phase 4: Troubleshooting Guide

#### If Toolkit Build Fails
1. Check for missing dependencies
2. Verify STREAMS_INSTALL environment variable
3. Check operator XML syntax
4. Review compilation logs

#### If SPL Compilation Fails
1. Check SPL syntax errors
2. Verify operator parameters match XML definition
3. Check file paths are correct
4. Ensure toolkit path is correct

#### If Runtime Fails
1. Check model files exist at specified paths
2. Verify audio files exist
3. Check for ONNX Runtime library issues
4. Review operator logs for initialization errors

#### If Output is Incorrect
1. Compare with standalone test program output
2. Check audio file format matches expectations
3. Verify model parameters (blank_id, model_type)
4. Enable debug logging in operator

### Phase 5: Success Criteria

The plan is successful when:
1. ✅ Both sample applications compile without errors
2. ✅ Applications run without crashes
3. ✅ Console output shows correct transcriptions
4. ✅ Output files contain expected text
5. ✅ Performance metrics show reasonable RTF

### Phase 6: Documentation

After successful execution:
1. Create `STREAMS_SAMPLES_WORKING.md` with:
   - Exact commands used
   - Actual output captured
   - Any issues encountered and solutions
   - Performance metrics

2. Update `README.md` with:
   - Working examples section
   - Quick start guide for users

## Key Files to Monitor

1. **Compilation Output**
   - `output/*/Makefile` - Generated makefiles
   - `output/*/bin/standalone` - Executable files

2. **Runtime Output**
   - Console logs with transcriptions
   - `nemo_ctc_output.txt` - NeMoCTCSample output
   - `BasicNeMoTest_transcript_*.txt` - BasicNeMoTest output

3. **Error Logs**
   - Application runtime logs
   - ONNX Runtime errors
   - Streams operator traces

## Risk Mitigation

1. **Model Path Issues**
   - Use absolute paths initially
   - Verify getThisToolkitDir() returns correct path

2. **Library Dependencies**
   - Ensure LD_LIBRARY_PATH includes ONNX Runtime
   - Check libwenetcpp.so is accessible

3. **Audio Format**
   - Verify WAV files are 16kHz, 16-bit, mono
   - Test with known working audio first

## Next Steps After Success

Once samples are working:
1. Create additional samples for different use cases
2. Implement streaming mode functionality
3. Add performance benchmarking
4. Create production deployment guide

## Summary

This plan focuses on the critical goal: demonstrating that the Teracloud Streams STT toolkit sample applications run correctly and produce the expected transcription output. The plan is straightforward:
1. Build the toolkit
2. Compile the samples
3. Run them
4. Verify the output matches expectations

Success is measured by actual execution producing correct transcriptions, not by code that "should" work.