# NeMo Integration Work Plan

## Objective
Complete the integration of NVIDIA NeMo cache-aware Conformer model support into the TeraCloud Streams STT toolkit and successfully test it with a REAL NeMo model.

## Current Situation
- ✅ All C++ code is implemented and compiles successfully
- ✅ Sample applications are created and build properly
- ✅ A real NeMo model has been downloaded from NVIDIA NGC
- ❌ The model needs to be exported from PyTorch format to ONNX format
- ❌ Python environment has dependency conflicts preventing export

## Work Plan

### Phase 1: Fix Python Environment and Export Model
**Priority: CRITICAL - This blocks everything else**

1. **Fix huggingface_hub compatibility**
   ```bash
   # Option A: Downgrade huggingface_hub
   pip3 install --user --force-reinstall "huggingface_hub==0.19.4"
   
   # Option B: Install older NeMo version
   pip3 uninstall -y nemo_toolkit
   pip3 install --user nemo_toolkit==1.22.0
   ```

2. **Export the downloaded NeMo model to ONNX**
   ```bash
   cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt
   python3 export_real_nemo.py
   ```
   
   Expected output:
   - `models/nemo_fastconformer_streaming/conformer_ctc.onnx`
   - `models/nemo_fastconformer_streaming/tokenizer.txt`

3. **Verify the exported model**
   - Check file size (should be ~40-50MB)
   - Verify it's a valid ONNX file
   - Check input/output tensor names

### Phase 2: Test with Real Model
**Can only proceed after Phase 1 succeeds**

1. **Run the standalone test**
   ```bash
   cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt
   ./run_nemo_test.sh --nemo-model models/nemo_fastconformer_streaming/conformer_ctc.onnx \
                      --audio-file test_data/audio/librispeech-1995-1837-0001.raw \
                      --verbose
   ```

2. **Expected issues to fix:**
   - Input/output tensor name mismatches
   - Cache tensor dimension mismatches
   - Missing tokenizer/vocabulary handling

### Phase 3: Fix Implementation Based on Real Model
**Based on errors from Phase 2**

1. **Update NeMoCacheAwareConformer.cpp**
   - Fix tensor names to match exported model
   - Adjust cache dimensions if needed
   - Add proper vocabulary/tokenizer loading

2. **Potential changes needed:**
   - The real model might not have cache tensors (CTC models typically don't)
   - May need to implement a different class for CTC models vs Transducer models
   - Input might be different shape than expected

### Phase 4: Complete End-to-End Testing

1. **Test with multiple audio files**
   ```bash
   for audio in test_data/audio/*.wav; do
     ./run_nemo_test.sh --nemo-model models/nemo_fastconformer_streaming/conformer_ctc.onnx \
                        --audio-file "$audio"
   done
   ```

2. **Test SPL application**
   - Build and run the SPL sample
   - Verify streaming performance

### Phase 5: Documentation Update

1. Update README with:
   - How to export NeMo models
   - Supported NeMo model types
   - Performance benchmarks

2. Create example for users

## Alternative Approach (If Python Issues Persist)

### Option A: Use Docker
```bash
docker run --rm -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.01 \
  python /workspace/export_real_nemo.py
```

### Option B: Find Pre-Exported ONNX Models
- Search Hugging Face for NeMo ONNX models
- Check NVIDIA forums for shared exports
- Use a different streaming model (like Sherpa-ONNX models)

### Option C: Implement Non-Cache Version First
- The downloaded model is CTC, which doesn't use cache
- Simplify implementation to handle CTC models without cache
- Add cache support later for Transducer models

## Key Files to Focus On

1. **For export issues:**
   - `/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/export_real_nemo.py`
   - Check Python environment with `pip3 list | grep -E "nemo|hugging"`

2. **For implementation fixes:**
   - `impl/src/NeMoCacheAwareConformer.cpp` - Main implementation
   - `impl/src/ModelFactory.cpp` - Factory function
   - `samples/CppONNX_OnnxSTT/test_nemo_standalone.cpp` - Test app

3. **Model location:**
   - Downloaded: `models/nemo_real/stt_en_conformer_ctc_small.nemo`
   - Need to export to: `models/nemo_fastconformer_streaming/conformer_ctc.onnx`

## Success Criteria
1. Successfully export NeMo model to ONNX format
2. Run test_nemo_standalone without crashes
3. Get actual transcription output (even if accuracy is poor)
4. Document the complete process for users

## DO NOT:
- Create mock models
- Skip steps to save time
- Use different model architectures as substitutes
- Give up on the NVIDIA NeMo model

## IMPORTANT: 
The user explicitly requested NO shortcuts. The NVIDIA NeMo model MUST work end-to-end with real data and real models.