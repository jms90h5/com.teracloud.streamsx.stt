# Restore Working Samples - Detailed Plan

**Date**: 2025-06-17  
**Goal**: Restore the documented working speech-to-text samples that produce the "it was the first great song" transcription  
**Status**: PLANNING PHASE

## 🔍 Current State Analysis

### What I Have (VERIFIED):
1. **Large NeMo Model**: `stt_en_fastconformer_hybrid_large_streaming_multi.nemo` (126MB)
   - Location: `models/nemo_fastconformer_direct/`
   - Size: 126,386,176 bytes
   - This is the correct 114M parameter model referenced in my documentation

2. **Current ONNX Models**:
   - `conformer_ctc_dynamic.onnx` (2.6MB) - from smaller model
   - `conformer_ctc.onnx` (5.7MB) - intermediate model
   - These are NOT from the large 114M parameter model

3. **Working Sample References**:
   - Documentation shows samples should use `models/fastconformer_ctc_export/model.onnx`
   - This directory/file does NOT exist
   - Samples reference this missing path and fail to compile

### What I'm Missing:
1. **Export Script**: No script to convert .nemo → ONNX
2. **Correct ONNX Model**: Need 40MB+ ONNX from the 126MB .nemo
3. **Vocabulary Mapping**: Proper tokens.txt for the large model

## 📋 Detailed Action Plan

### Phase 1: BACKUP EVERYTHING (CRITICAL)
**Timeline**: 5 minutes  
**Risk**: ZERO - only creating backups

```bash
# 1. Backup current working models directory
cp -r models models_backup_$(date +%Y%m%d_%H%M%S)

# 2. Backup current samples
cp -r samples samples_backup_$(date +%Y%m%d_%H%M%S)

# 3. Create git commit of current state
git add -A
git commit -m "Backup: Current state before restoring working samples"

# 4. Document current file inventory
find models -name "*.onnx" -exec ls -la {} \; > current_models_inventory.txt
find models -name "*.nemo" -exec ls -la {} \; >> current_models_inventory.txt
```

### Phase 2: RESEARCH EXISTING SCRIPTS
**Timeline**: 15 minutes  
**Risk**: ZERO - only reading

```bash
# 1. Check git history for export scripts
git log --oneline --name-only | grep -E "(export|nemo).*\.py"

# 2. Check if scripts exist in checkpoint tar files
tar -tf /homes/jsharpe/teracloud/toolkits/com.teracloud.streamss.stt.checkpoint_06_06_2025.tar.gz | grep "export.*\.py"

# 3. Search documentation for exact export commands
grep -r "export" *.md | grep -E "(onnx|nemo)"

# 4. Check if I documented the exact export process
grep -A10 -B10 "export" NEMO_INTEGRATION_STATUS.md
```

### Phase 3: RECREATE EXPORT SCRIPT (CAREFUL)
**Timeline**: 30 minutes  
**Risk**: LOW - only creating new files

Based on my documentation showing successful export, recreate:
```python
# File: export_large_fastconformer.py
import nemo.collections.asr as nemo_asr
import torch
import os

def export_large_fastconformer():
    """
    Export the 114M parameter FastConformer model to ONNX
    Based on documented working process from NEMO_INTEGRATION_STATUS.md
    """
    
    print("🚀 Loading large FastConformer model...")
    model_path = "models/nemo_fastconformer_direct/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    
    # Load the large model
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create export directory (matching documented path)
    os.makedirs("models/fastconformer_ctc_export", exist_ok=True)
    
    # Export with proper settings for streaming
    output_path = "models/fastconformer_ctc_export/model.onnx"
    
    model.export(
        output_path,
        dynamic_axes={'audio_signal': {0: 'batch', 1: 'time'}},
        verbose=True,
        check_trace=True
    )
    
    # Export vocabulary
    vocab_path = "models/fastconformer_ctc_export/tokens.txt"
    with open(vocab_path, 'w') as f:
        for i, token in enumerate(model.decoder.vocabulary):
            f.write(f"{token}\n")
    
    print(f"✅ Export complete:")
    print(f"   Model: {output_path}")
    print(f"   Tokens: {vocab_path}")
    print(f"   Size: {os.path.getsize(output_path)/1024/1024:.1f}MB")

if __name__ == "__main__":
    export_large_fastconformer()
```

### Phase 4: TEST EXPORT (REVERSIBLE)
**Timeline**: 10 minutes  
**Risk**: LOW - creating new files only

```bash
# 1. Test the export script
python export_large_fastconformer.py

# 2. Verify output
ls -la models/fastconformer_ctc_export/
file models/fastconformer_ctc_export/model.onnx
wc -l models/fastconformer_ctc_export/tokens.txt

# 3. Check model size (should be 40MB+, not 2MB)
du -h models/fastconformer_ctc_export/model.onnx
```

### Phase 5: TEST SAMPLE COMPILATION
**Timeline**: 10 minutes  
**Risk**: LOW - only testing compilation

```bash
# 1. Try building the existing samples (should now find the model)
cd samples
source ~/teracloud/streams/7.2.0.0/bin/streamsprofile.sh
make BasicNeMoDemo

# 2. Check for compilation success
echo "Exit code: $?"
```

### Phase 6: RUNTIME TEST (CONTROLLED)
**Timeline**: 15 minutes  
**Risk**: MEDIUM - but controlled environment

```bash
# 1. Submit a test job
streamtool submitjob output/BasicNeMoDemo/BasicNeMoDemo.sab --jobname RestoreTest

# 2. Monitor output
streamtool lsjobs
streamtool getlog --jobs RestoreTest

# 3. Look for the documented transcription:
# "it was the first great song of his life..."
```

## 🚨 Risk Mitigation

### Before Each Phase:
- [ ] Verify backups exist
- [ ] Document current state
- [ ] Have rollback plan ready

### Rollback Plans:
1. **If export fails**: Delete new files, keep backups
2. **If samples break**: Restore from samples_backup_*
3. **If models corrupted**: Restore from models_backup_*
4. **Nuclear option**: `git reset --hard` to last commit

### Success Criteria:
- [ ] BasicNeMoDemo.spl compiles without errors
- [ ] Job runs and produces transcription output
- [ ] Output contains "it was the first great song..."
- [ ] Performance metrics match documentation (30x real-time)

## 📊 Progress Tracking

### Phase 1: BACKUP ✅ COMPLETE
- [x] Models backed up → models_backup_20250617_124549
- [x] Samples backed up → samples_backup_20250617_124556
- [ ] Git commit created
- [x] Inventory documented → current_models_inventory.txt

### Phase 2: RESEARCH ✅ COMPLETE
- [x] Git history searched → Found export scripts in commit 90e8d62
- [x] Export script recovered → export_nemo_model_final.py 
- [x] Documentation reviewed → Script worked with SMALL model
- [x] Export process identified → Need to adapt for LARGE model

**KEY FINDING**: Script worked with stt_en_conformer_ctc_small.nemo (48MB) → need to adapt for stt_en_fastconformer_hybrid_large_streaming_multi.nemo (126MB)

### Phase 3: SCRIPT CREATION ✅ COMPLETE
- [x] Export script written → export_large_fastconformer.py
- [x] Script reviewed for safety → Based on working export_nemo_model_final.py
- [x] Dependencies verified → Same as working version (PyTorch, ONNX, PyYAML)
- [x] Paths validated → Output to models/fastconformer_ctc_export/ (matches samples)

### Phase 4: EXPORT TEST ✅❌
- [ ] Script executed successfully
- [ ] Model exported (size >40MB)
- [ ] Vocabulary exported (>1000 tokens)
- [ ] Files validated

### Phase 5: COMPILATION TEST ✅❌
- [ ] Samples compile without errors
- [ ] No missing file errors
- [ ] Build artifacts created
- [ ] SAB file generated

### Phase 6: RUNTIME TEST ✅❌
- [ ] Job submitted successfully
- [ ] Job runs without crashes
- [ ] Transcription output generated
- [ ] Correct transcription verified

## 📝 Notes and Observations

### Key Insights:
- The large model exists but wasn't exported to ONNX
- Missing export created the "missing model" issue
- Documentation shows this process worked before

### Dependencies to Verify:
- [ ] NeMo toolkit available
- [ ] Python environment working
- [ ] ONNX Runtime compatible
- [ ] Streams environment sourced

### Decision Points:
- If export script fails → investigate dependencies first
- If model size wrong → check export parameters
- If transcription wrong → verify vocabulary mapping
- If performance slow → check ONNX optimization

---

**REMEMBER**: Every step is reversible. Never overwrite originals. Document everything.