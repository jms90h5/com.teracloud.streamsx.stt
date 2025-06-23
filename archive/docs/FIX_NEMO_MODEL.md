# Fix NeMo Model - Action Plan

## Problem Confirmed
The current ONNX model is **fundamentally broken**:
- Only 533K parameters (should be 10M+)  
- Missing most weights (constant-folded during export)
- Size 2.5MB (should be 40MB+)
- Outputs same token regardless of input

## Solution Options (in order of priority)

### Option 1: Re-export ONNX Model Correctly ⭐
**Timeline: 1-2 hours**

```bash
# Install NeMo toolkit
pip install nemo_toolkit[asr]

# Re-export the model properly
python3 -c "
import nemo.collections.asr as nemo_asr

# Load the .nemo checkpoint
model = nemo_asr.models.EncDecCTCModel.restore_from('models/nemo_real/stt_en_conformer_ctc_small.nemo')

# Export to ONNX with proper settings
model.export('models/nemo_real/conformer_ctc_fixed.onnx', 
             verbose=True,
             check_trace=True,
             dynamic_axes={'audio_signal': {0: 'batch', 1: 'time'}})
"

# Update C++ code to use the fixed model
```

### Option 2: Download Working Pre-trained Model ⭐⭐
**Timeline: 30 minutes**

```bash
# Download from NVIDIA NGC
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.15.0/files/stt_en_conformer_ctc_small.nemo

# Or use model hub
python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.from_pretrained('stt_en_conformer_ctc_small')
model.export('models/working_conformer.onnx')
"
```

### Option 3: Use Different Model Architecture ⭐⭐⭐
**Timeline: 45 minutes**

Use a model that's known to export well:
- Try Wav2Vec2 (better ONNX support)
- Try Whisper ONNX (proven to work)
- Try SpeechBrain models

### Option 4: Fix Export Parameters
**Timeline: 1 hour**

The current export might have wrong parameters:
```python
# Better export settings
model.export(
    'conformer_fixed.onnx',
    dynamic_axes={'audio_signal': {0: 'batch', 1: 'time'}},
    opset_version=12,  # Try different opset
    do_constant_folding=False,  # Prevent weight folding
    verbose=True
)
```

## Immediate Action

Let me implement **Option 2** first (download working model) since it's fastest:

```bash
# 1. Install minimal NeMo
pip install torch torchaudio
pip install nemo_toolkit

# 2. Download and export working model
python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.from_pretrained('stt_en_conformer_ctc_small')
print('Model loaded successfully')
print(f'Model size: {sum(p.numel() for p in model.parameters()):,} parameters')

# Test transcription first
transcription = model.transcribe(['test_data/11-ibm-culture-2min.wav'])
print(f'Python transcription: {transcription[0]}')

# Export to ONNX
model.export('models/working_conformer_ctc.onnx')
print('Model exported successfully')
"

# 3. Update C++ code to use new model
# 4. Test with IBM audio
```

## Expected Results
After fixing the model:
- Model size: ~40MB (10M+ parameters)
- Different outputs for different inputs
- Actual transcription of IBM culture audio
- Token variety (not just token 95)

## Verification Steps
1. Check model size is >10MB
2. Verify parameter count >1M
3. Test with different inputs produce different outputs
4. Get actual English transcription