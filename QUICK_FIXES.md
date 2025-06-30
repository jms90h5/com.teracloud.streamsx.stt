# STT Toolkit Quick Fixes Reference

## 🚀 Quick Test to Verify Everything Works

```bash
# From toolkit root directory:
cd test
g++ -std=c++14 -O2 -I../impl/include -I../lib/onnxruntime/include \
    test_with_audio.cpp ../impl/lib/libs2t_impl.so \
    -L../lib/onnxruntime/lib -lonnxruntime -ldl \
    -Wl,-rpath,'$ORIGIN/../impl/lib' -Wl,-rpath,'$ORIGIN/../lib/onnxruntime/lib' \
    -o test_with_audio
./test_with_audio

# Should output: "it was the first great"
```

## 🔧 Common Fixes

### 1. SPL Won't Compile - "Lorem" Error

**Option A: Delete problematic files (Recommended)**
```bash
# Remove files with spaces from venv (one-time fix)
find impl/venv -name "* *" -type f -delete

# Now compile SPL normally
cd samples && make
```

**Option B: Temporarily move venv**
```bash
# Before compiling SPL:
mv impl/venv /tmp/backup

# Compile SPL
cd samples && make

# Restore after:
mv /tmp/backup impl/venv
```

### 2. Empty Tokens File After Export
```bash
# If tokens.txt is 0 bytes:
python fix_tokens.py
# OR create it:
cat > fix_tokens.py << 'EOF'
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
)
with open("opt/models/fastconformer_ctc_export/tokens.txt", 'w') as f:
    tokenizer = model.tokenizer
    for i in range(tokenizer.tokenizer.get_piece_size()):
        f.write(f"{tokenizer.tokenizer.id_to_piece(i)}\n")
print("✅ Generated tokens")
EOF
python fix_tokens.py
```

### 3. Wrong Transcription Output Schema
```spl
// WRONG:
stream<rstring text> Transcription = NeMoSTT(AudioStream) { ... }

// CORRECT:
stream<rstring transcription> Transcription = NeMoSTT(AudioStream) { ... }
```

### 4. Model Not Found Errors
```bash
# Check paths - model should be in:
opt/models/fastconformer_ctc_export/model.onnx
opt/models/fastconformer_ctc_export/tokens.txt

# If missing, export model:
source ./activate_python.sh
python impl/bin/export_model_ctc_patched.py
```

### 5. C++ Dimension Errors
**Already fixed in implementation!** Model expects `[batch, features=80, time]`.
If you see dimension errors, make sure you rebuilt the library:
```bash
cd impl && make clean && make
```

## 📋 Complete Setup Checklist

```bash
# 1. Setup environment
./setup.sh
source .envrc

# 2. Install NeMo dependencies
source ./activate_python.sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.18.0 onnxruntime==1.22.0 librosa soundfile scipy numpy
pip install nemo_toolkit[asr]==2.3.1

# 3. Export model
python impl/bin/export_model_ctc_patched.py
# Fix tokens if needed
[ -s "opt/models/fastconformer_ctc_export/tokens.txt" ] || python fix_tokens.py

# 4. Build C++ implementation
cd impl && make

# 5. Test C++
cd ../test
make test_with_audio
./test_with_audio
# Should output: "it was the first great"

# 6. Build SPL samples (with venv workaround)
cd ..
mv impl/venv /tmp/backup
cd samples && make
cd .. && mv /tmp/backup impl/venv
```

## 🔍 Debug Commands

```bash
# Check model files
ls -lh opt/models/fastconformer_ctc_export/
# Should show model.onnx (~438MB) and tokens.txt (~7.7KB)

# Check tokens count
wc -l opt/models/fastconformer_ctc_export/tokens.txt
# Should be 1024

# Check C++ library
ls -la impl/lib/libs2t_impl.so
# Should exist and be ~400KB

# Test Python environment
source ./activate_python.sh
python -c "import nemo; print('NeMo version:', nemo.__version__)"
```

## 📚 Full Documentation

- Detailed issues: `doc/KNOWN_ISSUES.md`
- Setup guide: `doc/SETUP_GUIDE.md`
- Test instructions: `test/README.md`
- Architecture: `doc/ARCHITECTURE.md`