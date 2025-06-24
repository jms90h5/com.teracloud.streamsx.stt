<!--
  End-to-end validation & recovery plan for the Streams STT toolkit.
  Tracks progress across model export, Python tests, C++ demos, and SPL samples.
  Highlights memory-heavy steps that may require manual execution.
-->
# Validation & Recovery Plan

Use this plan to systematically validate the toolkit and to resume where you left off.

## 0. Set up Python environment

> **Install only the ASR subset** to avoid megatron-core build failures.

```bash
# Create and activate a virtualenv
python3 -m venv venv_nemo
source venv_nemo/bin/activate

# Upgrade build tools and install minimal packages
pip install --upgrade pip setuptools wheel cython
pip install nemo-asr onnxruntime librosa numpy
```

## 1. Acquire & Verify Hybrid Model

> **Memory-Heavy:** Exporting the full NeMo streaming hybrid model can consume multiple GBs of RAM.
> **Tip:** Run the export steps manually in a separate shell if your agent may be interrupted.

#```bash
# 1.1 Activate NeMo environment
source venv_nemo/bin/activate

# 1.2 Run the export helper script (avoids interactive here-doc)
./export_hybrid_model.py

# 1.3 Confirm the ONNX file exists
ls -lh models/hybrid_streaming/fastconformer_hybrid_streaming.onnx
```bash

**Checkpoint 1:** ONNX model file exists and has non-zero size.

## 2. Python-Based Regression Tests

```bash
# 2.1 Run NeMo test harness
test/run_nemo_test.sh \
  --nemo-model models/hybrid_streaming/fastconformer_hybrid_streaming.onnx \
  --audio-file test_data/audio/librispeech-1995-1837-0001.raw --verbose

# 2.2 Stateless chunk & context experiment
python3 << 'PY'
import numpy as np, onnxruntime as ort, librosa

# Extract real mel features for one utterance
audio, sr = librosa.load("test_data/audio/librispeech-1995-1837-0001.wav", sr=16000)
mel = librosa.feature.melspectrogram(audio, sr=sr, n_fft=512, hop_length=160, n_mels=80)
logm = np.log(mel + 1e-10)[None, :, :]

sess = ort.InferenceSession(
    "models/hybrid_streaming/fastconformer_hybrid_streaming.onnx"
)
inp = sess.get_inputs()[0].name
out1 = sess.run(None, {inp: logm})[0]
out2 = sess.run(None, {inp: logm})[0]
print("Stateless repeat identical?", np.allclose(out1, out2))
PY

# 2.3 Check tokens file
wc -l models/hybrid_streaming/tokens.txt
```

**Checkpoint 2:** Python tests pass; stateless check prints True/False; tokens.txt is present.

## 3. C++ Integration & Demos

```bash
# 3.1 Build the core library
cd impl && make clean && make && cd ..

# 3.2 Build the C++ NeMo demo
cd samples/CppONNX_OnnxSTT && make -f Makefile.nemo clean all && cd ../..

# 3.3 Run the C++ demo test
test/run_nemo_test.sh --nemo-model models/hybrid_streaming/fastconformer_hybrid_streaming.onnx
```

**Checkpoint 3:** C++ demo compiles and runs, producing valid transcription.

## 4. Streams SPL Sample Applications

```bash
# 4.1 Build the toolkit & samples
spl-make toolkit.xml
spl-make samples/com.teracloud.streamsx.stt/Makefile

# 4.2 Run the SPL sample (80ms overlap streaming)
streamtool run samples/com.teracloud.streamsx.stt/NeMoCTCSample.spl \
  --application com.teracloud.streamsx.stt.NeMoCTCSample \
  --config encoderModel=models/hybrid_streaming/fastconformer_hybrid_streaming.onnx \
           vocabFile=models/hybrid_streaming/tokens.txt \
           cmvnFile=none modelType=NEMO_STREAMING \
           blankId=<blankTokenId> streamingMode=true \
           chunkOverlapMs=80

# 4.3 Test multiple latency modes: 0, 80, 480, 1040ms
for overlap in 0 80 480 1040; do
  echo "Overlap ${overlap}ms"
  streamtool run samples/com.teracloud.streamsx.stt/NeMoCTCSample.spl \
    --config streamingMode=true chunkOverlapMs=${overlap}
done
```

**Checkpoint 4:** SPL application runs successfully, streaming transcriptions at each mode.

## 5. Logging & Recovery

- After each section, mark this plan with ✅ if successful.
- If a heavy export step OOMs or crashes, rerun it manually and resume at that step.
- Record failures or deviations as GitHub issues for follow-up.

_End of Validation & Recovery Plan_