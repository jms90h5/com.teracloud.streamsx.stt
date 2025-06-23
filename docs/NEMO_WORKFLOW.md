<!--
  NeMo export & integration workflow.
  Extracted from NEMO_INTEGRATION_PLAN.md, NEMO_WORK_PLAN.md,
  NEMO_RESTORATION_PLAN.md, NEMO_INTEGRATION_STATUS.md,
  and NEMO_PYTHON_DEPENDENCIES.md.
-->
# NeMo Integration Workflow

This guide describes how to export, install, and integrate NVIDIA NeMo
FastConformer CTC models into the Streams STT toolkit, including Python
dependencies, restoration plans, and status tracking.

## Integration Plan
`NEMO_INTEGRATION_PLAN.md`

## Detailed Work Plan
`NEMO_WORK_PLAN.md`

## Restoration Plan
`NEMO_RESTORATION_PLAN.md`

## Integration Status
`NEMO_INTEGRATION_STATUS.md`

## Python Dependencies
`NEMO_PYTHON_DEPENDENCIES.md`

## Correct ONNX Export Process

The NVIDIA NeMo FastConformer model must be exported in **CTC-only** mode for stateless,
reliable inference. Hybrid RNNT exports are complex and error-prone. Follow this CTC export recipe:

```python
import nemo.collections.asr as nemo_asr

# Load pretrained FastConformer CTC model
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    "nvidia/stt_en_fastconformer_ctc_large"
)

# Configure CTC-only export (no RNNT joint decoder)
model.set_export_config({"decoder_type": "ctc"})

# Export to ONNX
model.export("models/fastconformer_ctc_export/model.onnx")

# Generate vocabulary file with BPE tokens and blank symbol
with open("models/fastconformer_ctc_export/tokens.txt", "w") as f:
    for i, token in enumerate(model.decoder.vocabulary):
        f.write(f"{token} {i}\n")
    # Append blank token at the end
    f.write(f"<blk> {len(model.decoder.vocabulary)}\n")
```

**Why:**
- CTC-only exports produce a single stateless ONNX graph without hidden cache state.
- Simplifies C++ integration by removing RNNT joint decoder complexity.
- Proven workflow used by downstream Sherpa-ONNX examples and production pipelines.

<!-- End of NeMo Integration Workflow -->
