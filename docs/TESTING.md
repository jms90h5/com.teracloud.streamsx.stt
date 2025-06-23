<!--
  Testing and validation strategy.
  Extracted from FASTCONFORMER_TEST_GUIDE.md,
  SYSTEMATIC_TESTING_PLAN.md,
  RESTORE_WORKING_SAMPLES_PLAN.md,
  FEATURE_EXTRACTION_ROOT_CAUSE.md.
-->
# Testing & Validation

This document outlines the systematic testing approach for the toolkit,
including feature‑extraction parity checks, sample restoration,
and deep‑dive root cause analyses.

> **Note:** All synthetic/fake‑data tests have been removed.  All validation uses real audio and ground‑truth feature vectors.

## FastConformer Test Guide
`FASTCONFORMER_TEST_GUIDE.md`

## Systematic Testing Plan
`SYSTEMATIC_TESTING_PLAN.md`

## Restore Working Samples Plan
`RESTORE_WORKING_SAMPLES_PLAN.md`

## Feature Extraction Root Cause Analysis
`FEATURE_EXTRACTION_ROOT_CAUSE.md`

## Stateless Chunk & Context Experiments

The CTC-only ONNX model is stateless by default and loses context between independent chunks.
Overlapping‑chunk strategies or cache-enabled exports must be used to maintain streaming context.

```python
# Example from legacy tests (test_current_model.py):
features = np.load("test_data/features.npy")  # shape [1, T, 80]

# Inference on full sequence
full_logits = session.run(None, {"input": features})[0]

# Overlapping chunks (50% overlap)
chunk1 = features[:, :160, :]
chunk2 = features[:,  80:240, :]
chunk3 = features[:, 160:, :]
out1 = session.run(None, {"input": chunk1})[0]
out2 = session.run(None, {"input": chunk2})[0]
out3 = session.run(None, {"input": chunk3})[0]

# Compare token sequences
tokens_full = np.argmax(full_logits[0], axis=-1)
tokens_c1   = np.argmax(out1[0], axis=-1)
tokens_c2   = np.argmax(out2[0], axis=-1)
tokens_c3   = np.argmax(out3[0], axis=-1)
```

**Why:**
- Confirms the model produces identical outputs for the same chunk (no internal cache).
- Highlights need for stateful export (cache tensors) or careful chunk stitching.

<!-- End of Testing & Validation -->
