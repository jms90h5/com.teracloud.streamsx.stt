<!--
  Combined implementation, operator updates, and solution summary.
  Extracted from IMPLEMENTATION_COMPLETE.md, IMPLEMENTATION_SUMMARY.md,
  OPERATOR_UPDATES_COMPLETE.md, and SOLUTION_SUMMARY.md.
-->
# Implementation Guide

This document consolidates the key design decisions, build details,
operator updates, and solution findings for the NeMo FastConformer CTC
integration into the Streams STT toolkit.

## Complete Implementation Details
`IMPLEMENTATION_COMPLETE.md`

## Implementation Summary
`IMPLEMENTATION_SUMMARY.md`

## Operator Updates Summary
`OPERATOR_UPDATES_COMPLETE.md`

## Feature Extraction Solution Summary
`SOLUTION_SUMMARY.md`

## Root‑Cause Analysis: Feature Extraction Mismatch

A deep investigation revealed that the original C++ feature extractor (`ImprovedFbank`) used a simple DFT implementation with `floor()` to map mel filter frequencies to FFT bins, producing zero-width filters and constant floor energy values (log of 1e-10 ≈ -23 dB).  This mismatch prevented the FastConformer model from recognizing speech and resulted in blank token outputs.

**Fix applied in `ImprovedFbank.cpp`:**
```cpp
// Original (floor) → zero-width filters
bin_points[i] = floor(n_fft * hz_points[i] / sample_rate);

// Updated (round) + guard against zero-width filters
bin_points[i] = round(n_fft * hz_points[i] / sample_rate);
if (bin_points[i] >= bin_points[i+1]) {
    bin_points[i+1] = bin_points[i] + 1;
}
```

Post‑fix, feature statistics closely match the Python/librosa reference (mean error ~0.4%), and the model no longer outputs only blank tokens.  For the full analysis, see `archive/docs/FEATURE_EXTRACTION_ROOT_CAUSE.md`.

<!-- End of Implementation Guide -->
