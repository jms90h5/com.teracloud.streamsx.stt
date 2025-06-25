#!/usr/bin/env python3
"""
Export the NVIDIA NeMo FastConformer Hybrid streaming model (CTC+RNNT)
with cache‑aware tensors for real-time streaming support.
"""
import nemo.collections.asr as nemo_asr

def main():
    print("Loading pretrained hybrid streaming model...")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    )

    print("Exporting ONNX with cache outputs for streaming...")
    model.export(
        "models/hybrid_streaming/fastconformer_hybrid_streaming.onnx",
        export_as_rnnt=True,
        cache_all_outputs=True
    )
    print("✅ Export complete: models/hybrid_streaming/fastconformer_hybrid_streaming.onnx")

if __name__ == "__main__":
    main()
