#ifndef ONNX_STT_INTERFACE_HPP
#define ONNX_STT_INTERFACE_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace onnx_stt {

/**
 * Interface for speech-to-text that completely hides ONNX Runtime headers
 * This prevents macro conflicts between ONNX Runtime and SPL
 */
class OnnxSTTInterface {
public:
    enum class ModelType {
        CACHE_AWARE_CONFORMER,
        NEMO_CTC
    };
    
    struct Config {
        std::string encoder_onnx_path;
        std::string vocab_path;
        std::string cmvn_stats_path;
        int sample_rate = 16000;
        int chunk_size_ms = 100;
        int num_mel_bins = 80;
        int frame_length_ms = 25;
        int frame_shift_ms = 10;
        int beam_size = 10;
        int blank_id = 0;
        int num_threads = 4;
        bool use_gpu = false;
        ModelType model_type = ModelType::CACHE_AWARE_CONFORMER;
    };
    
    struct TranscriptionResult {
        std::string text;
        bool is_final;
        double confidence;
        uint64_t timestamp_ms;
        uint64_t latency_ms;
    };
    
    struct Stats {
        uint64_t total_audio_ms = 0;
        uint64_t total_processing_ms = 0;
        double real_time_factor = 0.0;
    };
    
    virtual ~OnnxSTTInterface() = default;
    
    virtual bool initialize() = 0;
    virtual TranscriptionResult processAudioChunk(const int16_t* samples, 
                                                 size_t num_samples, 
                                                 uint64_t timestamp_ms) = 0;
    virtual void reset() = 0;
    virtual Stats getStats() const = 0;
};

// Factory function - implementation in .cpp file
std::unique_ptr<OnnxSTTInterface> createOnnxSTT(const OnnxSTTInterface::Config& config);

} // namespace onnx_stt

#endif // ONNX_STT_INTERFACE_HPP