#include "OnnxSTTInterface.hpp"
#include "OnnxSTTImpl.hpp"

namespace onnx_stt {

/**
 * Concrete implementation that wraps OnnxSTTImpl
 * This is the ONLY file that includes ONNX headers
 */
class OnnxSTTWrapper : public OnnxSTTInterface {
public:
    explicit OnnxSTTWrapper(const Config& config) {
        // Convert interface config to impl config
        OnnxSTTImpl::Config implConfig;
        implConfig.encoder_onnx_path = config.encoder_onnx_path;
        implConfig.vocab_path = config.vocab_path;
        implConfig.cmvn_stats_path = config.cmvn_stats_path;
        implConfig.sample_rate = config.sample_rate;
        implConfig.chunk_size_ms = config.chunk_size_ms;
        implConfig.num_mel_bins = config.num_mel_bins;
        implConfig.frame_length_ms = config.frame_length_ms;
        implConfig.frame_shift_ms = config.frame_shift_ms;
        implConfig.beam_size = config.beam_size;
        implConfig.blank_id = config.blank_id;
        implConfig.num_threads = config.num_threads;
        implConfig.use_gpu = config.use_gpu;
        
        // Convert model type
        if (config.model_type == ModelType::NEMO_CTC) {
            implConfig.model_type = OnnxSTTImpl::Config::NEMO_CTC;
        } else {
            implConfig.model_type = OnnxSTTImpl::Config::CACHE_AWARE_CONFORMER;
        }
        
        impl_ = std::make_unique<OnnxSTTImpl>(implConfig);
    }
    
    bool initialize() override {
        return impl_->initialize();
    }
    
    TranscriptionResult processAudioChunk(const int16_t* samples, 
                                        size_t num_samples, 
                                        uint64_t timestamp_ms) override {
        auto implResult = impl_->processAudioChunk(samples, num_samples, timestamp_ms);
        
        TranscriptionResult result;
        result.text = implResult.text;
        result.is_final = implResult.is_final;
        result.confidence = implResult.confidence;
        result.timestamp_ms = implResult.timestamp_ms;
        result.latency_ms = implResult.latency_ms;
        
        return result;
    }
    
    void reset() override {
        impl_->reset();
    }
    
    Stats getStats() const override {
        auto implStats = impl_->getStats();
        
        Stats stats;
        stats.total_audio_ms = implStats.total_audio_ms;
        stats.total_processing_ms = implStats.total_processing_ms;
        stats.real_time_factor = implStats.real_time_factor;
        
        return stats;
    }
    
private:
    std::unique_ptr<OnnxSTTImpl> impl_;
};

// Factory function implementation
std::unique_ptr<OnnxSTTInterface> createOnnxSTT(const OnnxSTTInterface::Config& config) {
    return std::make_unique<OnnxSTTWrapper>(config);
}

} // namespace onnx_stt