#ifndef NEMO_CACHE_AWARE_CONFORMER_HPP
#define NEMO_CACHE_AWARE_CONFORMER_HPP

#include "ModelInterface.hpp"
#include "CacheManager.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace onnx_stt {

/**
 * @brief NVidia NeMo Cache-Aware Streaming Conformer implementation
 * 
 * Supports NeMo FastConformer-Hybrid models with cache-aware streaming.
 * These models are trained with limited right context and maintain
 * cache tensors (cache_last_channel, cache_last_time) between chunks.
 * 
 * Model Architecture:
 * - Input: audio features + cache tensors
 * - Output: encoded features + updated cache tensors
 * - Cache: last_channel_cache, last_time_cache
 * 
 * Supported Models:
 * - stt_en_fastconformer_hybrid_large_streaming_multi (114M params)
 * - Custom NeMo cache-aware models exported with cache_support=True
 */
class NeMoCacheAwareConformer : public ModelInterface {
public:
    struct NeMoConfig {
        std::string model_path;
        int num_threads = 4;
        int batch_size = 1;
        int feature_dim = 80;           // 80-dim log-mel features
        int chunk_frames = 160;         // Recommended: divisible by 4, gives 40 frames after subsampling
        int last_channel_cache_size = 64;  // Configurable cache size
        int last_time_cache_size = 64;
        int num_cache_layers = 12;      // Number of layers with caching
        int hidden_size = 512;          // Model hidden dimension
        
        // Attention context configuration (affects latency)
        // [70,0]: 0ms, [70,1]: 80ms, [70,16]: 480ms, [70,33]: 1040ms
        int att_context_size_left = 70;
        int att_context_size_right = 0;  // 0ms latency by default
        
        // Vocabulary file path for token decoding
        std::string vocab_path = "";
    };

    explicit NeMoCacheAwareConformer(const NeMoConfig& config);
    virtual ~NeMoCacheAwareConformer();

    // ModelInterface implementation
    bool initialize(const ModelConfig& config) override;
    ModelInterface::TranscriptionResult processChunk(const std::vector<std::vector<float>>& features,
                                                    uint64_t timestamp_ms) override;
    void reset() override;
    std::map<std::string, double> getStats() const override;
    bool supportsStreaming() const override { return true; }
    int getFeatureDim() const override { return config_.feature_dim; }
    int getChunkFrames() const override { return config_.chunk_frames; }
    const ModelConfig& getConfig() const override { return model_config_; }

private:
    NeMoConfig config_;
    ModelConfig model_config_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    // Cache management
    std::unique_ptr<CacheManager> cache_manager_;
    
    // Model input/output names
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    
    // Cache tensors for streaming
    std::vector<float> cache_last_channel_;
    std::vector<float> cache_last_time_;
    bool cache_initialized_;
    
    // Statistics
    mutable uint64_t total_chunks_processed_;
    mutable uint64_t total_processing_time_ms_;
    mutable uint64_t cache_updates_;
    
    // Vocabulary for token decoding
    std::vector<std::string> vocabulary_;
    bool vocab_loaded_;
    
    // Private methods
    bool initializeONNXSession();
    bool initializeCacheTensors();
    bool loadVocabulary(const std::string& vocab_path);
    std::vector<Ort::Value> prepareCacheInputs();
    void updateCacheFromOutputs(std::vector<Ort::Value>& outputs);
    std::string decodeTokens(const float* logits, size_t logits_size);
    std::string decodeCTCTokens(const float* log_probs, int64_t seq_len, int64_t num_classes);
    void updateStats(uint64_t processing_time_ms) const;
};

} // namespace onnx_stt

#endif // NEMO_CACHE_AWARE_CONFORMER_HPP