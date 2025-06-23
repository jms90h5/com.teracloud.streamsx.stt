#ifndef ZIPFORMER_MODEL_HPP
#define ZIPFORMER_MODEL_HPP

#include "ModelInterface.hpp"
#include "ZipformerRNNT.hpp"
#include "CacheManager.hpp"
#include <memory>

namespace onnx_stt {

/**
 * Zipformer RNN-T model implementation
 * Wrapper around the existing ZipformerRNNT class to implement ModelInterface
 */
class ZipformerModel : public ModelInterface {
public:
    explicit ZipformerModel(const ModelConfig& config);
    ~ZipformerModel() override = default;
    
    bool initialize(const ModelConfig& config) override;
    
    TranscriptionResult processChunk(const std::vector<std::vector<float>>& features, 
                                   uint64_t timestamp_ms) override;
    
    void reset() override;
    
    const ModelConfig& getConfig() const override { return config_; }
    
    std::map<std::string, double> getStats() const override;
    
    bool supportsStreaming() const override { return true; }
    
    int getFeatureDim() const override { return config_.feature_dim; }
    
    int getChunkFrames() const override { return config_.chunk_frames; }
    
private:
    ModelConfig config_;
    std::unique_ptr<ZipformerRNNT> zipformer_;
    std::unique_ptr<CacheManager> cache_manager_;
    
    // Performance tracking
    uint64_t total_chunks_processed_;
    uint64_t total_processing_time_ms_;
    std::chrono::steady_clock::time_point last_process_time_;
    
    // Helper methods
    ZipformerRNNT::Config convertToZipformerConfig(const ModelConfig& config);
    TranscriptionResult convertFromZipformerResult(const ZipformerRNNT::Result& result, 
                                                  uint64_t timestamp_ms, 
                                                  uint64_t latency_ms);
};

/**
 * Generic ONNX model base class for other model types
 * Provides common ONNX Runtime functionality
 */
class GenericOnnxModel : public ModelInterface {
public:
    explicit GenericOnnxModel(const ModelConfig& config);
    ~GenericOnnxModel() override = default;
    
    bool initialize(const ModelConfig& config) override;
    
    TranscriptionResult processChunk(const std::vector<std::vector<float>>& features, 
                                   uint64_t timestamp_ms) override;
    
    void reset() override;
    
    const ModelConfig& getConfig() const override { return config_; }
    
    std::map<std::string, double> getStats() const override;
    
    bool supportsStreaming() const override { return true; }
    
    int getFeatureDim() const override { return config_.feature_dim; }
    
    int getChunkFrames() const override { return config_.chunk_frames; }
    
protected:
    ModelConfig config_;
    std::unique_ptr<CacheManager> cache_manager_;
    
    // ONNX Runtime components (to be implemented by subclasses)
    virtual bool loadModels() = 0;
    virtual TranscriptionResult runInference(const std::vector<std::vector<float>>& features, 
                                            uint64_t timestamp_ms) = 0;
    
    // Vocabulary management
    std::vector<std::string> vocabulary_;
    bool loadVocabulary(const std::string& vocab_path);
    std::string tokensToText(const std::vector<int>& tokens);
};

} // namespace onnx_stt

#endif // ZIPFORMER_MODEL_HPP