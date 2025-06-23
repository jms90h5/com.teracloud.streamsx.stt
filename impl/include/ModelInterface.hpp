#ifndef MODEL_INTERFACE_HPP
#define MODEL_INTERFACE_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "CacheManager.hpp"

namespace onnx_stt {

/**
 * Abstract interface for different ASR model architectures
 * Supports Zipformer, Conformer, WeNet, SpeechBrain, etc.
 */
class ModelInterface {
public:
    struct ModelConfig {
        // Model paths
        std::string encoder_path;
        std::string decoder_path;
        std::string joiner_path;
        std::string vocab_path;
        
        // Model parameters
        int sample_rate = 16000;
        int chunk_frames = 32;
        int feature_dim = 80;
        int vocab_size = 0;
        
        // Decoding parameters
        int beam_size = 10;
        int blank_id = 0;
        float blank_penalty = 0.0f;
        
        // Performance settings
        int num_threads = 4;
        bool use_gpu = false;
        std::string provider = "cpu";  // cpu, cuda, tensorrt
        
        // Cache configuration
        CacheManager::CacheConfig cache_config;
        
        // Model type identification
        enum ModelType {
            ZIPFORMER_RNNT,
            CONFORMER_RNNT,
            WENET_CONFORMER,
            SPEECHBRAIN_CRDNN,
            NVIDIA_NEMO,
            CUSTOM
        } model_type = ZIPFORMER_RNNT;
    };
    
    struct TranscriptionResult {
        std::string text;
        bool is_final;
        double confidence;
        uint64_t timestamp_ms;
        uint64_t latency_ms;
        std::vector<float> token_probs;  // Optional: token-level probabilities
    };
    
    virtual ~ModelInterface() = default;
    
    // Initialize model with configuration
    virtual bool initialize(const ModelConfig& config) = 0;
    
    // Process audio features and return transcription
    virtual TranscriptionResult processChunk(const std::vector<std::vector<float>>& features, 
                                            uint64_t timestamp_ms) = 0;
    
    // Reset model state (caches, beam search, etc.)
    virtual void reset() = 0;
    
    // Get model configuration
    virtual const ModelConfig& getConfig() const = 0;
    
    // Get model statistics
    virtual std::map<std::string, double> getStats() const = 0;
    
    // Check if model supports streaming
    virtual bool supportsStreaming() const = 0;
    
    // Get expected feature dimension
    virtual int getFeatureDim() const = 0;
    
    // Get expected chunk size in frames
    virtual int getChunkFrames() const = 0;
};

/**
 * Factory functions for different model types
 */
std::unique_ptr<ModelInterface> createZipformerModel(const ModelInterface::ModelConfig& config);
std::unique_ptr<ModelInterface> createConformerModel(const ModelInterface::ModelConfig& config);
std::unique_ptr<ModelInterface> createWenetModel(const ModelInterface::ModelConfig& config);
std::unique_ptr<ModelInterface> createSpeechBrainModel(const ModelInterface::ModelConfig& config);
std::unique_ptr<ModelInterface> createNeMoModel(const ModelInterface::ModelConfig& config);

// Unified model factory function
std::unique_ptr<ModelInterface> createModel(const ModelInterface::ModelConfig& config);

/**
 * Model factory - automatically detects model type and creates appropriate instance
 */
std::unique_ptr<ModelInterface> createModel(const ModelInterface::ModelConfig& config);

} // namespace onnx_stt

#endif // MODEL_INTERFACE_HPP