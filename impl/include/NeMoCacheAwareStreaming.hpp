#pragma once

#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "ModelInterface.hpp"
#include "FeatureExtractor.hpp"

namespace onnx_stt {

/**
 * @brief NVIDIA NeMo Cache-Aware Streaming FastConformer implementation
 * 
 * This class implements streaming speech recognition using NVIDIA's cache-aware
 * FastConformer model with support for multiple latency settings and hybrid
 * CTC+RNN-T decoders.
 * 
 * Model: nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
 * Architecture: 17-layer FastConformer encoder (512 d_model) + Hybrid decoders
 * Features: Cache-aware streaming, chunked attention, multi-latency support
 */
class NeMoCacheAwareStreaming : public ModelInterface {
public:
    /**
     * @brief Streaming latency configurations
     */
    enum class LatencyMode {
        ULTRA_LOW = 0,    // 0ms latency (fully causal)
        VERY_LOW = 1,     // 80ms latency 
        LOW = 16,         // 480ms latency
        MEDIUM = 33       // 1040ms latency
    };
    
    /**
     * @brief Decoder type selection
     */
    enum class DecoderType {
        CTC,        // Fast CTC decoder for streaming
        RNNT,       // RNN-T decoder for better accuracy
        HYBRID      // Use both decoders with confidence scoring
    };

private:
    // Model configuration
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int N_MELS = 80;
    static constexpr int D_MODEL = 512;
    static constexpr int N_LAYERS = 17;
    static constexpr int VOCAB_SIZE = 1024;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> ctc_decoder_session_;
    std::unique_ptr<Ort::Session> rnnt_decoder_session_;
    
    // Model state and caching
    LatencyMode latency_mode_;
    DecoderType decoder_type_;
    
    // Cache management for streaming
    struct StreamingCache {
        std::vector<std::vector<float>> encoder_states;  // Layer-wise hidden states
        std::vector<std::vector<float>> attention_cache; // Attention cache for each layer
        std::vector<float> prediction_cache;             // RNN-T prediction network cache
        int processed_frames;                            // Number of processed audio frames
        int context_size;                                // Current context size based on latency
    };
    
    std::unique_ptr<StreamingCache> cache_;
    
    // Feature extraction
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    
    // Vocabulary and tokenization
    std::vector<std::string> vocabulary_;
    std::string unk_token_ = "<unk>";
    std::string blank_token_ = "<blank>";
    
    // Streaming configuration
    int chunk_size_;           // Audio chunk size based on latency mode
    int context_frames_;       // Look-ahead frames for attention
    float confidence_threshold_ = 0.3f;
    
    // Model configuration
    ModelConfig config_;
    
public:
    /**
     * @brief Constructor
     * @param model_dir Directory containing ONNX model files
     * @param latency_mode Desired latency configuration
     * @param decoder_type Decoder type to use
     */
    NeMoCacheAwareStreaming(const std::string& model_dir, 
                           LatencyMode latency_mode = LatencyMode::LOW,
                           DecoderType decoder_type = DecoderType::CTC);
    
    /**
     * @brief Destructor
     */
    ~NeMoCacheAwareStreaming() override = default;
    
    /**
     * @brief Initialize the model with configuration
     */
    bool initialize(const ModelConfig& config) override;
    
    /**
     * @brief Process audio features
     */
    TranscriptionResult processChunk(const std::vector<std::vector<float>>& features, 
                                   uint64_t timestamp_ms) override;
    
    /**
     * @brief Reset streaming state and cache
     */
    void reset() override;
    
    /**
     * @brief Get model configuration
     */
    const ModelConfig& getConfig() const override;
    
    /**
     * @brief Get model statistics
     */
    std::map<std::string, double> getStats() const override;
    
    /**
     * @brief Check if model supports streaming
     */
    bool supportsStreaming() const override { return true; }
    
    /**
     * @brief Get expected feature dimension
     */
    int getFeatureDim() const override { return N_MELS; }
    
    /**
     * @brief Get expected chunk size in frames
     */
    int getChunkFrames() const override;
    
    // Legacy compatibility methods
    bool initialize();
    std::string processAudioChunk(const float* audio_chunk, 
                                 int chunk_size, 
                                 bool is_final = false);
    std::string getModelInfo() const;
    
    /**
     * @brief Set latency mode (affects chunk size and context)
     * @param mode New latency mode
     */
    void setLatencyMode(LatencyMode mode);
    
    /**
     * @brief Set decoder type
     * @param type Decoder type to use
     */
    void setDecoderType(DecoderType type);
    
    /**
     * @brief Get current streaming statistics
     */
    struct StreamingStats {
        int total_frames_processed;
        float average_processing_time_ms;
        float current_latency_ms;
        int cache_size_mb;
    };
    
    StreamingStats getStreamingStats() const;

private:
    /**
     * @brief Load ONNX model sessions
     */
    bool loadONNXModels(const std::string& model_dir);
    
    /**
     * @brief Load vocabulary from file
     */
    bool loadVocabulary(const std::string& vocab_file);
    
    /**
     * @brief Configure streaming parameters based on latency mode
     */
    void configureLatencyMode();
    
    /**
     * @brief Extract mel-spectrogram features from audio
     */
    std::vector<std::vector<float>> extractFeatures(const float* audio, int length);
    
    /**
     * @brief Run encoder with cache management
     */
    std::vector<std::vector<float>> runEncoder(const std::vector<std::vector<float>>& features);
    
    /**
     * @brief Run CTC decoder
     */
    std::string runCTCDecoder(const std::vector<std::vector<float>>& encoder_output);
    
    /**
     * @brief Run RNN-T decoder with prediction network caching
     */
    std::string runRNNTDecoder(const std::vector<std::vector<float>>& encoder_output);
    
    /**
     * @brief Decode token IDs to text using vocabulary
     */
    std::string decodeTokens(const std::vector<int>& token_ids);
    
    /**
     * @brief Update encoder cache with new states
     */
    void updateEncoderCache(const std::vector<std::vector<float>>& new_states);
    
    /**
     * @brief Get appropriate context size for current latency mode
     */
    int getContextSize() const;
    
    /**
     * @brief Apply cache-aware attention mechanism
     */
    std::vector<std::vector<float>> applyCacheAwareAttention(
        const std::vector<std::vector<float>>& inputs,
        int layer_idx);
};

/**
 * @brief Factory function to create NeMo cache-aware streaming model
 */
std::unique_ptr<NeMoCacheAwareStreaming> createNeMoCacheAwareModel(
    const std::string& model_dir,
    NeMoCacheAwareStreaming::LatencyMode latency = NeMoCacheAwareStreaming::LatencyMode::LOW,
    NeMoCacheAwareStreaming::DecoderType decoder = NeMoCacheAwareStreaming::DecoderType::CTC
);

} // namespace onnx_stt