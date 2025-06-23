#ifndef STT_PIPELINE_HPP
#define STT_PIPELINE_HPP

#include "VADInterface.hpp"
#include "FeatureExtractor.hpp"
#include "ModelInterface.hpp"
#include <memory>
#include <vector>
#include <chrono>

namespace onnx_stt {

/**
 * Complete Speech-to-Text pipeline integrating VAD, feature extraction, and ASR model
 * 
 * Pipeline flow:
 * 1. Audio input → VAD (Voice Activity Detection)
 * 2. If speech detected → Feature extraction (filterbank/kaldifeat)
 * 3. Features → ASR model (Zipformer/Conformer/etc.)
 * 4. Model output → Text transcription
 */
class STTPipeline {
public:
    struct Config {
        // VAD configuration
        VADInterface::Config vad_config;
        bool enable_vad = true;
        
        // Feature extraction configuration
        FeatureExtractor::Config feature_config;
        enum FeatureType {
            IMPROVED_FBANK,  // Real feature extraction using ImprovedFbank
            KALDIFEAT        // Real feature extraction using Kaldi
        } feature_type = KALDIFEAT;
        
        // Model configuration
        ModelInterface::ModelConfig model_config;
        
        // Pipeline settings
        int sample_rate = 16000;
        bool enable_partial_results = true;
        float silence_threshold_sec = 0.5f;  // Seconds of silence before finalizing
        
        // Performance settings
        bool enable_profiling = false;
    };
    
    struct Result {
        std::string text;
        bool is_final;
        double confidence;
        uint64_t timestamp_ms;
        uint64_t latency_ms;
        
        // Optional: detailed component timing
        uint64_t vad_latency_ms = 0;
        uint64_t feature_latency_ms = 0;
        uint64_t model_latency_ms = 0;
        
        // VAD information
        bool speech_detected = true;
        float vad_confidence = 1.0f;
    };
    
    struct Stats {
        uint64_t total_chunks_processed = 0;
        uint64_t speech_chunks = 0;
        uint64_t silence_chunks = 0;
        
        double avg_vad_latency_ms = 0.0;
        double avg_feature_latency_ms = 0.0;
        double avg_model_latency_ms = 0.0;
        double avg_total_latency_ms = 0.0;
        
        double real_time_factor = 0.0;
        
        // Component statistics
        std::map<std::string, double> vad_stats;
        std::map<std::string, double> model_stats;
    };
    
    explicit STTPipeline(const Config& config);
    ~STTPipeline() = default;
    
    // Initialize all components
    bool initialize();
    
    // Process audio chunk (int16 format)
    Result processAudio(const int16_t* samples, size_t num_samples, uint64_t timestamp_ms);
    
    // Process audio chunk (float format)
    Result processAudio(const std::vector<float>& audio, uint64_t timestamp_ms);
    
    // Reset all components
    void reset();
    
    // Get pipeline configuration
    const Config& getConfig() const { return config_; }
    
    // Get pipeline statistics
    Stats getStats() const;
    
    // Enable/disable components
    void enableVAD(bool enable) { config_.enable_vad = enable; }
    void enablePartialResults(bool enable) { config_.enable_partial_results = enable; }
    
private:
    Config config_;
    
    // Pipeline components
    std::unique_ptr<VADInterface> vad_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<ModelInterface> model_;
    
    // State management
    std::vector<float> audio_buffer_;
    uint64_t last_speech_time_ms_;
    bool in_speech_segment_;
    
    // Performance tracking
    mutable Stats stats_;
    std::chrono::steady_clock::time_point last_process_time_;
    
    // Helper methods
    bool initializeVAD();
    bool initializeFeatureExtractor();
    bool initializeModel();
    
    Result processAudioInternal(const std::vector<float>& audio, uint64_t timestamp_ms);
    void updateStats(const Result& result);
    
    std::vector<float> convertInt16ToFloat(const int16_t* samples, size_t num_samples);
};

/**
 * Pipeline factory functions for common configurations
 */
std::unique_ptr<STTPipeline> createZipformerPipeline(const std::string& model_dir, bool enable_vad = true);
std::unique_ptr<STTPipeline> createConformerPipeline(const std::string& model_dir, bool enable_vad = true);
std::unique_ptr<STTPipeline> createWenetPipeline(const std::string& model_dir, bool enable_vad = true);
std::unique_ptr<STTPipeline> createNeMoPipeline(const std::string& model_path, bool enable_vad = true);

} // namespace onnx_stt

#endif // STT_PIPELINE_HPP