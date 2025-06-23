#include "STTPipeline.hpp"
#include "SileroVAD.hpp"
#include "KaldifeatExtractor.hpp"
#include "ZipformerModel.hpp"
#include "NeMoCacheAwareConformer.hpp"
#include <iostream>
#include <chrono>
#include <numeric>

namespace onnx_stt {

STTPipeline::STTPipeline(const Config& config) 
    : config_(config), last_speech_time_ms_(0), in_speech_segment_(false) {}

bool STTPipeline::initialize() {
    try {
        // Initialize VAD if enabled
        if (config_.enable_vad && !initializeVAD()) {
            std::cerr << "Failed to initialize VAD" << std::endl;
            return false;
        }
        
        // Initialize feature extractor
        if (!initializeFeatureExtractor()) {
            std::cerr << "Failed to initialize feature extractor" << std::endl;
            return false;
        }
        
        // Initialize ASR model
        if (!initializeModel()) {
            std::cerr << "Failed to initialize ASR model" << std::endl;
            return false;
        }
        
        std::cout << "STTPipeline initialized successfully" << std::endl;
        std::cout << "  VAD: " << (config_.enable_vad ? "enabled" : "disabled") << std::endl;
        std::cout << "  Feature extractor: " << (config_.feature_type == Config::KALDIFEAT ? "kaldifeat" : "simple_fbank") << std::endl;
        std::cout << "  Model: " << config_.model_config.model_type << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "STTPipeline initialization failed: " << e.what() << std::endl;
        return false;
    }
}

STTPipeline::Result STTPipeline::processAudio(const int16_t* samples, size_t num_samples, uint64_t timestamp_ms) {
    auto audio_float = convertInt16ToFloat(samples, num_samples);
    return processAudio(audio_float, timestamp_ms);
}

STTPipeline::Result STTPipeline::processAudio(const std::vector<float>& audio, uint64_t timestamp_ms) {
    return processAudioInternal(audio, timestamp_ms);
}

STTPipeline::Result STTPipeline::processAudioInternal(const std::vector<float>& audio, uint64_t timestamp_ms) {
    auto start_time = std::chrono::steady_clock::now();
    
    Result result;
    result.timestamp_ms = timestamp_ms;
    result.is_final = false;
    result.confidence = 0.0;
    
    // Step 1: Voice Activity Detection
    auto vad_start = std::chrono::steady_clock::now();
    
    if (config_.enable_vad && vad_) {
        auto vad_result = vad_->processChunk(audio, timestamp_ms);
        result.speech_detected = vad_result.is_speech;
        result.vad_confidence = vad_result.confidence;
        
        auto vad_end = std::chrono::steady_clock::now();
        result.vad_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            vad_end - vad_start).count();
        
        // Update speech segment tracking
        if (vad_result.is_speech) {
            last_speech_time_ms_ = timestamp_ms;
            in_speech_segment_ = true;
        } else {
            // Check if we've been in silence long enough to finalize
            if (in_speech_segment_ && 
                timestamp_ms - last_speech_time_ms_ > config_.silence_threshold_sec * 1000) {
                result.is_final = true;
                in_speech_segment_ = false;
            }
        }
        
        // Skip processing if no speech detected
        if (!vad_result.is_speech && !in_speech_segment_) {
            stats_.silence_chunks++;
            stats_.total_chunks_processed++;
            
            auto end_time = std::chrono::steady_clock::now();
            result.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            updateStats(result);
            return result;
        }
        
        stats_.speech_chunks++;
    } else {
        result.speech_detected = true;
        result.vad_confidence = 1.0f;
        result.vad_latency_ms = 0;
        stats_.speech_chunks++;
    }
    
    // Step 2: Feature Extraction
    auto feature_start = std::chrono::steady_clock::now();
    
    auto features = feature_extractor_->computeFeatures(audio);
    
    auto feature_end = std::chrono::steady_clock::now();
    result.feature_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        feature_end - feature_start).count();
    
    // Step 3: ASR Model Processing
    auto model_start = std::chrono::steady_clock::now();
    
    auto model_result = model_->processChunk(features, timestamp_ms);
    
    auto model_end = std::chrono::steady_clock::now();
    result.model_latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        model_end - model_start).count();
    
    // Combine results
    result.text = model_result.text;
    result.confidence = model_result.confidence;
    
    // Override finality if model says it's final or if VAD detected end of speech
    if (model_result.is_final) {
        result.is_final = true;
    }
    
    // Calculate total latency
    auto end_time = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Update statistics
    stats_.total_chunks_processed++;
    updateStats(result);
    
    return result;
}

void STTPipeline::reset() {
    if (vad_) {
        vad_->reset();
    }
    if (model_) {
        model_->reset();
    }
    
    audio_buffer_.clear();
    last_speech_time_ms_ = 0;
    in_speech_segment_ = false;
    
    // Reset statistics
    stats_ = Stats();
}

STTPipeline::Stats STTPipeline::getStats() const {
    // Update real-time factor
    if (stats_.total_chunks_processed > 0) {
        // Assuming 100ms chunks (configurable)
        double total_audio_ms = stats_.total_chunks_processed * 100.0;
        double total_processing_ms = stats_.avg_total_latency_ms * stats_.total_chunks_processed;
        stats_.real_time_factor = total_processing_ms / total_audio_ms;
    }
    
    // Get component stats
    if (vad_) {
        // VAD doesn't have a getStats method in our interface, so we'll use defaults
        stats_.vad_stats["enabled"] = config_.enable_vad ? 1.0 : 0.0;
    }
    
    if (model_) {
        stats_.model_stats = model_->getStats();
    }
    
    return stats_;
}

bool STTPipeline::initializeVAD() {
    if (!config_.enable_vad) {
        return true;
    }
    
    // Try Silero VAD first, fall back to energy VAD
    vad_ = createSileroVAD(config_.vad_config);
    if (!vad_) {
        std::cout << "Silero VAD not available, using energy-based VAD" << std::endl;
        vad_ = createEnergyVAD(config_.vad_config);
    }
    
    return vad_ != nullptr;
}

bool STTPipeline::initializeFeatureExtractor() {
    switch (config_.feature_type) {
        case Config::KALDIFEAT:
            feature_extractor_ = createKaldifeat(config_.feature_config);
            if (!feature_extractor_) {
                std::cout << "Info: Kaldifeat not available, using ImprovedFbank" << std::endl;
                feature_extractor_ = createImprovedFbank(config_.feature_config);
            }
            break;
        case Config::IMPROVED_FBANK:
            feature_extractor_ = createImprovedFbank(config_.feature_config);
            break;
        default:
            return false;
    }
    
    return feature_extractor_ != nullptr;
}

bool STTPipeline::initializeModel() {
    model_ = createModel(config_.model_config);
    return model_ != nullptr;
}

void STTPipeline::updateStats(const Result& result) {
    // Update latency averages
    double n = static_cast<double>(stats_.total_chunks_processed);
    
    stats_.avg_vad_latency_ms = (stats_.avg_vad_latency_ms * (n - 1) + result.vad_latency_ms) / n;
    stats_.avg_feature_latency_ms = (stats_.avg_feature_latency_ms * (n - 1) + result.feature_latency_ms) / n;
    stats_.avg_model_latency_ms = (stats_.avg_model_latency_ms * (n - 1) + result.model_latency_ms) / n;
    stats_.avg_total_latency_ms = (stats_.avg_total_latency_ms * (n - 1) + result.latency_ms) / n;
}

std::vector<float> STTPipeline::convertInt16ToFloat(const int16_t* samples, size_t num_samples) {
    std::vector<float> result(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        result[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    return result;
}

// Factory functions
std::unique_ptr<STTPipeline> createZipformerPipeline(const std::string& model_dir, bool enable_vad) {
    STTPipeline::Config config;
    
    // VAD configuration
    config.enable_vad = enable_vad;
    config.vad_config.sample_rate = 16000;
    config.vad_config.speech_threshold = 0.5f;
    
    // Feature configuration
    config.feature_type = STTPipeline::Config::KALDIFEAT;
    config.feature_config.sample_rate = 16000;
    config.feature_config.num_mel_bins = 80;
    config.feature_config.frame_length_ms = 25;
    config.feature_config.frame_shift_ms = 10;
    
    // Model configuration
    config.model_config.model_type = ModelInterface::ModelConfig::ZIPFORMER_RNNT;
    config.model_config.encoder_path = model_dir + "/encoder-epoch-99-avg-1.onnx";
    config.model_config.decoder_path = model_dir + "/decoder-epoch-99-avg-1.onnx";
    config.model_config.joiner_path = model_dir + "/joiner-epoch-99-avg-1.onnx";
    config.model_config.vocab_path = model_dir + "/tokens.txt";
    
    config.model_config.sample_rate = 16000;
    config.model_config.chunk_frames = 32;
    config.model_config.feature_dim = 80;
    config.model_config.beam_size = 10;
    config.model_config.blank_id = 0;
    
    // Cache configuration for Zipformer
    config.model_config.cache_config = CacheManager::createZipformerConfig(5);
    
    auto pipeline = std::make_unique<STTPipeline>(config);
    if (pipeline->initialize()) {
        return pipeline;
    }
    
    return nullptr;
}

std::unique_ptr<STTPipeline> createConformerPipeline(const std::string& /* model_dir */, bool /* enable_vad */) {
    // Placeholder - would configure for Conformer models
    std::cerr << "ConformerPipeline not yet implemented" << std::endl;
    return nullptr;
}

std::unique_ptr<STTPipeline> createWenetPipeline(const std::string& /* model_dir */, bool /* enable_vad */) {
    // Placeholder - would configure for WeNet models
    std::cerr << "WenetPipeline not yet implemented" << std::endl;
    return nullptr;
}

std::unique_ptr<STTPipeline> createNeMoPipeline(const std::string& model_path, bool enable_vad) {
    // Create pipeline configuration for NeMo cache-aware Conformer
    STTPipeline::Config config;
    
    // VAD configuration
    config.enable_vad = enable_vad;
    config.vad_config.sample_rate = 16000;
    config.vad_config.window_size_ms = 64;
    config.vad_config.speech_threshold = 0.5f;
    
    // Feature extraction configuration
    config.feature_type = STTPipeline::Config::KALDIFEAT;
    config.feature_config.sample_rate = 16000;
    config.feature_config.num_mel_bins = 80;
    config.feature_config.frame_length_ms = 25;
    config.feature_config.frame_shift_ms = 10;
    
    // Model configuration for NeMo (single .onnx file)
    config.model_config.encoder_path = model_path;  // NeMo uses single model file
    config.model_config.model_type = ModelInterface::ModelConfig::NVIDIA_NEMO;
    config.model_config.chunk_frames = 160;  // 160 frames = 1.6 seconds, divisible by 4
    config.model_config.feature_dim = 80;
    config.model_config.num_threads = 4;
    
    // Set vocab path - assume it's in the same directory as the model
    std::string model_dir = model_path.substr(0, model_path.find_last_of("/\\"));
    config.model_config.vocab_path = model_dir + "/tokenizer.txt";
    
    // Pipeline settings
    config.sample_rate = 16000;
    config.enable_partial_results = true;
    config.silence_threshold_sec = 0.5f;
    config.enable_profiling = false;
    
    try {
        auto pipeline = std::make_unique<STTPipeline>(config);
        
        if (!pipeline->initialize()) {
            std::cerr << "Failed to initialize NeMo pipeline" << std::endl;
            return nullptr;
        }
        
        std::cout << "NeMo pipeline created successfully" << std::endl;
        return pipeline;
        
    } catch (const std::exception& e) {
        std::cerr << "Error creating NeMo pipeline: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace onnx_stt