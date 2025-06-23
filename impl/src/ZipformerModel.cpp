#include "ZipformerModel.hpp"
#include <chrono>
#include <iostream>
#include <fstream>

namespace onnx_stt {

// ZipformerModel Implementation
ZipformerModel::ZipformerModel(const ModelConfig& config) 
    : config_(config), total_chunks_processed_(0), total_processing_time_ms_(0) {}

bool ZipformerModel::initialize(const ModelConfig& config) {
    config_ = config;
    
    try {
        // Initialize cache manager for Zipformer
        cache_manager_ = std::make_unique<CacheManager>(config_.cache_config);
        if (!cache_manager_->initialize()) {
            std::cerr << "Failed to initialize cache manager" << std::endl;
            return false;
        }
        
        // Convert config and create Zipformer
        auto zipformer_config = convertToZipformerConfig(config_);
        zipformer_ = std::make_unique<ZipformerRNNT>(zipformer_config);
        
        if (!zipformer_->initialize()) {
            std::cerr << "Failed to initialize Zipformer model" << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "ZipformerModel initialization failed: " << e.what() << std::endl;
        return false;
    }
}

ModelInterface::TranscriptionResult ZipformerModel::processChunk(
    const std::vector<std::vector<float>>& features, uint64_t timestamp_ms) {
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Convert features to format expected by ZipformerRNNT
    // For now, we'll flatten the features since ZipformerRNNT expects a vector<float>
    std::vector<float> flattened_features;
    for (const auto& frame : features) {
        flattened_features.insert(flattened_features.end(), frame.begin(), frame.end());
    }
    
    // Process with ZipformerRNNT
    auto zipformer_result = zipformer_->processChunk(flattened_features);
    
    auto end_time = std::chrono::steady_clock::now();
    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    // Update statistics
    total_chunks_processed_++;
    total_processing_time_ms_ += latency_ms;
    last_process_time_ = end_time;
    
    // Convert result
    return convertFromZipformerResult(zipformer_result, timestamp_ms, latency_ms);
}

void ZipformerModel::reset() {
    if (zipformer_) {
        zipformer_->reset();
    }
    if (cache_manager_) {
        cache_manager_->reset();
    }
    total_chunks_processed_ = 0;
    total_processing_time_ms_ = 0;
}

std::map<std::string, double> ZipformerModel::getStats() const {
    std::map<std::string, double> stats;
    
    stats["total_chunks_processed"] = static_cast<double>(total_chunks_processed_);
    stats["total_processing_time_ms"] = static_cast<double>(total_processing_time_ms_);
    
    if (total_chunks_processed_ > 0) {
        stats["avg_latency_ms"] = static_cast<double>(total_processing_time_ms_) / 
                                 static_cast<double>(total_chunks_processed_);
    } else {
        stats["avg_latency_ms"] = 0.0;
    }
    
    // Real-time factor (assuming 390ms chunks)
    double total_audio_ms = total_chunks_processed_ * 390.0;
    if (total_audio_ms > 0) {
        stats["real_time_factor"] = static_cast<double>(total_processing_time_ms_) / total_audio_ms;
    } else {
        stats["real_time_factor"] = 0.0;
    }
    
    return stats;
}

ZipformerRNNT::Config ZipformerModel::convertToZipformerConfig(const ModelConfig& config) {
    ZipformerRNNT::Config zipformer_config;
    
    zipformer_config.encoder_onnx_path = config.encoder_path;
    zipformer_config.decoder_onnx_path = config.decoder_path;
    zipformer_config.joiner_onnx_path = config.joiner_path;
    zipformer_config.vocab_path = config.vocab_path;
    
    zipformer_config.sample_rate = config.sample_rate;
    zipformer_config.chunk_frames = config.chunk_frames;
    zipformer_config.feature_dim = config.feature_dim;
    zipformer_config.beam_size = config.beam_size;
    zipformer_config.blank_id = config.blank_id;
    zipformer_config.blank_penalty = config.blank_penalty;
    
    zipformer_config.num_threads = config.num_threads;
    zipformer_config.use_gpu = config.use_gpu;
    zipformer_config.provider = config.provider;
    
    return zipformer_config;
}

ModelInterface::TranscriptionResult ZipformerModel::convertFromZipformerResult(
    const ZipformerRNNT::Result& result, uint64_t timestamp_ms, uint64_t latency_ms) {
    
    TranscriptionResult converted_result;
    converted_result.text = result.text;
    converted_result.is_final = result.is_final;
    converted_result.confidence = result.confidence;
    converted_result.timestamp_ms = timestamp_ms;
    converted_result.latency_ms = latency_ms;
    
    // Copy token probabilities if available
    if (!result.token_probs.empty()) {
        converted_result.token_probs = result.token_probs;
    }
    
    return converted_result;
}

// GenericOnnxModel Implementation
GenericOnnxModel::GenericOnnxModel(const ModelConfig& config) : config_(config) {}

bool GenericOnnxModel::initialize(const ModelConfig& config) {
    config_ = config;
    
    // Initialize cache manager
    cache_manager_ = std::make_unique<CacheManager>(config_.cache_config);
    if (!cache_manager_->initialize()) {
        std::cerr << "Failed to initialize cache manager" << std::endl;
        return false;
    }
    
    // Load vocabulary
    if (!config_.vocab_path.empty()) {
        if (!loadVocabulary(config_.vocab_path)) {
            std::cerr << "Failed to load vocabulary from " << config_.vocab_path << std::endl;
            return false;
        }
    }
    
    // Load models (implemented by subclasses)
    return loadModels();
}

ModelInterface::TranscriptionResult GenericOnnxModel::processChunk(
    const std::vector<std::vector<float>>& features, uint64_t timestamp_ms) {
    
    // Implemented by subclasses
    return runInference(features, timestamp_ms);
}

void GenericOnnxModel::reset() {
    if (cache_manager_) {
        cache_manager_->reset();
    }
}

std::map<std::string, double> GenericOnnxModel::getStats() const {
    // Basic stats - can be extended by subclasses
    std::map<std::string, double> stats;
    stats["vocabulary_size"] = static_cast<double>(vocabulary_.size());
    stats["cache_tensors"] = static_cast<double>(cache_manager_ ? cache_manager_->getNumCaches() : 0);
    return stats;
}

bool GenericOnnxModel::loadVocabulary(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return false;
    }
    
    vocabulary_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocabulary_.push_back(line);
        }
    }
    
    return !vocabulary_.empty();
}

std::string GenericOnnxModel::tokensToText(const std::vector<int>& tokens) {
    std::string text;
    
    for (int token_id : tokens) {
        if (token_id >= 0 && token_id < static_cast<int>(vocabulary_.size())) {
            if (!text.empty() && !vocabulary_[token_id].empty()) {
                text += " ";
            }
            text += vocabulary_[token_id];
        }
    }
    
    return text;
}

// Factory functions
std::unique_ptr<ModelInterface> createZipformerModel(const ModelInterface::ModelConfig& config) {
    auto model = std::make_unique<ZipformerModel>(config);
    if (model->initialize(config)) {
        return model;
    }
    return nullptr;
}

std::unique_ptr<ModelInterface> createConformerModel(const ModelInterface::ModelConfig& config) {
    // Placeholder - would implement ConformerModel class
    std::cerr << "ConformerModel not yet implemented" << std::endl;
    return nullptr;
}

std::unique_ptr<ModelInterface> createWenetModel(const ModelInterface::ModelConfig& config) {
    // Placeholder - would implement WenetModel class
    std::cerr << "WenetModel not yet implemented" << std::endl;
    return nullptr;
}

std::unique_ptr<ModelInterface> createSpeechBrainModel(const ModelInterface::ModelConfig& config) {
    // Placeholder - would implement SpeechBrainModel class
    std::cerr << "SpeechBrainModel not yet implemented" << std::endl;
    return nullptr;
}

std::unique_ptr<ModelInterface> createModel(const ModelInterface::ModelConfig& config) {
    switch (config.model_type) {
        case ModelInterface::ModelConfig::ZIPFORMER_RNNT:
            return createZipformerModel(config);
        case ModelInterface::ModelConfig::CONFORMER_RNNT:
            return createConformerModel(config);
        case ModelInterface::ModelConfig::WENET_CONFORMER:
            return createWenetModel(config);
        case ModelInterface::ModelConfig::SPEECHBRAIN_CRDNN:
            return createSpeechBrainModel(config);
        default:
            std::cerr << "Unknown model type: " << config.model_type << std::endl;
            return nullptr;
    }
}

} // namespace onnx_stt