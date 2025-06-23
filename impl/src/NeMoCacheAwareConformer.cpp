#include "../include/NeMoCacheAwareConformer.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace onnx_stt {

NeMoCacheAwareConformer::NeMoCacheAwareConformer(const NeMoConfig& config)
    : config_(config)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , cache_initialized_(false)
    , total_chunks_processed_(0)
    , total_processing_time_ms_(0)
    , cache_updates_(0)
    , vocab_loaded_(false) {
    
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeMoCacheAwareConformer");
    
    // Setup input/output names for NeMo CTC model (no cache)
    // Note: Large FastConformer model uses "processed_signal" instead of "audio_signal"
    input_names_ = {
        "processed_signal"    // Features input: [batch, time, features]
    };
    
    output_names_ = {
        "log_probs"           // CTC log probabilities: [batch, time, num_classes]
    };
}

NeMoCacheAwareConformer::~NeMoCacheAwareConformer() {
    // Cleanup handled by unique_ptr destructors
}

bool NeMoCacheAwareConformer::initialize(const ModelConfig& config) {
    model_config_ = config;
    
    std::cout << "Initializing NeMo Cache-Aware Conformer model..." << std::endl;
    std::cout << "Model path: " << config_.model_path << std::endl;
    std::cout << "Chunk frames: " << config_.chunk_frames << std::endl;
    std::cout << "Attention context: [" << config_.att_context_size_left 
              << "," << config_.att_context_size_right << "]" << std::endl;
    
    if (!initializeONNXSession()) {
        std::cerr << "Failed to initialize ONNX session" << std::endl;
        return false;
    }
    
    if (!initializeCacheTensors()) {
        std::cerr << "Failed to initialize cache tensors" << std::endl;
        return false;
    }
    
    // Load vocabulary if path provided
    if (!config_.vocab_path.empty()) {
        if (!loadVocabulary(config_.vocab_path)) {
            std::cerr << "Warning: Failed to load vocabulary from " << config_.vocab_path << std::endl;
            std::cerr << "Token decoding will output token IDs instead of text" << std::endl;
        }
    }
    
    std::cout << "NeMo Cache-Aware Conformer initialized successfully" << std::endl;
    return true;
}

bool NeMoCacheAwareConformer::initializeONNXSession() {
    try {
        // Check if model file exists
        std::ifstream model_file(config_.model_path);
        if (!model_file.good()) {
            std::cerr << "Model file not found: " << config_.model_path << std::endl;
            return false;
        }
        
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config_.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);
        
        // Verify model inputs/outputs
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();
        
        std::cout << "Model loaded: " << num_inputs << " inputs, " << num_outputs << " outputs" << std::endl;
        
        // Print input/output info for debugging
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "Input " << i << ": " << input_name.get() << " shape: [";
            for (size_t j = 0; j < input_shape.size(); ++j) {
                std::cout << input_shape[j];
                if (j < input_shape.size() - 1) std::cout << ",";
            }
            std::cout << "]" << std::endl;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing ONNX session: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCacheAwareConformer::initializeCacheTensors() {
    try {
        // Initialize cache_last_channel: [layers, batch, cache_size, hidden]
        size_t channel_cache_size = config_.num_cache_layers * config_.batch_size * 
                                   config_.last_channel_cache_size * config_.hidden_size;
        cache_last_channel_.resize(channel_cache_size, 0.0f);
        
        // Initialize cache_last_time: [layers, batch, hidden, cache_size]  
        size_t time_cache_size = config_.num_cache_layers * config_.batch_size *
                                config_.hidden_size * config_.last_time_cache_size;
        cache_last_time_.resize(time_cache_size, 0.0f);
        
        cache_initialized_ = true;
        
        std::cout << "Cache tensors initialized:" << std::endl;
        std::cout << "  Channel cache size: " << channel_cache_size << " elements" << std::endl;
        std::cout << "  Time cache size: " << time_cache_size << " elements" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing cache tensors: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCacheAwareConformer::loadVocabulary(const std::string& vocab_path) {
    try {
        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open()) {
            std::cerr << "Cannot open vocabulary file: " << vocab_path << std::endl;
            return false;
        }
        
        vocabulary_.clear();
        std::string line;
        
        while (std::getline(vocab_file, line)) {
            // Each line contains the token (tab/space separated from any other info)
            // Extract just the token part - token is BEFORE the tab, not after
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                vocabulary_.push_back(line.substr(0, tab_pos));  // Take text BEFORE tab
            } else {
                vocabulary_.push_back(line);
            }
        }
        
        vocab_file.close();
        vocab_loaded_ = true;
        
        std::cout << "Loaded vocabulary with " << vocabulary_.size() << " tokens from " << vocab_path << std::endl;
        
        // Print first few tokens for verification
        if (vocabulary_.size() >= 10) {
            std::cout << "First 10 tokens: ";
            for (size_t i = 0; i < 10; ++i) {
                std::cout << "[" << i << "]=" << vocabulary_[i] << " ";
            }
            std::cout << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading vocabulary: " << e.what() << std::endl;
        return false;
    }
}

ModelInterface::TranscriptionResult NeMoCacheAwareConformer::processChunk(const std::vector<std::vector<float>>& features,
                                                                     uint64_t timestamp_ms) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ModelInterface::TranscriptionResult result;
    result.timestamp_ms = timestamp_ms;
    result.is_final = true;
    result.confidence = 0.0f;
    
    try {
        if (!cache_initialized_) {
            throw std::runtime_error("Cache tensors not initialized");
        }
        
        if (features.empty() || features[0].empty()) {
            throw std::runtime_error("Empty features provided");
        }
        
        // Prepare input tensors
        
        // 1. Audio signal tensor: [batch, time, features] = [1, time_frames, 80]
        size_t batch_size = 1;
        size_t feature_dim = features[0].size();
        size_t time_frames = features.size();
        
        // Model has subsampling factor 4 (not 8), attention expects 125 frames
        // 125 frames after subsampling = 125 * 4 = 500 input frames needed  
        size_t required_frames = 500;  // Fixed size to match model's expected {125,8,64} shape
        
        std::cout << "Processing chunk: " << time_frames << " frames";
        if (time_frames != required_frames) {
            std::cout << " (padding to " << required_frames << " for model compatibility)";
        }
        std::cout << std::endl;
        
        // Pad or truncate features to exactly 160 frames
        std::vector<std::vector<float>> padded_features;
        padded_features.reserve(required_frames);
        
        for (size_t i = 0; i < required_frames; ++i) {
            if (i < time_frames) {
                // Use original feature
                padded_features.push_back(features[i]);
            } else {
                // Pad with zero vector
                padded_features.push_back(std::vector<float>(feature_dim, 0.0f));
            }
        }
        time_frames = required_frames;
        
        // Flatten features for ONNX (keep [time, features] format as [batch, time, features])
        std::vector<float> audio_signal_data;
        audio_signal_data.reserve(batch_size * time_frames * feature_dim);
        
        // Debug: Check feature statistics
        float min_feat = std::numeric_limits<float>::max();
        float max_feat = std::numeric_limits<float>::lowest();
        float sum_feat = 0.0f;
        
        for (size_t time = 0; time < time_frames; ++time) {
            for (size_t feat = 0; feat < feature_dim; ++feat) {
                float val = padded_features[time][feat];
                audio_signal_data.push_back(val);
                min_feat = std::min(min_feat, val);
                max_feat = std::max(max_feat, val);
                sum_feat += val;
            }
        }
        
        float avg_feat = sum_feat / (time_frames * feature_dim);
        std::cout << "Feature stats: min=" << min_feat << ", max=" << max_feat 
                  << ", avg=" << avg_feat << std::endl;
        
        std::vector<int64_t> audio_signal_shape = {static_cast<int64_t>(batch_size), 
                                                  static_cast<int64_t>(time_frames),
                                                  static_cast<int64_t>(feature_dim)};
        auto audio_signal_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, audio_signal_data.data(), audio_signal_data.size(),
            audio_signal_shape.data(), audio_signal_shape.size());
        
        // Prepare input vector (NeMo CTC model only needs audio_signal)
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(audio_signal_tensor));
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            output_names_.data(), output_names_.size());
        
        // Process outputs
        if (output_tensors.size() >= 1) {
            // Extract log_probs output: [batch, seq_len, num_classes]
            auto& log_probs_tensor = output_tensors[0];
            float* log_probs_data = log_probs_tensor.GetTensorMutableData<float>();
            auto log_probs_shape = log_probs_tensor.GetTensorTypeAndShapeInfo().GetShape();
            
            // Shape should be [1, seq_len//4, 128] due to subsampling in our model
            int64_t seq_len_out = log_probs_shape[1]; 
            int64_t num_classes = log_probs_shape[2];
            
            // Decode CTC output (simplified - argmax for now)
            result.text = decodeCTCTokens(log_probs_data, seq_len_out, num_classes);
            result.confidence = 0.85f;  // Placeholder confidence
            
        } else {
            throw std::runtime_error("No output tensors from NeMo model");
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.latency_ms = static_cast<uint64_t>(duration.count());
        
        updateStats(duration.count());
        
    } catch (const std::exception& e) {
        std::cerr << "Error in processChunk: " << e.what() << std::endl;
        result.text = "";
        result.confidence = 0.0f;
        result.is_final = false;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.latency_ms = static_cast<uint64_t>(duration.count());
    }
    
    return result;
}

void NeMoCacheAwareConformer::updateCacheFromOutputs(std::vector<Ort::Value>& outputs) {
    try {
        if (outputs.size() >= 4) {
            // Update cache_last_channel from output[2]
            auto& new_channel_cache = outputs[2];
            float* new_channel_data = new_channel_cache.GetTensorMutableData<float>();
            auto channel_shape = new_channel_cache.GetTensorTypeAndShapeInfo().GetShape();
            
            size_t channel_cache_elements = 1;
            for (auto dim : channel_shape) {
                channel_cache_elements *= static_cast<size_t>(dim);
            }
            
            if (channel_cache_elements <= cache_last_channel_.size()) {
                std::copy(new_channel_data, new_channel_data + channel_cache_elements, 
                         cache_last_channel_.begin());
            }
            
            // Update cache_last_time from output[3]
            auto& new_time_cache = outputs[3];
            float* new_time_data = new_time_cache.GetTensorMutableData<float>();
            auto time_shape = new_time_cache.GetTensorTypeAndShapeInfo().GetShape();
            
            size_t time_cache_elements = 1;
            for (auto dim : time_shape) {
                time_cache_elements *= static_cast<size_t>(dim);
            }
            
            if (time_cache_elements <= cache_last_time_.size()) {
                std::copy(new_time_data, new_time_data + time_cache_elements,
                         cache_last_time_.begin());
            }
            
            cache_updates_++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error updating cache: " << e.what() << std::endl;
    }
}

std::string NeMoCacheAwareConformer::decodeCTCTokens(const float* log_probs, int64_t seq_len, int64_t num_classes) {
    // Simple CTC decoding using argmax (greedy decoding)
    // In production, would use beam search CTC decoding
    
    std::vector<int> token_ids;
    std::vector<float> max_probs;
    
    // Find best token for each time step
    for (int64_t t = 0; t < seq_len; ++t) {
        float max_prob = -std::numeric_limits<float>::infinity();
        int best_token = 0;
        
        for (int64_t c = 0; c < num_classes; ++c) {
            float prob = log_probs[t * num_classes + c];
            if (prob > max_prob) {
                max_prob = prob;
                best_token = static_cast<int>(c);
            }
        }
        
        token_ids.push_back(best_token);
        max_probs.push_back(max_prob);
    }
    
    // Debug: Print raw tokens for first few time steps
    if (seq_len > 0) {
        std::cout << "Raw CTC tokens: ";
        for (int64_t t = 0; t < std::min(seq_len, static_cast<int64_t>(10)); ++t) {
            std::cout << token_ids[t] << "(" << max_probs[t] << ") ";
        }
        if (seq_len > 10) std::cout << "...";
        std::cout << std::endl;
    }
    
    // Simple CTC collapse - remove consecutive duplicates and blanks (token 0)
    std::vector<int> collapsed_tokens;
    int prev_token = -1;
    
    for (int token : token_ids) {
        if (token != 0 && token != prev_token) {  // 0 is blank token
            collapsed_tokens.push_back(token);
        }
        prev_token = token;
    }
    
    // Convert token IDs to text using vocabulary
    std::string result;
    
    if (vocab_loaded_ && !vocabulary_.empty()) {
        // Use vocabulary to decode tokens
        for (int token_id : collapsed_tokens) {
            if (token_id >= 0 && token_id < static_cast<int>(vocabulary_.size())) {
                std::string token = vocabulary_[token_id];
                
                // Handle subword tokens (starting with ##)
                if (token.substr(0, 2) == "##") {
                    result += token.substr(2);  // Remove ## prefix
                } else if (!result.empty() && !token.empty()) {
                    result += " " + token;  // Add space before whole words
                } else {
                    result += token;
                }
            } else {
                // Unknown token
                result += " [UNK:" + std::to_string(token_id) + "]";
            }
        }
    } else {
        // Fallback to showing token IDs if vocab not loaded
        result = "[NeMo CTC IDs: ";
        for (size_t i = 0; i < collapsed_tokens.size(); ++i) {
            if (i > 0) result += " ";
            result += std::to_string(collapsed_tokens[i]);
        }
        result += "]";
    }
    
    return result;
}

std::string NeMoCacheAwareConformer::decodeTokens(const float* /* logits */, size_t logits_size) {
    // Simplified decoding - in real implementation would use:
    // - CTC decoding for CTC head
    // - RNN-T decoding for transducer head
    // - Hybrid combination of both
    
    // For now, return placeholder indicating NeMo processing
    std::string result = "[NeMo Cache-Aware Conformer: ";
    result += std::to_string(logits_size);
    result += " encoded features]";
    
    return result;
}

void NeMoCacheAwareConformer::reset() {
    // Reset cache tensors to zero
    std::fill(cache_last_channel_.begin(), cache_last_channel_.end(), 0.0f);
    std::fill(cache_last_time_.begin(), cache_last_time_.end(), 0.0f);
    
    // Reset statistics
    total_chunks_processed_ = 0;
    total_processing_time_ms_ = 0;
    cache_updates_ = 0;
    
    std::cout << "NeMo Cache-Aware Conformer cache reset" << std::endl;
}

std::map<std::string, double> NeMoCacheAwareConformer::getStats() const {
    std::map<std::string, double> stats;
    
    stats["total_chunks_processed"] = static_cast<double>(total_chunks_processed_);
    stats["total_processing_time_ms"] = static_cast<double>(total_processing_time_ms_);
    stats["cache_updates"] = static_cast<double>(cache_updates_);
    
    if (total_chunks_processed_ > 0) {
        stats["average_processing_time_ms"] = static_cast<double>(total_processing_time_ms_) / 
                                            static_cast<double>(total_chunks_processed_);
    } else {
        stats["average_processing_time_ms"] = 0.0;
    }
    
    stats["model_type"] = 1.0; // Indicator for NeMo model
    stats["chunk_frames"] = static_cast<double>(config_.chunk_frames);
    stats["feature_dim"] = static_cast<double>(config_.feature_dim);
    stats["cache_channel_size"] = static_cast<double>(cache_last_channel_.size());
    stats["cache_time_size"] = static_cast<double>(cache_last_time_.size());
    
    return stats;
}

void NeMoCacheAwareConformer::updateStats(uint64_t processing_time_ms) const {
    total_chunks_processed_++;
    total_processing_time_ms_ += processing_time_ms;
}

} // namespace onnx_stt