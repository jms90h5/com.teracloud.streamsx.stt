#include "CacheManager.hpp"
#include <algorithm>
#include <iostream>

namespace onnx_stt {

CacheManager::CacheManager(const CacheConfig& config) : config_(config) {}

bool CacheManager::initialize() {
    try {
        switch (config_.cache_type) {
            case CacheConfig::ZIPFORMER_35_CACHE:
                initializeZipformerCaches();
                break;
            case CacheConfig::CONFORMER_3_CACHE:
                initializeConformerCaches();
                break;
            case CacheConfig::WENET_SINGLE_CACHE:
                initializeWenetCaches();
                break;
            case CacheConfig::SPEECHBRAIN_H0_CACHE:
                initializeSpeechBrainCaches();
                break;
            case CacheConfig::CUSTOM_CACHE:
                initializeCustomCaches();
                break;
            default:
                std::cerr << "Unknown cache type" << std::endl;
                return false;
        }
        
        std::cout << "Initialized " << cache_tensors_.size() + int_cache_tensors_.size() 
                  << " cache tensors" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize cache manager: " << e.what() << std::endl;
        return false;
    }
}

void CacheManager::reset() {
    // Reset all float caches to zero
    for (auto& cache_pair : cache_tensors_) {
        std::fill(cache_pair.second.begin(), cache_pair.second.end(), 0.0f);
    }
    
    // Reset all int caches to zero
    for (auto& cache_pair : int_cache_tensors_) {
        std::fill(cache_pair.second.begin(), cache_pair.second.end(), 0);
    }
}

std::vector<Ort::Value> CacheManager::getInputCaches(Ort::MemoryInfo& memory_info) {
    std::vector<Ort::Value> input_caches;
    
    for (const auto& name : config_.cache_tensor_names) {
        auto shape_it = config_.cache_shapes.find(name);
        if (shape_it == config_.cache_shapes.end()) {
            std::cerr << "Warning: No shape found for cache " << name << std::endl;
            continue;
        }
        
        const auto& shape = shape_it->second;
        auto dtype_it = config_.cache_data_types.find(name);
        std::string dtype = (dtype_it != config_.cache_data_types.end()) ? 
                           dtype_it->second : "float32";
        
        if (dtype == "float32") {
            auto& cache_data = getFloatCache(name);
            input_caches.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, cache_data.data(), cache_data.size(),
                shape.data(), shape.size()));
        } else if (dtype == "int64") {
            auto& cache_data = getIntCache(name);
            input_caches.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, cache_data.data(), cache_data.size(),
                shape.data(), shape.size()));
        }
    }
    
    return input_caches;
}

void CacheManager::updateCaches(const std::vector<Ort::Value>& output_caches) {
    if (output_caches.size() != config_.cache_tensor_names.size()) {
        std::cerr << "Warning: Output cache count mismatch" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < output_caches.size(); ++i) {
        const auto& name = config_.cache_tensor_names[i];
        const auto& output_cache = output_caches[i];
        
        auto dtype_it = config_.cache_data_types.find(name);
        std::string dtype = (dtype_it != config_.cache_data_types.end()) ? 
                           dtype_it->second : "float32";
        
        if (dtype == "float32") {
            auto& cache_data = getFloatCache(name);
            const float* output_data = output_cache.GetTensorData<float>();
            size_t size = std::min(cache_data.size(), 
                                  static_cast<size_t>(output_cache.GetTensorTypeAndShapeInfo().GetElementCount()));
            std::copy(output_data, output_data + size, cache_data.begin());
        } else if (dtype == "int64") {
            auto& cache_data = getIntCache(name);
            const int64_t* output_data = output_cache.GetTensorData<int64_t>();
            size_t size = std::min(cache_data.size(), 
                                  static_cast<size_t>(output_cache.GetTensorTypeAndShapeInfo().GetElementCount()));
            std::copy(output_data, output_data + size, cache_data.begin());
        }
    }
}

void CacheManager::initializeZipformerCaches() {
    // Zipformer has 35 cache tensors: 7 types × 5 layers
    const std::vector<std::string> cache_types = {
        "cached_len", "cached_avg", "cached_key", "cached_val", 
        "cached_val2", "cached_conv1", "cached_conv2"
    };
    
    int num_layers = config_.num_layers > 0 ? config_.num_layers : 5;
    
    for (int layer = 0; layer < num_layers; ++layer) {
        for (const auto& cache_type : cache_types) {
            std::string cache_name = cache_type + "_" + std::to_string(layer);
            config_.cache_tensor_names.push_back(cache_name);
            
            // Different cache types have different shapes
            std::vector<int64_t> shape;
            if (cache_type == "cached_len") {
                shape = {1, 1};  // [batch, 1]
                config_.cache_data_types[cache_name] = "int64";
                getIntCache(cache_name).resize(1, 0);
            } else if (cache_type == "cached_avg") {
                shape = {1, 256};  // [batch, feature_dim]
                config_.cache_data_types[cache_name] = "float32";
                getFloatCache(cache_name).resize(256, 0.0f);
            } else {
                // Key, value, conv caches
                shape = {1, 32, 256};  // [batch, sequence, feature_dim]
                config_.cache_data_types[cache_name] = "float32";
                getFloatCache(cache_name).resize(32 * 256, 0.0f);
            }
            
            config_.cache_shapes[cache_name] = shape;
        }
    }
}

void CacheManager::initializeConformerCaches() {
    // Conformer typically has 3 types of caches
    config_.cache_tensor_names = {"encoder_out_cache", "cnn_cache", "att_cache"};
    
    int hidden_size = config_.hidden_size > 0 ? config_.hidden_size : 512;
    int num_layers = config_.num_layers > 0 ? config_.num_layers : 12;
    
    // Encoder output cache
    config_.cache_shapes["encoder_out_cache"] = {1, 32, hidden_size};
    config_.cache_data_types["encoder_out_cache"] = "float32";
    getFloatCache("encoder_out_cache").resize(32 * hidden_size, 0.0f);
    
    // CNN cache
    config_.cache_shapes["cnn_cache"] = {1, num_layers, 32, hidden_size};
    config_.cache_data_types["cnn_cache"] = "float32";
    getFloatCache("cnn_cache").resize(num_layers * 32 * hidden_size, 0.0f);
    
    // Attention cache
    config_.cache_shapes["att_cache"] = {1, num_layers, 32, hidden_size};
    config_.cache_data_types["att_cache"] = "float32";
    getFloatCache("att_cache").resize(num_layers * 32 * hidden_size, 0.0f);
}

void CacheManager::initializeWenetCaches() {
    // WeNet uses a single cache tensor
    config_.cache_tensor_names = {"cache"};
    
    int hidden_size = config_.hidden_size > 0 ? config_.hidden_size : 512;
    config_.cache_shapes["cache"] = {1, 32, hidden_size};
    config_.cache_data_types["cache"] = "float32";
    getFloatCache("cache").resize(32 * hidden_size, 0.0f);
}

void CacheManager::initializeSpeechBrainCaches() {
    // SpeechBrain uses h0 cache
    config_.cache_tensor_names = {"h0"};
    
    int hidden_size = config_.hidden_size > 0 ? config_.hidden_size : 512;
    config_.cache_shapes["h0"] = {1, hidden_size};
    config_.cache_data_types["h0"] = "float32";
    getFloatCache("h0").resize(hidden_size, 0.0f);
}

void CacheManager::initializeCustomCaches() {
    // Initialize based on user-provided configuration
    for (const auto& name : config_.cache_tensor_names) {
        auto shape_it = config_.cache_shapes.find(name);
        if (shape_it == config_.cache_shapes.end()) {
            std::cerr << "Warning: No shape specified for cache " << name << std::endl;
            continue;
        }
        
        auto dtype_it = config_.cache_data_types.find(name);
        std::string dtype = (dtype_it != config_.cache_data_types.end()) ? 
                           dtype_it->second : "float32";
        
        // Calculate total size
        size_t total_size = 1;
        for (int64_t dim : shape_it->second) {
            total_size *= static_cast<size_t>(dim);
        }
        
        if (dtype == "float32") {
            getFloatCache(name).resize(total_size, 0.0f);
        } else if (dtype == "int64") {
            getIntCache(name).resize(total_size, 0);
        }
    }
}

std::vector<float>& CacheManager::getFloatCache(const std::string& name) {
    return cache_tensors_[name];
}

std::vector<int64_t>& CacheManager::getIntCache(const std::string& name) {
    return int_cache_tensors_[name];
}

// Factory methods
CacheManager::CacheConfig CacheManager::createZipformerConfig(int num_layers) {
    CacheConfig config;
    config.cache_type = CacheConfig::ZIPFORMER_35_CACHE;
    config.chunk_frames = 32;
    config.num_layers = num_layers;
    return config;
}

CacheManager::CacheConfig CacheManager::createConformerConfig(int num_layers, int hidden_size) {
    CacheConfig config;
    config.cache_type = CacheConfig::CONFORMER_3_CACHE;
    config.chunk_frames = 16;
    config.num_layers = num_layers;
    config.hidden_size = hidden_size;
    return config;
}

CacheManager::CacheConfig CacheManager::createWenetConfig() {
    CacheConfig config;
    config.cache_type = CacheConfig::WENET_SINGLE_CACHE;
    config.chunk_frames = 16;
    config.hidden_size = 512;
    return config;
}

CacheManager::CacheConfig CacheManager::createSpeechBrainConfig(int hidden_size) {
    CacheConfig config;
    config.cache_type = CacheConfig::SPEECHBRAIN_H0_CACHE;
    config.chunk_frames = 5;
    config.hidden_size = hidden_size;
    return config;
}

} // namespace onnx_stt