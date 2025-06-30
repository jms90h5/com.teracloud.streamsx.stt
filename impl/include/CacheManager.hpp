#ifndef CACHE_MANAGER_HPP
#define CACHE_MANAGER_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "onnx_wrapper.hpp"

namespace onnx_stt {

/**
 * Configurable cache management for different streaming ASR models
 * Supports various cache schemas (Zipformer, Conformer, etc.)
 */
class CacheManager {
public:
    struct CacheConfig {
        // Cache tensor specifications
        std::vector<std::string> cache_tensor_names;
        std::map<std::string, std::vector<int64_t>> cache_shapes;
        std::map<std::string, std::string> cache_data_types;  // "float32", "int64", etc.
        
        // Model-specific parameters
        int chunk_frames = 32;
        int num_layers = 0;
        int hidden_size = 0;
        
        // Cache type identifiers
        enum CacheType {
            ZIPFORMER_35_CACHE,      // 35 cache tensors (7 types × 5 layers)
            CONFORMER_3_CACHE,       // encoder_out_cache, cnn_cache, att_cache
            WENET_SINGLE_CACHE,      // Single "cache" tensor
            SPEECHBRAIN_H0_CACHE,    // "h0" cache tensor
            CUSTOM_CACHE             // User-defined cache schema
        } cache_type = ZIPFORMER_35_CACHE;
    };
    
    explicit CacheManager(const CacheConfig& config);
    ~CacheManager() = default;
    
    // Initialize cache tensors
    bool initialize();
    
    // Reset all caches to initial state
    void reset();
    
    // Get cache tensors for model input
    std::vector<Ort::Value> getInputCaches(Ort::MemoryInfo& memory_info);
    
    // Update caches from model output
    void updateCaches(const std::vector<Ort::Value>& output_caches);
    
    // Get cache configuration
    const CacheConfig& getConfig() const { return config_; }
    
    // Get number of cache tensors
    size_t getNumCaches() const { return cache_tensors_.size(); }
    
    // Get cache tensor names (for model I/O)
    const std::vector<std::string>& getCacheNames() const { return config_.cache_tensor_names; }
    
    // Factory methods for predefined cache types
    static CacheConfig createZipformerConfig(int num_layers = 5);
    static CacheConfig createConformerConfig(int num_layers = 12, int hidden_size = 512);
    static CacheConfig createWenetConfig();
    static CacheConfig createSpeechBrainConfig(int hidden_size = 512);
    
private:
    CacheConfig config_;
    
    // Cache data storage
    std::map<std::string, std::vector<float>> cache_tensors_;
    std::map<std::string, std::vector<int64_t>> int_cache_tensors_;
    
    // Helper methods
    void initializeZipformerCaches();
    void initializeConformerCaches();
    void initializeWenetCaches();
    void initializeSpeechBrainCaches();
    void initializeCustomCaches();
    
    std::vector<float>& getFloatCache(const std::string& name);
    std::vector<int64_t>& getIntCache(const std::string& name);
};

} // namespace onnx_stt

#endif // CACHE_MANAGER_HPP