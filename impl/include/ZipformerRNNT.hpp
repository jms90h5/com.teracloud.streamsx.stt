#ifndef ZIPFORMER_RNNT_HPP
#define ZIPFORMER_RNNT_HPP

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <array>
#include <algorithm>
#include "onnxruntime_cxx_api.h"

namespace onnx_stt {

/**
 * Zipformer RNN-T implementation for streaming ASR
 * Handles the complete encoder-decoder-joiner pipeline with cache management
 */
class ZipformerRNNT {
public:
    struct Config {
        // Model paths
        std::string encoder_path;
        std::string decoder_path;
        std::string joiner_path;
        std::string tokens_path;
        
        // Model parameters
        int num_layers = 5;
        int chunk_size = 39;  // frames per chunk
        int feature_dim = 80;
        int encoder_dim = 384;
        int decoder_dim = 512;
        
        // Decoding parameters
        int beam_size = 4;
        float blank_penalty = 0.0f;
        int max_active_paths = 4;
        
        // Performance
        int num_threads = 4;
    };
    
    // Cache state for streaming
    struct CacheState {
        // Length caches
        std::array<std::vector<int64_t>, 5> cached_len;
        
        // Average caches
        std::array<std::vector<float>, 5> cached_avg;
        
        // Attention key/value caches
        std::array<std::vector<float>, 5> cached_key;
        std::array<std::vector<float>, 5> cached_val;
        std::array<std::vector<float>, 5> cached_val2;
        
        // Convolution caches
        std::array<std::vector<float>, 5> cached_conv1;
        std::array<std::vector<float>, 5> cached_conv2;
        
        // Initialize cache dimensions based on model
        void initialize(const Config& config);
        
        // Get all cache tensors as ONNX values
        std::vector<Ort::Value> toOnnxValues(Ort::MemoryInfo& memory_info);
        
        // Update caches from model outputs
        void updateFromOutputs(const std::vector<Ort::Value>& outputs);
    };
    
    // Hypothesis for beam search
    struct Hypothesis {
        std::vector<int> tokens;           // Token sequence
        std::vector<float> decoder_state;  // Decoder hidden state
        float score = 0.0f;                // Log probability
        
        bool operator<(const Hypothesis& other) const {
            return score < other.score;  // For priority queue
        }
    };
    
    // Result from processing
    struct Result {
        std::string text;
        std::vector<int> tokens;
        float confidence;
        bool is_final;
    };
    
    explicit ZipformerRNNT(const Config& config);
    ~ZipformerRNNT();
    
    // Initialize models
    bool initialize();
    
    // Process audio chunk (streaming)
    Result processChunk(const std::vector<float>& features);
    
    // Finalize decoding (end of stream)
    Result finalize();
    
    // Reset for new utterance
    void reset();
    
private:
    Config config_;
    
    // ONNX Runtime
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    
    // Model sessions
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;
    std::unique_ptr<Ort::Session> joiner_;
    
    // Vocabulary
    std::vector<std::string> tokens_;
    int blank_id_ = 0;
    
    // Streaming state
    CacheState cache_state_;
    std::vector<Hypothesis> hypotheses_;
    
    // Internal methods
    std::vector<float> runEncoder(const std::vector<float>& features);
    std::vector<float> runDecoder(const std::vector<int>& tokens, 
                                 const std::vector<float>& state);
    std::vector<float> runJoiner(const std::vector<float>& encoder_out,
                                const std::vector<float>& decoder_out);
    
    // Beam search
    void beamSearchStep(const std::vector<float>& encoder_out);
    std::string tokensToText(const std::vector<int>& tokens);
    
    // Helper methods
    bool loadTokens(const std::string& path);
    std::vector<int64_t> getEncoderCacheShape(int layer, const std::string& cache_type);
};

// Implementation of CacheState methods
inline void ZipformerRNNT::CacheState::initialize(const Config& config) {
    // Initialize cache dimensions based on Zipformer architecture
    // These match the input shapes we saw in the test
    
    // Layer configurations
    const int layer_dims[] = {2, 4, 3, 2, 4};
    const int downsample_factors[] = {64, 32, 16, 8, 32};
    
    for (int i = 0; i < 5; ++i) {
        int num_layers = layer_dims[i];
        int downsample = downsample_factors[i];
        
        // Length caches: [num_layers, batch_size]
        cached_len[i].resize(num_layers * 1, 0);
        
        // Average caches: [num_layers, batch_size, encoder_dim]
        cached_avg[i].resize(num_layers * 1 * config.encoder_dim, 0.0f);
        
        // Attention caches
        int head_dim = 192;  // For keys
        int value_dim = 96;  // For values
        
        // Key cache: [num_layers, downsample, batch_size, head_dim]
        cached_key[i].resize(num_layers * downsample * 1 * head_dim, 0.0f);
        
        // Value caches: [num_layers, downsample, batch_size, value_dim]
        cached_val[i].resize(num_layers * downsample * 1 * value_dim, 0.0f);
        cached_val2[i].resize(num_layers * downsample * 1 * value_dim, 0.0f);
        
        // Conv caches: [num_layers, batch_size, encoder_dim, kernel_size]
        int kernel_size = 30;
        cached_conv1[i].resize(num_layers * 1 * config.encoder_dim * kernel_size, 0.0f);
        cached_conv2[i].resize(num_layers * 1 * config.encoder_dim * kernel_size, 0.0f);
    }
}

inline std::vector<Ort::Value> ZipformerRNNT::CacheState::toOnnxValues(
    Ort::MemoryInfo& memory_info) {
    
    std::vector<Ort::Value> values;
    
    // Create tensors for each cache in the correct order
    // This must match the model's input order exactly
    
    const int layer_dims[] = {2, 4, 3, 2, 4};
    const int downsample_factors[] = {64, 32, 16, 8, 32};
    const int encoder_dim = 384;
    
    // Add cached_len tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], 1};
        values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info,
            cached_len[i].data(),
            cached_len[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_avg tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], 1, encoder_dim};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_avg[i].data(),
            cached_avg[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_key tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], downsample_factors[i], 1, 192};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_key[i].data(),
            cached_key[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_val tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], downsample_factors[i], 1, 96};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_val[i].data(),
            cached_val[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_val2 tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], downsample_factors[i], 1, 96};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_val2[i].data(),
            cached_val2[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_conv1 tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], 1, encoder_dim, 30};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_conv1[i].data(),
            cached_conv1[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    // Add cached_conv2 tensors
    for (int i = 0; i < 5; ++i) {
        std::vector<int64_t> shape = {layer_dims[i], 1, encoder_dim, 30};
        values.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            cached_conv2[i].data(),
            cached_conv2[i].size(),
            shape.data(),
            shape.size()
        ));
    }
    
    return values;
}

inline void ZipformerRNNT::CacheState::updateFromOutputs(
    const std::vector<Ort::Value>& outputs) {
    
    // The encoder outputs new cache values in the same order as inputs
    // Starting from output index 1 (index 0 is the encoder output)
    
    // Skip the first output (encoder_out) and process cache updates
    size_t output_idx = 1;
    
    // Update cached_len
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const int64_t* data = outputs[output_idx].GetTensorData<int64_t>();
            size_t size = cached_len[i].size();
            std::copy(data, data + size, cached_len[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_avg
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_avg[i].size();
            std::copy(data, data + size, cached_avg[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_key
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_key[i].size();
            std::copy(data, data + size, cached_key[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_val
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_val[i].size();
            std::copy(data, data + size, cached_val[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_val2
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_val2[i].size();
            std::copy(data, data + size, cached_val2[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_conv1
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_conv1[i].size();
            std::copy(data, data + size, cached_conv1[i].begin());
            output_idx++;
        }
    }
    
    // Update cached_conv2
    for (int i = 0; i < 5; ++i) {
        if (output_idx < outputs.size()) {
            const float* data = outputs[output_idx].GetTensorData<float>();
            size_t size = cached_conv2[i].size();
            std::copy(data, data + size, cached_conv2[i].begin());
            output_idx++;
        }
    }
}

} // namespace onnx_stt

#endif // ZIPFORMER_RNNT_HPP