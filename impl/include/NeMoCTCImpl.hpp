#pragma once

#include "onnx_wrapper.hpp"
#include "ImprovedFbank.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

class NeMoCTCImpl {
public:
    NeMoCTCImpl();
    ~NeMoCTCImpl();
    
    // Initialize with CTC model and tokens
    bool initialize(const std::string& model_path, const std::string& tokens_path);
    
    // Process audio and return transcription
    std::string transcribe(const std::vector<float>& audio_samples);
    
    // Get model info
    std::string getModelInfo() const;
    bool isInitialized() const { return initialized_; }
    
private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // State
    bool initialized_;
    
    // Vocabulary
    std::unordered_map<int, std::string> vocab_;
    int blank_id_;
    
    // Feature extractor using ImprovedFbank
    std::unique_ptr<improved_fbank::FbankComputer> fbank_computer_;
    
    // Helper methods
    bool loadVocabulary(const std::string& tokens_path);
    std::vector<float> extractMelFeatures(const std::vector<float>& audio_samples);
    std::string ctcDecode(const std::vector<float>& logits, const std::vector<int64_t>& shape);
};