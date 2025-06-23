#ifndef NEMO_CTC_MODEL_HPP
#define NEMO_CTC_MODEL_HPP

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include "ImprovedFbank.hpp"

namespace onnx_stt {

/**
 * @brief NeMo FastConformer CTC model implementation
 * 
 * This class implements inference for NeMo FastConformer models
 * exported to ONNX with CTC output head.
 * 
 * Model expects:
 * - Input: preprocessed mel-spectrogram features (no normalization)
 * - Output: log probabilities for CTC decoding
 */
class NeMoCTCModel {
public:
    struct Config {
        std::string model_path;
        std::string vocab_path;
        int sample_rate = 16000;
        int n_mels = 80;
        int n_fft = 512;
        float window_size_ms = 25.0f;
        float window_stride_ms = 10.0f;
        float dither = 1e-5f;
        int blank_id = 1024;  // Default for NeMo models
        int num_threads = 4;
    };
    
    struct TranscriptionResult {
        std::string text;
        std::vector<int> token_ids;
        std::vector<float> confidences;
        float avg_confidence;
        int num_frames;
    };
    
    explicit NeMoCTCModel(const Config& config);
    ~NeMoCTCModel();
    
    // Initialize model and vocabulary
    bool initialize();
    
    // Process audio features (mel-spectrogram)
    TranscriptionResult processFeatures(const std::vector<std::vector<float>>& features);
    
    // Extract features from audio
    std::vector<std::vector<float>> extractFeatures(const std::vector<float>& audio);
    
    // Process raw audio
    TranscriptionResult processAudio(const std::vector<float>& audio);
    
    // Get vocabulary
    const std::vector<std::string>& getVocabulary() const { return vocabulary_; }
    
private:
    Config config_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    // Model info
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> vocabulary_;
    
    // Feature extractor
    std::unique_ptr<improved_fbank::FbankComputer> fbank_computer_;
    
    // Random generator for dither
    std::default_random_engine generator_;
    std::normal_distribution<float> dither_dist_;
    
    // Private methods
    bool loadModel();
    bool loadVocabulary();
    void addDither(std::vector<float>& audio);
    std::string greedyCTCDecode(const std::vector<std::vector<float>>& log_probs);
    std::string handleBPETokens(const std::vector<int>& tokens);
};

} // namespace onnx_stt

#endif // NEMO_CTC_MODEL_HPP