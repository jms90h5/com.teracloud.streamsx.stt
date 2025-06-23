#ifndef SILERO_VAD_HPP
#define SILERO_VAD_HPP

#include "VADInterface.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

namespace onnx_stt {

/**
 * Silero VAD implementation using ONNX Runtime
 * 
 * Silero VAD is a state-of-the-art voice activity detection model
 * that can distinguish speech from silence, music, and noise.
 * 
 * Model: https://github.com/snakers4/silero-vad
 */
class SileroVAD : public VADInterface {
public:
    explicit SileroVAD(const Config& config);
    ~SileroVAD() override = default;
    
    bool initialize(const Config& config) override;
    
    VADResult processChunk(const int16_t* samples, 
                          size_t num_samples, 
                          uint64_t timestamp_ms) override;
    
    VADResult processChunk(const std::vector<float>& audio, 
                          uint64_t timestamp_ms) override;
    
    void reset() override;
    
    const Config& getConfig() const override { return config_; }
    
private:
    Config config_;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // Model metadata
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Internal state for streaming
    std::vector<float> state_h_;
    std::vector<float> state_c_;
    std::vector<float> audio_buffer_;
    
    // Model configuration
    int window_size_samples_;
    int frame_shift_samples_;
    
    // Helper methods
    bool loadModel(const std::string& model_path);
    std::vector<float> preprocessAudio(const std::vector<float>& audio);
    VADResult runInference(const std::vector<float>& audio_chunk, uint64_t timestamp_ms);
    void convertInt16ToFloat(const int16_t* samples, size_t num_samples, 
                            std::vector<float>& output);
};

/**
 * Simple energy-based VAD as fallback
 */
class EnergyVAD : public VADInterface {
public:
    explicit EnergyVAD(const Config& config);
    ~EnergyVAD() override = default;
    
    bool initialize(const Config& config) override;
    
    VADResult processChunk(const int16_t* samples, 
                          size_t num_samples, 
                          uint64_t timestamp_ms) override;
    
    VADResult processChunk(const std::vector<float>& audio, 
                          uint64_t timestamp_ms) override;
    
    void reset() override;
    
    const Config& getConfig() const override { return config_; }
    
private:
    Config config_;
    float energy_threshold_;
    std::vector<float> energy_history_;
    size_t history_size_;
    
    float calculateEnergy(const std::vector<float>& audio);
    void updateThreshold(float energy);
};

} // namespace onnx_stt

#endif // SILERO_VAD_HPP