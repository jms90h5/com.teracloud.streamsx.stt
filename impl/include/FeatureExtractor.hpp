#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <vector>
#include <memory>
#include <string>

namespace onnx_stt {

/**
 * Abstract interface for audio feature extraction
 * Supports different feature extraction backends (simple_fbank, kaldifeat, etc.)
 */
class FeatureExtractor {
public:
    struct Config {
        int sample_rate = 16000;
        int num_mel_bins = 80;
        int frame_length_ms = 25;
        int frame_shift_ms = 10;
        float low_freq = 20.0f;
        float high_freq = -400.0f;  // Negative means use Nyquist - 400
        bool use_energy = true;
        bool use_log_fbank = true;
        bool apply_cmvn = false;
        std::string cmvn_stats_path;
    };
    
    virtual ~FeatureExtractor() = default;
    
    // Initialize with configuration
    virtual bool initialize(const Config& config) = 0;
    
    // Extract features from audio samples
    virtual std::vector<std::vector<float>> computeFeatures(const std::vector<float>& audio) = 0;
    
    // Extract features from int16 samples
    virtual std::vector<std::vector<float>> computeFeatures(const int16_t* samples, size_t num_samples) = 0;
    
    // Get configuration
    virtual const Config& getConfig() const = 0;
    
    // Get feature dimension
    virtual int getFeatureDim() const = 0;
};

/**
 * Factory functions for different feature extractors
 */
// WARNING: createSimpleFbank generates FAKE data and should NEVER be used!
// std::unique_ptr<FeatureExtractor> createSimpleFbank(const FeatureExtractor::Config& config);
std::unique_ptr<FeatureExtractor> createKaldifeat(const FeatureExtractor::Config& config);
std::unique_ptr<FeatureExtractor> createImprovedFbank(const FeatureExtractor::Config& config);

} // namespace onnx_stt

#endif // FEATURE_EXTRACTOR_HPP