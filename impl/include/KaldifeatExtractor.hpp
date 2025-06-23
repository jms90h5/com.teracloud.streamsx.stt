#ifndef KALDIFEAT_EXTRACTOR_HPP
#define KALDIFEAT_EXTRACTOR_HPP

#include "FeatureExtractor.hpp"
// REMOVED: #include "simple_fbank.hpp" - generates FAKE data, NEVER use!
#include <memory>

// Forward declarations for kaldifeat types
namespace kaldifeat {
    class OnlineFbank;
    struct FbankOptions;
}

namespace onnx_stt {

/**
 * Kaldifeat-based feature extractor
 * Uses the C++17 static library version of kaldifeat
 * NEVER falls back to simple_fbank as it generates FAKE data
 */
class KaldifeatExtractor : public FeatureExtractor {
public:
    explicit KaldifeatExtractor(const Config& config);
    ~KaldifeatExtractor() override = default;
    
    bool initialize(const Config& config) override;
    
    std::vector<std::vector<float>> computeFeatures(const std::vector<float>& audio) override;
    std::vector<std::vector<float>> computeFeatures(const int16_t* samples, size_t num_samples) override;
    
    const Config& getConfig() const override { return config_; }
    int getFeatureDim() const override;
    
private:
    Config config_;
    bool kaldifeat_available_;
    
    // REMOVED: simple_fbank fallback - it generates FAKE data!
    
    // CMVN statistics
    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_var_;
    bool cmvn_loaded_;
    
    // Helper methods
    bool loadCmvnStats(const std::string& stats_path);
    void applyCmvn(std::vector<std::vector<float>>& features);
    std::vector<float> convertInt16ToFloat(const int16_t* samples, size_t num_samples);
    
    // Kaldifeat-specific members (conditionally compiled)
#ifdef HAVE_KALDIFEAT
    std::unique_ptr<kaldifeat::OnlineFbank> kaldifeat_fbank_;
    std::unique_ptr<kaldifeat::FbankOptions> kaldifeat_opts_;
#endif
};

// REMOVED: SimpleFbankExtractor class - it used simple_fbank which generates FAKE data!

} // namespace onnx_stt

#endif // KALDIFEAT_EXTRACTOR_HPP