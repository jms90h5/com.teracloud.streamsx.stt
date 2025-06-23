#include "../include/FeatureExtractor.hpp"
#include "../include/ImprovedFbank.hpp"
#include <iostream>

namespace onnx_stt {

class ImprovedFbankAdapter : public FeatureExtractor {
public:
    ImprovedFbankAdapter() : fbank_(nullptr) {}
    
    bool initialize(const Config& config) override {
        config_ = config;
        
        // Create ImprovedFbank with proper configuration
        improved_fbank::FbankComputer::Options opts;
        opts.sample_rate = config.sample_rate;
        opts.num_mel_bins = config.num_mel_bins;
        opts.frame_length_ms = config.frame_length_ms;
        opts.frame_shift_ms = config.frame_shift_ms;
        opts.low_freq = config.low_freq;
        opts.high_freq = (config.high_freq < 0) ? (config.sample_rate / 2.0f + config.high_freq) : config.high_freq;
        opts.use_energy = config.use_energy;
        opts.apply_log = config.use_log_fbank;
        
        fbank_ = std::make_unique<improved_fbank::FbankComputer>(
            opts
        );
        
        // Load CMVN stats if path provided
        if (!config.cmvn_stats_path.empty() && config.apply_cmvn) {
            // Note: ImprovedFbank doesn't have built-in CMVN loading, would need to add that separately
            std::cerr << "Warning: CMVN stats loading not implemented in ImprovedFbank adapter" << std::endl;
        }
        
        return true;
    }
    
    std::vector<std::vector<float>> computeFeatures(const std::vector<float>& audio) override {
        if (!fbank_) {
            std::cerr << "ERROR: ImprovedFbankAdapter not initialized!" << std::endl;
            return {};
        }
        
        // ImprovedFbank already returns features as vector<vector<float>> in [time][features] format
        return fbank_->computeFeatures(audio);
    }
    
    std::vector<std::vector<float>> computeFeatures(const int16_t* samples, size_t num_samples) override {
        // Convert int16 to float
        std::vector<float> audio(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            audio[i] = samples[i] / 32768.0f;
        }
        return computeFeatures(audio);
    }
    
    const Config& getConfig() const override {
        return config_;
    }
    
    int getFeatureDim() const override {
        return config_.num_mel_bins;
    }
    
private:
    Config config_;
    std::unique_ptr<improved_fbank::FbankComputer> fbank_;
};

// Factory function
std::unique_ptr<FeatureExtractor> createImprovedFbank(const FeatureExtractor::Config& config) {
    auto extractor = std::make_unique<ImprovedFbankAdapter>();
    if (extractor->initialize(config)) {
        return extractor;
    }
    return nullptr;
}

} // namespace onnx_stt