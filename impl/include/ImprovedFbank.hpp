#ifndef IMPROVED_FBANK_HPP
#define IMPROVED_FBANK_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <complex>
#include <memory>

namespace improved_fbank {

/**
 * Proper mel filterbank feature extractor for NeMo compatibility
 * Implements real mel-scale filter banks with FFT-based spectrogram computation
 */
class FbankComputer {
public:
    struct Options {
        int sample_rate = 16000;
        int num_mel_bins = 80;
        int frame_length_ms = 25;
        int frame_shift_ms = 10;
        int n_fft = 512;
        float low_freq = 0.0f;
        float high_freq = 8000.0f;  // Nyquist frequency for 16kHz
        bool use_energy = true;
        bool apply_log = true;
        float dither = 1e-5f;  // NeMo default dither
        bool normalize_per_feature = true;  // NeMo: normalize: per_feature
    };
    
    explicit FbankComputer(const Options& opts);
    
    // Main feature computation method
    std::vector<std::vector<float>> computeFeatures(const std::vector<float>& audio);
    
    // Apply CMVN normalization if stats are available
    void setCMVNStats(const std::vector<float>& mean_stats, const std::vector<float>& var_stats, int frame_count);
    
    int getFeatureDim() const { return opts_.num_mel_bins; }
    
private:
    Options opts_;
    int frame_length_samples_;
    int frame_shift_samples_;
    
    // Window function (Hann window for NeMo compatibility)
    std::vector<float> window_;
    
    // Mel filterbank matrix [num_mel_bins x (n_fft/2 + 1)]
    std::vector<std::vector<float>> mel_filterbank_;
    
    // CMVN statistics
    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_var_;
    bool cmvn_available_;
    
    // Private methods
    void initializeWindow();
    void initializeMelFilterbank();
    float melScale(float freq);
    float invMelScale(float mel);
    std::vector<float> computeFFT(const std::vector<float>& frame);
    std::vector<float> applyMelFilterbank(const std::vector<float>& power_spectrum);
    void applyCMVN(std::vector<std::vector<float>>& features);
    void applyDither(std::vector<float>& audio);
};

// Utility function to load CMVN stats from NeMo format
std::unique_ptr<FbankComputer> createNeMoCompatibleFbank(const std::string& cmvn_stats_path = "");

} // namespace improved_fbank

#endif // IMPROVED_FBANK_HPP