#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include "impl/include/ImprovedFbank.hpp"

int main() {
    std::cout << "=== Testing Feature Extraction Only ===" << std::endl;
    
    // Initialize feature extractor
    improved_fbank::FbankComputer::Options opts;
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;
    opts.frame_length_ms = 25;
    opts.frame_shift_ms = 10;
    opts.n_fft = 512;
    opts.apply_log = true;
    opts.dither = 1e-5f;
    opts.normalize_per_feature = false;  // Critical: NO normalization
    
    improved_fbank::FbankComputer fbank(opts);
    
    // Load audio
    std::ifstream wav_file("test_data/audio/librispeech-1995-1837-0001.wav", std::ios::binary);
    if (!wav_file.is_open()) {
        std::cerr << "Cannot open WAV file" << std::endl;
        return 1;
    }
    
    // Skip WAV header (44 bytes)
    wav_file.seekg(44);
    
    // Read audio samples
    std::vector<int16_t> samples;
    int16_t sample;
    while (wav_file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples.push_back(sample);
    }
    wav_file.close();
    
    std::cout << "Loaded " << samples.size() << " samples" << std::endl;
    
    // Convert to float
    std::vector<float> float_samples(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float_samples[i] = samples[i] / 32768.0f;
    }
    
    // Extract features
    auto features = fbank.computeFeatures(float_samples);
    
    std::cout << "Extracted " << features.size() << " frames" << std::endl;
    if (!features.empty()) {
        std::cout << "Feature dimensions: " << features[0].size() << std::endl;
    }
    
    // Calculate statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;
    size_t count = 0;
    
    for (const auto& frame : features) {
        for (float val : frame) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            count++;
        }
    }
    
    float mean = sum / count;
    
    // Calculate std dev
    float sum_sq = 0.0f;
    for (const auto& frame : features) {
        for (float val : frame) {
            float diff = val - mean;
            sum_sq += diff * diff;
        }
    }
    float std_dev = std::sqrt(sum_sq / count);
    
    std::cout << "\nFeature statistics:" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std: " << std_dev << std::endl;
    
    std::cout << "\nExpected statistics from working features:" << std::endl;
    std::cout << "  Min: -10.73" << std::endl;
    std::cout << "  Max: 6.68" << std::endl;
    std::cout << "  Mean: -3.91" << std::endl;
    std::cout << "  Std: 2.77" << std::endl;
    
    // Save features for comparison
    std::ofstream out("cpp_features_test.bin", std::ios::binary);
    for (const auto& frame : features) {
        out.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(float));
    }
    out.close();
    
    std::cout << "\nSaved features to cpp_features_test.bin" << std::endl;
    
    return 0;
}