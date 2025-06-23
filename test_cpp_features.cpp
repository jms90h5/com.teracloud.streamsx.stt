/**
 * Test C++ feature extraction to compare with working Python features
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include "ImprovedFbank.hpp"

// Simple WAV reader
bool readWav(const std::string& filename, std::vector<float>& audio, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    // Skip WAV header (44 bytes)
    file.seekg(44);
    
    // Read samples as int16 and convert to float
    std::vector<int16_t> samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples.push_back(sample);
    }
    
    // Convert to float
    audio.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    sample_rate = 16000;  // Assume 16kHz
    return true;
}

void saveFeatures(const std::string& filename, const std::vector<std::vector<float>>& features) {
    std::ofstream file(filename, std::ios::binary);
    for (const auto& frame : features) {
        file.write(reinterpret_cast<const char*>(frame.data()), frame.size() * sizeof(float));
    }
}

void printStats(const std::vector<std::vector<float>>& features) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;
    int count = 0;
    
    for (const auto& frame : features) {
        for (float val : frame) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            count++;
        }
    }
    
    float mean = sum / count;
    std::cout << "Feature stats: min=" << min_val << ", max=" << max_val << ", mean=" << mean << std::endl;
}

int main() {
    std::cout << "Testing C++ feature extraction..." << std::endl;
    
    // Load audio
    std::vector<float> audio;
    int sample_rate;
    if (!readWav("test_data/audio/librispeech-1995-1837-0001.wav", audio, sample_rate)) {
        return 1;
    }
    std::cout << "Loaded " << audio.size() << " samples" << std::endl;
    
    // Configure feature extraction to match working features
    improved_fbank::FbankComputer::Options opts;
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;
    opts.frame_length_ms = 25;
    opts.frame_shift_ms = 10;
    opts.n_fft = 512;
    opts.low_freq = 0.0f;
    opts.high_freq = 8000.0f;
    opts.apply_log = true;
    opts.dither = 0.0f;  // No dither for comparison
    opts.normalize_per_feature = false;  // NO normalization!
    
    std::cout << "\nFeature extraction config:" << std::endl;
    std::cout << "  Sample rate: " << opts.sample_rate << std::endl;
    std::cout << "  Mel bins: " << opts.num_mel_bins << std::endl;
    std::cout << "  Frame length: " << opts.frame_length_ms << " ms" << std::endl;
    std::cout << "  Frame shift: " << opts.frame_shift_ms << " ms" << std::endl;
    std::cout << "  FFT size: " << opts.n_fft << std::endl;
    std::cout << "  Apply log: " << (opts.apply_log ? "yes" : "no") << std::endl;
    std::cout << "  Normalize: " << (opts.normalize_per_feature ? "yes" : "NO") << std::endl;
    
    // Extract features
    improved_fbank::FbankComputer fbank(opts);
    auto features = fbank.computeFeatures(audio);
    
    std::cout << "\nExtracted " << features.size() << " frames" << std::endl;
    if (!features.empty()) {
        std::cout << "Feature dimension: " << features[0].size() << std::endl;
    }
    
    // Print stats
    printStats(features);
    
    // Expected stats from working features:
    std::cout << "\nExpected stats from working features:" << std::endl;
    std::cout << "  min=-10.7301, max=6.6779, mean=-3.9120" << std::endl;
    
    // Save features
    saveFeatures("cpp_features.bin", features);
    std::cout << "\nSaved features to cpp_features.bin" << std::endl;
    
    // Save debug info
    std::ofstream debug("cpp_features_debug.txt");
    debug << "Shape: [" << features.size() << ", " << (features.empty() ? 0 : features[0].size()) << "]" << std::endl;
    debug << "First 5 frames (first 10 values each):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), features.size()); ++i) {
        debug << "Frame " << i << ": ";
        for (size_t j = 0; j < std::min(size_t(10), features[i].size()); ++j) {
            debug << features[i][j] << " ";
        }
        debug << "..." << std::endl;
    }
    
    return 0;
}