/**
 * Test using the working CMVN configuration from documentation
 * Based on the pattern from test_nemo_improvements.cpp and test_real_nemo.cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "impl/include/ImprovedFbank.hpp"

// Simple WAV header structure
struct WavHeader {
    char riff[4];
    uint32_t size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

std::vector<float> loadWavFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }
    
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Read audio data
    std::vector<int16_t> samples(header.data_size / 2);
    file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
    
    // Convert to float
    std::vector<float> audio(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    return audio;
}

int main() {
    std::cout << "=== Testing Working CMVN Configuration ===" << std::endl;
    
    // Use the createNeMoCompatibleFbank function with CMVN stats
    std::string cmvn_path = "samples/CppONNX_OnnxSTT/models/global_cmvn.stats";
    auto fbank = improved_fbank::createNeMoCompatibleFbank(cmvn_path);
    
    if (!fbank) {
        std::cerr << "Failed to create NeMo-compatible fbank" << std::endl;
        return 1;
    }
    
    // Load test audio
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    std::vector<float> audio = loadWavFile(audio_file);
    
    if (audio.empty()) {
        std::cerr << "Failed to load audio file: " << audio_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded audio: " << audio.size() << " samples" << std::endl;
    
    // Extract features using working configuration
    std::cout << "Extracting features with CMVN normalization..." << std::endl;
    auto features = fbank->computeFeatures(audio);
    
    std::cout << "Feature extraction results:" << std::endl;
    std::cout << "  Number of frames: " << features.size() << std::endl;
    
    if (!features.empty()) {
        // Check if features look normalized (should be roughly zero-mean)
        float total_sum = 0;
        float total_count = 0;
        float min_val = features[0][0];
        float max_val = features[0][0];
        
        for (const auto& frame : features) {
            for (float val : frame) {
                total_sum += val;
                total_count++;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        
        float avg_val = total_sum / total_count;
        
        std::cout << "  Feature statistics:" << std::endl;
        std::cout << "    Min: " << min_val << std::endl;
        std::cout << "    Max: " << max_val << std::endl;
        std::cout << "    Average: " << avg_val << std::endl;
        
        // Show first frame
        std::cout << "  First frame (first 10 features):" << std::endl;
        for (int i = 0; i < 10 && i < features[0].size(); i++) {
            std::cout << "    [" << i << "] = " << features[0][i] << std::endl;
        }
        
        // Check if features are properly normalized
        if (std::abs(avg_val) < 1.0) {  // Should be roughly zero-mean
            std::cout << "✅ Features appear properly normalized (mean ≈ 0)" << std::endl;
        } else {
            std::cout << "❌ Features may not be properly normalized (mean = " << avg_val << ")" << std::endl;
        }
        
        // Check range - normalized features should be roughly in [-3, 3] range
        if (min_val > -10 && max_val < 10) {
            std::cout << "✅ Feature range looks reasonable for normalized data" << std::endl;
        } else {
            std::cout << "❌ Feature range seems too large: [" << min_val << ", " << max_val << "]" << std::endl;
        }
    }
    
    return 0;
}