/**
 * Test without CMVN - since model expects normalize: NA
 */

#include <iostream>
#include <fstream>
#include <vector>
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
    std::cout << "=== Testing Features WITHOUT CMVN (normalize: NA) ===" << std::endl;
    
    // Create standard ImprovedFbank without CMVN
    improved_fbank::FbankComputer::Options opts;
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;
    opts.frame_length_ms = 25;
    opts.frame_shift_ms = 10;
    opts.n_fft = 512;
    opts.apply_log = true;
    opts.dither = 1e-5f;
    opts.normalize_per_feature = false;  // NO NORMALIZATION per model config
    
    improved_fbank::FbankComputer fbank(opts);
    
    // Load test audio
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    std::vector<float> audio = loadWavFile(audio_file);
    
    if (audio.empty()) {
        std::cerr << "Failed to load audio file: " << audio_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded audio: " << audio.size() << " samples" << std::endl;
    
    // Extract features from first few seconds
    std::vector<float> test_audio(audio.begin(), audio.begin() + std::min<size_t>(32000, audio.size()));  // 2 seconds
    
    std::cout << "Extracting features without CMVN normalization..." << std::endl;
    auto features = fbank.computeFeatures(test_audio);
    
    std::cout << "Feature extraction results:" << std::endl;
    std::cout << "  Number of frames: " << features.size() << std::endl;
    
    if (!features.empty()) {
        // Check feature statistics
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
        
        // These should be log mel features, typically in range [-20, +10]
        if (min_val > -30 && max_val < 20 && avg_val > -15 && avg_val < 5) {
            std::cout << "✅ Feature range looks good for log mel features" << std::endl;
        } else {
            std::cout << "❌ Feature range unusual for log mel: [" << min_val << ", " << max_val << "], avg=" << avg_val << std::endl;
        }
        
        // Check if not all values are the floor
        bool has_variation = false;
        for (const auto& frame : features) {
            for (float val : frame) {
                if (std::abs(val - min_val) > 0.1) {
                    has_variation = true;
                    break;
                }
            }
            if (has_variation) break;
        }
        
        if (has_variation) {
            std::cout << "✅ Features have good variation (not all floor values)" << std::endl;
        } else {
            std::cout << "❌ All features are at floor value - indicates zero energy" << std::endl;
        }
    }
    
    return 0;
}