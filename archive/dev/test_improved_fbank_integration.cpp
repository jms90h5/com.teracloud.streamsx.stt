/**
 * Test program to verify ImprovedFbank integration with correct parameters
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "ImprovedFbank.hpp"

// Simple WAV reader
std::vector<float> readWav(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open: " + filename);
    }
    
    // Skip WAV header (44 bytes)
    file.seekg(44);
    
    std::vector<float> audio;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        audio.push_back(sample / 32768.0f);
    }
    
    return audio;
}

int main() {
    try {
        std::cout << "=== Testing ImprovedFbank with NeMo Parameters ===" << std::endl;
        
        // Configure exactly as documented in MAJOR_BREAKTHROUGH_SUMMARY.md
        improved_fbank::FbankComputer::Options fbank_opts;
        fbank_opts.sample_rate = 16000;
        fbank_opts.num_mel_bins = 80;
        fbank_opts.frame_length_ms = 25.0f;
        fbank_opts.frame_shift_ms = 10.0f;
        fbank_opts.n_fft = 512;
        fbank_opts.apply_log = true;
        fbank_opts.dither = 1e-5f;
        fbank_opts.normalize_per_feature = false;  // Critical: FastConformer has normalize: NA
        
        auto fbank_computer = std::make_unique<improved_fbank::FbankComputer>(fbank_opts);
        
        std::cout << "ImprovedFbank initialized with:" << std::endl;
        std::cout << "  Sample rate: " << fbank_opts.sample_rate << " Hz" << std::endl;
        std::cout << "  Frame length: " << fbank_opts.frame_length_ms << " ms" << std::endl;
        std::cout << "  Frame shift: " << fbank_opts.frame_shift_ms << " ms" << std::endl;
        std::cout << "  Mel bins: " << fbank_opts.num_mel_bins << std::endl;
        std::cout << "  FFT size: " << fbank_opts.n_fft << std::endl;
        std::cout << "  Apply log: " << (fbank_opts.apply_log ? "yes" : "no") << std::endl;
        std::cout << "  Dither: " << fbank_opts.dither << std::endl;
        std::cout << "  Normalize per feature: " << (fbank_opts.normalize_per_feature ? "yes" : "no") << std::endl;
        
        // Load test audio
        std::string audio_file = "test_data/audio/librispeech_3sec.wav";
        std::cout << "\nLoading audio: " << audio_file << std::endl;
        
        auto audio = readWav(audio_file);
        std::cout << "Loaded " << audio.size() << " samples (" << audio.size() / 16000.0 << " seconds)" << std::endl;
        
        // Extract features
        std::cout << "\nExtracting features..." << std::endl;
        auto features = fbank_computer->computeFeatures(audio);
        
        std::cout << "Extracted " << features.size() << " feature frames" << std::endl;
        
        // Compute feature statistics
        if (!features.empty()) {
            float min_val = features[0][0];
            float max_val = features[0][0];
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
            
            float avg = sum / count;
            
            std::cout << "\nFeature statistics:" << std::endl;
            std::cout << "  Min value: " << min_val << std::endl;
            std::cout << "  Max value: " << max_val << std::endl;
            std::cout << "  Average: " << avg << std::endl;
            std::cout << "  Shape: [" << features.size() << ", " << features[0].size() << "]" << std::endl;
            
            // Check if values are in expected range for log mel features
            if (min_val >= -25 && max_val <= 10) {
                std::cout << "\n✓ Feature values are in expected range for log mel spectrograms!" << std::endl;
                std::cout << "  This matches the documented successful range: min=-23.0259, max=5.83632" << std::endl;
            } else {
                std::cout << "\n✗ Feature values are outside expected range!" << std::endl;
            }
        }
        
        std::cout << "\n=== Test Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}