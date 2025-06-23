/**
 * Test program to verify FFT fix in ImprovedFbank
 * Compares C++ features with Python reference
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
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

void saveFeatures(const std::string& filename, const std::vector<std::vector<float>>& features) {
    std::ofstream file(filename);
    file << "Shape: [" << features.size() << ", " << (features.empty() ? 0 : features[0].size()) << "]\n";
    file << "First 5 frames (first 10 values each):\n";
    for (size_t i = 0; i < std::min(size_t(5), features.size()); ++i) {
        file << "Frame " << i << ": ";
        for (size_t j = 0; j < std::min(size_t(10), features[i].size()); ++j) {
            file << std::fixed << std::setprecision(6) << features[i][j] << " ";
        }
        file << "...\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Testing FFT Fix in ImprovedFbank ===" << std::endl;
        
        // Configure exactly as Python
        improved_fbank::FbankComputer::Options fbank_opts;
        fbank_opts.sample_rate = 16000;
        fbank_opts.num_mel_bins = 80;
        fbank_opts.frame_length_ms = 25.0f;  // 400 samples
        fbank_opts.frame_shift_ms = 10.0f;   // 160 samples
        fbank_opts.n_fft = 512;
        fbank_opts.low_freq = 0.0f;
        fbank_opts.high_freq = 8000.0f;
        fbank_opts.apply_log = true;
        fbank_opts.dither = 1e-5f;
        fbank_opts.use_energy = true;
        fbank_opts.normalize_per_feature = false;  // Python doesn't normalize
        
        auto fbank_computer = std::make_unique<improved_fbank::FbankComputer>(fbank_opts);
        
        // Use audio file from command line or default
        std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        if (argc > 1) {
            audio_file = argv[1];
        }
        
        std::cout << "\nLoading audio: " << audio_file << std::endl;
        auto audio = readWav(audio_file);
        std::cout << "Loaded " << audio.size() << " samples (" << audio.size() / 16000.0 << " seconds)" << std::endl;
        
        // Extract features
        std::cout << "\nExtracting features with FFT..." << std::endl;
        auto features = fbank_computer->computeFeatures(audio);
        
        std::cout << "Extracted " << features.size() << " feature frames" << std::endl;
        
        // Compute statistics
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
            
            // Print first frame values (compare with Python)
            std::cout << "\nFirst frame (first 10 values):" << std::endl;
            for (int i = 0; i < 10 && i < features[0].size(); ++i) {
                std::cout << "  [" << i << "]: " << std::fixed << std::setprecision(6) << features[0][i] << std::endl;
            }
            
            // Check against Python reference values
            std::cout << "\nPython reference (from save_python_features.py):" << std::endl;
            std::cout << "  First value: -9.597xxx" << std::endl;
            std::cout << "  Range: min=-23.0259, max=5.83632" << std::endl;
            
            // Check if first value is close to Python
            float diff = std::abs(features[0][0] - (-9.597f));
            if (diff < 0.5f) {
                std::cout << "\n✓ First value matches Python! (difference: " << diff << ")" << std::endl;
            } else {
                std::cout << "\n✗ First value differs from Python by " << diff << std::endl;
            }
            
            // Save features for detailed comparison
            saveFeatures("cpp_features_fft.txt", features);
            std::cout << "\nSaved features to cpp_features_fft.txt for comparison" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}