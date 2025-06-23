/**
 * Debug feature extraction differences
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "ImprovedFbank.hpp"

int main() {
    try {
        std::cout << "=== Debugging Feature Extraction ===" << std::endl;
        
        // Load test audio
        std::string wav_path = "test_audio_16k.wav";
        std::ifstream file(wav_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open " + wav_path);
        }
        
        // Skip WAV header (44 bytes)
        file.seekg(44);
        
        // Read audio data
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        size_t data_size = file_size - 44;
        size_t num_samples = data_size / sizeof(int16_t);
        
        file.seekg(44);
        std::vector<int16_t> audio_data(num_samples);
        file.read(reinterpret_cast<char*>(audio_data.data()), data_size);
        
        std::cout << "Loaded " << num_samples << " samples" << std::endl;
        
        // Convert to float with normalization
        std::vector<float> audio_float(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            audio_float[i] = audio_data[i] / 32768.0f;
        }
        
        // Check audio statistics
        float min_val = *std::min_element(audio_float.begin(), audio_float.end());
        float max_val = *std::max_element(audio_float.begin(), audio_float.end());
        float sum = 0;
        for (auto val : audio_float) sum += val;
        float mean = sum / audio_float.size();
        
        std::cout << "\nAudio statistics:" << std::endl;
        std::cout << "  Min: " << min_val << ", Max: " << max_val << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
        std::cout << "  First 5 samples: ";
        for (int i = 0; i < 5; i++) {
            std::cout << audio_float[i] << " ";
        }
        std::cout << std::endl;
        
        // Create feature extractor with NeMo settings
        ImprovedFbank fbank;
        fbank.Init(
            400,      // frame_length (25ms at 16kHz)
            160,      // frame_shift (10ms at 16kHz)  
            80,       // num_mel_bins
            16000,    // sample_rate
            20,       // low_freq
            0,        // high_freq (0 = nyquist)
            1e-5f,    // dither
            false,    // remove_dc_offset
            "natural",// log_type
            false     // use_energy
        );
        
        // Extract features
        std::vector<float> features = fbank.ComputeFeatures(audio_float);
        int n_frames = features.size() / 80;
        
        std::cout << "\nFeature extraction:" << std::endl;
        std::cout << "  Number of frames: " << n_frames << std::endl;
        std::cout << "  Total features: " << features.size() << std::endl;
        
        // Check feature statistics
        float feat_min = *std::min_element(features.begin(), features.end());
        float feat_max = *std::max_element(features.begin(), features.end());
        float feat_sum = 0;
        for (auto val : features) feat_sum += val;
        float feat_mean = feat_sum / features.size();
        
        std::cout << "\nFeature statistics:" << std::endl;
        std::cout << "  Min: " << feat_min << ", Max: " << feat_max << std::endl;
        std::cout << "  Mean: " << feat_mean << std::endl;
        
        // Print first frame (first 80 values)
        std::cout << "\nFirst 5 features of first frame:" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "  " << features[i] << std::endl;
        }
        
        // Load Python features for comparison
        std::ifstream py_file("python_features.npy", std::ios::binary);
        if (py_file.is_open()) {
            // Skip NPY header
            char magic[6];
            py_file.read(magic, 6);
            uint8_t major, minor;
            py_file.read(reinterpret_cast<char*>(&major), 1);
            py_file.read(reinterpret_cast<char*>(&minor), 1);
            uint16_t header_len;
            py_file.read(reinterpret_cast<char*>(&header_len), 2);
            std::string header(header_len, ' ');
            py_file.read(&header[0], header_len);
            
            // Read Python features (874 frames x 80 features)
            std::vector<float> py_features(874 * 80);
            py_file.read(reinterpret_cast<char*>(py_features.data()), py_features.size() * sizeof(float));
            
            std::cout << "\nPython features (first 5 of first frame):" << std::endl;
            for (int i = 0; i < 5; i++) {
                std::cout << "  " << py_features[i] << std::endl;
            }
            
            // Compare
            std::cout << "\nDifferences:" << std::endl;
            for (int i = 0; i < 5; i++) {
                float diff = std::abs(features[i] - py_features[i]);
                std::cout << "  Feature " << i << " diff: " << diff << std::endl;
            }
        }
        
        // Save C++ features for further analysis
        std::ofstream out_file("cpp_features.bin", std::ios::binary);
        out_file.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(float));
        std::cout << "\nSaved C++ features to cpp_features.bin" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}