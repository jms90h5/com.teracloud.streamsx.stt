#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "ImprovedFbank.hpp"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

// Simple WAV reader
std::vector<float> readWav(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Skip WAV header (44 bytes)
    file.seekg(44);
    
    std::vector<int16_t> samples_int16;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples_int16.push_back(sample);
    }
    
    // Convert to float [-1, 1]
    std::vector<float> samples_float;
    samples_float.reserve(samples_int16.size());
    for (int16_t s : samples_int16) {
        samples_float.push_back(s / 32768.0f);
    }
    
    return samples_float;
}

void testKaldiFbank(const std::vector<float>& audio) {
    std::cout << "\n=== Testing Kaldi-native-fbank ===" << std::endl;
    
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = 16000;
    opts.frame_opts.frame_length_ms = 25;
    opts.frame_opts.frame_shift_ms = 10;
    opts.frame_opts.dither = 1e-5f;
    opts.frame_opts.window_type = "hann";
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.preemph_coeff = 0.0f;
    opts.frame_opts.snip_edges = false;
    
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.low_freq = 0;
    opts.mel_opts.high_freq = 8000;
    
    opts.use_energy = false;
    opts.use_log_fbank = true;
    opts.use_power = true;
    
    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(16000, audio.data(), audio.size());
    fbank.InputFinished();
    
    int32_t num_frames = fbank.NumFramesReady();
    std::cout << "Frames: " << num_frames << std::endl;
    
    if (num_frames > 0) {
        const float* frame0 = fbank.GetFrame(0);
        std::cout << "First frame, first 10 features:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "  [" << i << "] = " << frame0[i] << std::endl;
        }
        
        // Save for comparison
        std::ofstream out("kaldi_features.bin", std::ios::binary);
        for (int t = 0; t < num_frames && t < 125; t++) {
            const float* frame = fbank.GetFrame(t);
            out.write(reinterpret_cast<const char*>(frame), 80 * sizeof(float));
        }
        out.close();
        std::cout << "Saved to kaldi_features.bin" << std::endl;
    }
}

void testImprovedFbank(const std::vector<float>& audio) {
    std::cout << "\n=== Testing ImprovedFbank ===" << std::endl;
    
    improved_fbank::FbankComputer::Options opts;
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;
    opts.frame_length_ms = 25;
    opts.frame_shift_ms = 10;
    opts.n_fft = 512;
    opts.low_freq = 0.0f;
    opts.high_freq = 8000.0f;
    opts.apply_log = true;
    opts.dither = 1e-5f;
    opts.normalize_per_feature = false;  // No normalization
    
    improved_fbank::FbankComputer fbank(opts);
    auto features = fbank.computeFeatures(audio);
    
    std::cout << "Frames: " << features.size() << std::endl;
    
    if (!features.empty()) {
        std::cout << "First frame, first 10 features:" << std::endl;
        for (int i = 0; i < 10 && i < features[0].size(); i++) {
            std::cout << "  [" << i << "] = " << features[0][i] << std::endl;
        }
        
        // Save for comparison
        std::ofstream out("improved_features.bin", std::ios::binary);
        for (size_t t = 0; t < features.size() && t < 125; t++) {
            out.write(reinterpret_cast<const char*>(features[t].data()), 
                     features[t].size() * sizeof(float));
        }
        out.close();
        std::cout << "Saved to improved_features.bin" << std::endl;
    }
}

int main() {
    try {
        std::string wav_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        std::cout << "Loading " << wav_file << std::endl;
        
        auto audio = readWav(wav_file);
        std::cout << "Loaded " << audio.size() << " samples" << std::endl;
        
        // Test both feature extractors
        testKaldiFbank(audio);
        testImprovedFbank(audio);
        
        // Load and compare with Python reference
        std::cout << "\n=== Comparing with Python NeMo reference ===" << std::endl;
        std::ifstream ref_file("nemo_features_no_norm.npy", std::ios::binary);
        if (ref_file) {
            // Skip NumPy header (rough approximation)
            ref_file.seekg(128);  
            
            // Read some features
            std::vector<float> ref_features(80);
            ref_file.read(reinterpret_cast<char*>(ref_features.data()), 80 * sizeof(float));
            
            std::cout << "Python NeMo first frame, first 10 features:" << std::endl;
            for (int i = 0; i < 10; i++) {
                std::cout << "  [" << i << "] = " << ref_features[i] << std::endl;
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}