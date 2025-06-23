// Direct test of ImprovedFbank vs Kaldi features
#include <iostream>
#include <vector>
#include <fstream>
#include "ImprovedFbank.hpp"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace knf = ::knf;

int main() {
    // Load test audio
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    std::ifstream wav_file(audio_file, std::ios::binary);
    if (!wav_file) {
        std::cerr << "Failed to open " << audio_file << std::endl;
        return 1;
    }
    
    // Skip WAV header (44 bytes)
    wav_file.seekg(44);
    
    // Read audio samples
    std::vector<int16_t> audio_int16;
    int16_t sample;
    while (wav_file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        audio_int16.push_back(sample);
    }
    wav_file.close();
    
    // Convert to float
    std::vector<float> audio_float;
    audio_float.reserve(audio_int16.size());
    for (int16_t s : audio_int16) {
        audio_float.push_back(s / 32768.0f);
    }
    
    std::cout << "Loaded " << audio_float.size() << " audio samples" << std::endl;
    
    // Test 1: Kaldi features
    {
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
        fbank.AcceptWaveform(16000, audio_float.data(), audio_float.size());
        fbank.InputFinished();
        
        int32_t num_frames = fbank.NumFramesReady();
        std::cout << "\nKaldi features: " << num_frames << " frames" << std::endl;
        
        if (num_frames > 0) {
            const float* frame0 = fbank.GetFrame(0);
            std::cout << "  First 5 features: ";
            for (int i = 0; i < 5; i++) {
                std::cout << frame0[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Test 2: ImprovedFbank features
    {
        improved_fbank::FbankComputer::Options opts;
        opts.sample_rate = 16000;
        opts.num_mel_bins = 80;
        opts.frame_length_ms = 25;
        opts.frame_shift_ms = 10;
        opts.n_fft = 512;
        opts.apply_log = true;
        opts.dither = 1e-5f;
        opts.normalize_per_feature = false;
        
        improved_fbank::FbankComputer fbank(opts);
        auto features_2d = fbank.computeFeatures(audio_float);
        
        std::cout << "\nImprovedFbank features: " << features_2d.size() << " frames" << std::endl;
        
        if (!features_2d.empty() && !features_2d[0].empty()) {
            std::cout << "  First 5 features: ";
            for (int i = 0; i < 5 && i < features_2d[0].size(); i++) {
                std::cout << features_2d[0][i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    return 0;
}