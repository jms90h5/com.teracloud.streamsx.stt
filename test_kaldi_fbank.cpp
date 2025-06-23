/**
 * Test Kaldi native fbank feature extraction
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

int main() {
    try {
        std::cout << "=== Testing Kaldi Native FBank ===" << std::endl;
        
        // Load test audio
        std::string wav_path = "test_data/audio/librispeech-1995-1837-0001.wav";
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
        
        // Convert to float
        std::vector<float> audio_float(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            audio_float[i] = audio_data[i] / 32768.0f;
        }
        
        // Configure Kaldi fbank to match NeMo
        knf::FbankOptions opts;
        opts.frame_opts.samp_freq = 16000;
        opts.frame_opts.frame_length_ms = 25;
        opts.frame_opts.frame_shift_ms = 10;
        opts.frame_opts.dither = 1e-5f;
        opts.frame_opts.window_type = "hann";
        opts.frame_opts.remove_dc_offset = true;
        opts.frame_opts.preemph_coeff = 0.0f;  // No pre-emphasis
        opts.frame_opts.snip_edges = false;    // Important for frame count
        
        opts.mel_opts.num_bins = 80;
        opts.mel_opts.low_freq = 0;
        opts.mel_opts.high_freq = 8000;
        
        opts.use_energy = false;
        opts.use_log_fbank = true;
        opts.use_power = true;  // Power spectrum
        
        std::cout << "\nFbank options:" << std::endl;
        std::cout << opts.ToString() << std::endl;
        
        // Create online feature extractor
        knf::OnlineFbank fbank(opts);
        
        // Process audio
        fbank.AcceptWaveform(16000, audio_float.data(), audio_float.size());
        fbank.InputFinished();
        
        int32_t num_frames = fbank.NumFramesReady();
        int32_t dim = fbank.Dim();
        
        std::cout << "\nExtracted " << num_frames << " frames with " << dim << " features each" << std::endl;
        
        // Get first frame
        if (num_frames > 0) {
            const float* frame_ptr = fbank.GetFrame(0);
            
            std::cout << "\nFirst frame (first 5 features): ";
            for (int i = 0; i < 5 && i < dim; i++) {
                std::cout << frame_ptr[i] << " ";
            }
            std::cout << std::endl;
            
            // Compute stats
            float min_val = frame_ptr[0], max_val = frame_ptr[0], sum = 0;
            for (int i = 0; i < dim; i++) {
                min_val = std::min(min_val, frame_ptr[i]);
                max_val = std::max(max_val, frame_ptr[i]);
                sum += frame_ptr[i];
            }
            float mean = sum / dim;
            
            std::cout << "\nFrame stats - Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean << std::endl;
        }
        
        // Save features for comparison - in [features, time] format
        std::vector<float> all_features(dim * std::min(num_frames, 125));
        
        // First collect all frames
        for (int32_t t = 0; t < num_frames && t < 125; t++) {
            const float* frame_ptr = fbank.GetFrame(t);
            for (int32_t f = 0; f < dim; f++) {
                // Store in [time, features] temporarily
                all_features[t * dim + f] = frame_ptr[f];
            }
        }
        
        // Now transpose to [features, time] and save
        std::ofstream out_file("kaldi_features.bin", std::ios::binary);
        int saved_frames = std::min(num_frames, 125);
        for (int32_t f = 0; f < dim; f++) {
            for (int32_t t = 0; t < saved_frames; t++) {
                float val = all_features[t * dim + f];
                out_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        out_file.close();
        
        std::cout << "\nSaved features to kaldi_features.bin" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}