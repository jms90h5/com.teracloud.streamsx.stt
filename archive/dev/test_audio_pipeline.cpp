/**
 * Test audio pipeline to debug why features are all zeros
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
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
    
    std::cout << "WAV file info:" << std::endl;
    std::cout << "  Sample rate: " << header.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << header.channels << std::endl;
    std::cout << "  Bits per sample: " << header.bits_per_sample << std::endl;
    std::cout << "  Data size: " << header.data_size << " bytes" << std::endl;
    
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
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    
    // Load audio
    std::vector<float> audio = loadWavFile(audio_file);
    if (audio.empty()) {
        return 1;
    }
    
    std::cout << "\nAudio statistics:" << std::endl;
    float min_val = *std::min_element(audio.begin(), audio.end());
    float max_val = *std::max_element(audio.begin(), audio.end());
    float avg_val = 0;
    for (float s : audio) avg_val += std::abs(s);
    avg_val /= audio.size();
    
    std::cout << "  Samples: " << audio.size() << std::endl;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Average abs value: " << avg_val << std::endl;
    
    // Test first few samples
    std::cout << "\nFirst 10 samples:" << std::endl;
    for (int i = 0; i < 10 && i < audio.size(); i++) {
        std::cout << "  [" << i << "] = " << audio[i] << std::endl;
    }
    
    // Create feature extractor
    improved_fbank::FbankComputer::Options opts;
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;
    opts.frame_length_ms = 25;
    opts.frame_shift_ms = 10;
    opts.apply_log = true;
    opts.dither = 1e-5f;
    
    improved_fbank::FbankComputer fbank(opts);
    
    // Extract features from first second
    std::vector<float> first_second(audio.begin(), audio.begin() + std::min<size_t>(16000, audio.size()));
    auto features = fbank.computeFeatures(first_second);
    
    std::cout << "\nFeature extraction results:" << std::endl;
    std::cout << "  Number of frames: " << features.size() << std::endl;
    
    if (!features.empty()) {
        std::cout << "\nFirst frame features:" << std::endl;
        float frame_min = *std::min_element(features[0].begin(), features[0].end());
        float frame_max = *std::max_element(features[0].begin(), features[0].end());
        float frame_avg = 0;
        for (float f : features[0]) frame_avg += f;
        frame_avg /= features[0].size();
        
        std::cout << "  Min: " << frame_min << std::endl;
        std::cout << "  Max: " << frame_max << std::endl;
        std::cout << "  Average: " << frame_avg << std::endl;
        
        std::cout << "\nFirst 10 features of first frame:" << std::endl;
        for (int i = 0; i < 10 && i < features[0].size(); i++) {
            std::cout << "  [" << i << "] = " << features[0][i] << std::endl;
        }
    }
    
    return 0;
}