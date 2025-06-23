/**
 * Test the fixed OnnxSTTImpl with real feature extraction
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "impl/include/OnnxSTTImpl.hpp"

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
    std::cout << "=== Testing Fixed OnnxSTTImpl with Real Feature Extraction ===" << std::endl;
    
    // Configure OnnxSTTImpl - try the dynamic conformer model
    onnx_stt::OnnxSTTImpl::Config config;
    config.encoder_onnx_path = "models/nemo_fastconformer_streaming/conformer_ctc_dynamic.onnx";
    config.vocab_path = "models/nemo_fastconformer_streaming/tokenizer.txt";
    config.sample_rate = 16000;
    config.num_mel_bins = 80;
    config.frame_length_ms = 25;
    config.frame_shift_ms = 10;
    config.num_threads = 4;
    
    // Create and initialize
    onnx_stt::OnnxSTTImpl stt(config);
    
    std::cout << "Initializing OnnxSTTImpl..." << std::endl;
    if (!stt.initialize()) {
        std::cerr << "Failed to initialize OnnxSTTImpl" << std::endl;
        return 1;
    }
    
    std::cout << "✅ OnnxSTTImpl initialized successfully" << std::endl;
    
    // Load test audio
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    std::vector<float> audio = loadWavFile(audio_file);
    
    if (audio.empty()) {
        std::cerr << "Failed to load audio file: " << audio_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded audio: " << audio.size() << " samples (" 
              << (audio.size() / 16000.0) << " seconds)" << std::endl;
    
    // Pad audio to meet minimum requirement if needed
    const size_t required_samples = 80000;  // 5 seconds for 500 frames
    if (audio.size() < required_samples) {
        std::cout << "Padding audio from " << audio.size() << " to " << required_samples << " samples" << std::endl;
        audio.resize(required_samples, 0.0f);  // Pad with silence
    }
    
    // Process audio in chunks - FastConformer needs 500 frames = 80,000 samples  
    const size_t chunk_size = 80000;  // 5 second chunks (enough for 500 frames)
    size_t total_processed = 0;
    
    for (size_t i = 0; i < audio.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, audio.size());
        std::vector<float> chunk(audio.begin() + i, audio.begin() + end);
        
        std::cout << "\nProcessing chunk " << (i / chunk_size + 1) 
                  << " (" << chunk.size() << " samples)..." << std::endl;
        
        // Convert float to int16_t for the API
        std::vector<int16_t> chunk_int16(chunk.size());
        for (size_t j = 0; j < chunk.size(); j++) {
            chunk_int16[j] = static_cast<int16_t>(chunk[j] * 32767.0f);
        }
        
        auto result = stt.processAudioChunk(chunk_int16.data(), chunk_int16.size(), i * 1000 / 16000);  // timestamp in ms
        
        std::cout << "  Result:" << std::endl;
        std::cout << "    Text: \"" << result.text << "\"" << std::endl;
        std::cout << "    Confidence: " << result.confidence << std::endl;
        std::cout << "    Is final: " << (result.is_final ? "yes" : "no") << std::endl;
        std::cout << "    Latency: " << result.latency_ms << "ms" << std::endl;
        
        total_processed += chunk.size();
        
        if (!result.text.empty()) {
            std::cout << "🎉 Non-empty transcription received!" << std::endl;
        }
    }
    
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Total processed: " << total_processed << " samples" << std::endl;
    
    return 0;
}