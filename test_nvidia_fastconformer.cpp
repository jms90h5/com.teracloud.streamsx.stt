/**
 * Test NVIDIA FastConformer Hybrid Large Streaming Multi model
 * This test specifically uses the exported ONNX version of nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
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
    
    std::cout << "WAV file info:" << std::endl;
    std::cout << "  Sample rate: " << header.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << header.channels << std::endl;
    std::cout << "  Bits per sample: " << header.bits_per_sample << std::endl;
    std::cout << "  Data size: " << header.data_size << " bytes" << std::endl;
    
    // Read audio data
    std::vector<int16_t> samples(header.data_size / 2);
    file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
    
    // Convert to float normalized to [-1, 1]
    std::vector<float> audio(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    return audio;
}

int main() {
    std::cout << "=== Testing NVIDIA FastConformer Hybrid Large Streaming Multi ===" << std::endl;
    
    // Configure for NVIDIA FastConformer model
    onnx_stt::OnnxSTTImpl::Config config;
    
    // Use the exported NVIDIA model (44MB)
    config.encoder_onnx_path = "models/fastconformer_ctc_export/model.onnx";
    config.vocab_path = "models/fastconformer_ctc_export/tokens.txt";
    
    // FastConformer model parameters
    config.sample_rate = 16000;
    config.num_mel_bins = 80;
    config.frame_length_ms = 25;
    config.frame_shift_ms = 10;
    config.num_threads = 4;
    
    // Create and initialize
    onnx_stt::OnnxSTTImpl stt(config);
    
    std::cout << "\nInitializing NVIDIA FastConformer model..." << std::endl;
    if (!stt.initialize()) {
        std::cerr << "Failed to initialize OnnxSTTImpl" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model initialized successfully" << std::endl;
    
    // Load test audio
    std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
    std::cout << "\nLoading audio file: " << audio_file << std::endl;
    
    std::vector<float> audio = loadWavFile(audio_file);
    
    if (audio.empty()) {
        std::cerr << "Failed to load audio file: " << audio_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << audio.size() << " samples (" 
              << (audio.size() / 16000.0) << " seconds)" << std::endl;
    
    // Process audio in streaming fashion
    // FastConformer with 8x subsampling needs careful chunk sizing
    const size_t chunk_samples = 16000;  // 1 second chunks
    const size_t overlap_samples = 1600; // 100ms overlap for context
    
    std::cout << "\nProcessing audio in streaming mode..." << std::endl;
    std::cout << "Chunk size: " << chunk_samples << " samples (" 
              << (chunk_samples / 16000.0) << " seconds)" << std::endl;
    std::cout << "Overlap: " << overlap_samples << " samples (" 
              << (overlap_samples / 16000.0) << " seconds)" << std::endl;
    
    size_t position = 0;
    int chunk_num = 0;
    std::string full_transcript;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (position < audio.size()) {
        // Extract chunk with overlap
        size_t chunk_start = (position > overlap_samples) ? position - overlap_samples : 0;
        size_t chunk_end = std::min(position + chunk_samples, audio.size());
        
        std::vector<float> chunk(audio.begin() + chunk_start, audio.begin() + chunk_end);
        
        // Pad if necessary (FastConformer may need minimum chunk size)
        if (chunk.size() < chunk_samples) {
            chunk.resize(chunk_samples, 0.0f);
        }
        
        std::cout << "\nChunk " << chunk_num << ": ";
        std::cout << chunk.size() << " samples, ";
        std::cout << "position " << position << "/" << audio.size() << std::endl;
        
        // Process chunk
        auto chunk_start_time = std::chrono::high_resolution_clock::now();
        
        bool is_final = (position + chunk_samples >= audio.size());
        stt.acceptAudio(chunk, is_final);
        
        // Get transcription
        std::string transcription = stt.getTranscription();
        float confidence = stt.getConfidence();
        
        auto chunk_end_time = std::chrono::high_resolution_clock::now();
        auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end_time - chunk_start_time);
        
        std::cout << "Processing time: " << chunk_duration.count() << "ms" << std::endl;
        std::cout << "Transcription: \"" << transcription << "\"" << std::endl;
        std::cout << "Confidence: " << confidence << std::endl;
        
        // Accumulate transcription
        if (!transcription.empty() && transcription != full_transcript) {
            full_transcript = transcription;
        }
        
        // Move to next chunk
        position += chunk_samples;
        chunk_num++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Full transcription: \"" << full_transcript << "\"" << std::endl;
    std::cout << "Total processing time: " << total_duration.count() << "ms" << std::endl;
    std::cout << "Audio duration: " << (audio.size() / 16000.0 * 1000) << "ms" << std::endl;
    std::cout << "Real-time factor: " << (total_duration.count() / (audio.size() / 16000.0 * 1000)) << std::endl;
    
    // Expected transcription for librispeech-1995-1837-0001.wav:
    // "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"
    
    std::cout << "\n✅ Test completed successfully!" << std::endl;
    
    return 0;
}