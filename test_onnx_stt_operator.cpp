/**
 * Test program for OnnxSTT operator functionality
 * This tests the actual implementation used by Streams
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include "impl/include/OnnxSTTInterface.hpp"

// Simple WAV reader
std::vector<int16_t> readWavFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Skip WAV header (44 bytes)
    file.seekg(44);
    
    // Read samples
    std::vector<int16_t> samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        samples.push_back(sample);
    }
    
    return samples;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <tokens.txt> <audio.wav>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string vocab_path = argv[2];
    std::string audio_path = argv[3];
    
    try {
        std::cout << "=== Testing OnnxSTT Operator Implementation ===" << std::endl;
        
        // Configure OnnxSTT
        onnx_stt::OnnxSTTInterface::Config config;
        config.encoder_onnx_path = model_path;
        config.vocab_path = vocab_path;
        config.cmvn_stats_path = "none";
        config.sample_rate = 16000;
        config.chunk_size_ms = 100;
        config.num_threads = 4;
        config.use_gpu = false;
        config.model_type = onnx_stt::OnnxSTTInterface::ModelType::NEMO_CTC;
        config.blank_id = 1024;  // NeMo CTC blank
        
        std::cout << "Creating OnnxSTT instance..." << std::endl;
        auto onnx_impl = onnx_stt::createOnnxSTT(config);
        
        std::cout << "Initializing..." << std::endl;
        if (!onnx_impl->initialize()) {
            std::cerr << "Failed to initialize OnnxSTT" << std::endl;
            return 1;
        }
        
        // Read audio
        std::cout << "Reading audio: " << audio_path << std::endl;
        auto samples = readWavFile(audio_path);
        std::cout << "Loaded " << samples.size() << " samples" << std::endl;
        
        // Process audio
        std::cout << "Processing audio..." << std::endl;
        auto result = onnx_impl->processAudioChunk(samples.data(), samples.size(), 0);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Text: " << result.text << std::endl;
        std::cout << "Is Final: " << (result.is_final ? "yes" : "no") << std::endl;
        std::cout << "Confidence: " << result.confidence << std::endl;
        std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
        
        // Get stats
        auto stats = onnx_impl->getStats();
        std::cout << "\n=== Performance ===" << std::endl;
        std::cout << "Total audio: " << stats.total_audio_ms << " ms" << std::endl;
        std::cout << "Total processing: " << stats.total_processing_ms << " ms" << std::endl;
        std::cout << "Real-time factor: " << stats.real_time_factor << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}