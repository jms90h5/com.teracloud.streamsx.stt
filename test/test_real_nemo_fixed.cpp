/**
 * Standalone C++ test application for NVidia NeMo cache-aware Conformer
 * 
 * This demonstrates the NeMo STT pipeline without SPL dependencies:
 * - Direct C++ API usage
 * - NeMo FastConformer streaming model
 * - Voice Activity Detection
 * - Performance benchmarking
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include <getopt.h>
#include <iomanip>

// Include the STT pipeline headers
#include "../../impl/include/STTPipeline.hpp"

using namespace onnx_stt;

struct TestConfig {
    std::string nemo_model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
    std::string audio_file_path = "test_data/audio/librispeech-1995-1837-0001.wav";
    bool enable_vad = true;
    bool verbose = false;
    int chunk_size_ms = 160;  // 160ms chunks for NeMo
    int sample_rate = 16000;
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --nemo-model PATH     Path to NeMo ONNX model file\n"
              << "  --audio-file PATH     Path to raw audio file (16kHz, 16-bit, mono)\n"
              << "  --enable-vad          Enable Voice Activity Detection (default: true)\n"
              << "  --disable-vad         Disable Voice Activity Detection\n"
              << "  --chunk-size MS       Chunk size in milliseconds (default: 160)\n"
              << "  --verbose             Enable verbose output\n"
              << "  --help                Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " --nemo-model models/fastconformer.onnx --audio-file test.raw\n"
              << std::endl;
}

bool parseArguments(int argc, char* argv[], TestConfig& config) {
    static struct option long_options[] = {
        {"nemo-model", required_argument, 0, 'm'},
        {"audio-file", required_argument, 0, 'a'},
        {"enable-vad", no_argument, 0, 'v'},
        {"disable-vad", no_argument, 0, 'V'},
        {"chunk-size", required_argument, 0, 'c'},
        {"verbose", no_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "m:a:vVc:rh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'm':
                config.nemo_model_path = optarg;
                break;
            case 'a':
                config.audio_file_path = optarg;
                break;
            case 'v':
                config.enable_vad = true;
                break;
            case 'V':
                config.enable_vad = false;
                break;
            case 'c':
                config.chunk_size_ms = std::atoi(optarg);
                break;
            case 'r':
                config.verbose = true;
                break;
            case 'h':
                printUsage(argv[0]);
                return false;
            case '?':
                std::cerr << "Unknown option. Use --help for usage information." << std::endl;
                return false;
            default:
                return false;
        }
    }
    
    return true;
}

std::vector<int16_t> loadAudioFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open audio file: " + filename);
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read audio samples (16-bit signed)
    std::vector<int16_t> audio_data(file_size / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(audio_data.data()), file_size);
    
    if (!file) {
        throw std::runtime_error("Failed to read audio file completely");
    }
    
    std::cout << "Loaded audio file: " << filename << std::endl;
    std::cout << "  Size: " << file_size << " bytes" << std::endl;
    std::cout << "  Samples: " << audio_data.size() << std::endl;
    std::cout << "  Duration: " << (audio_data.size() / 16000.0) << " seconds" << std::endl;
    
    return audio_data;
}

int main(int argc, char* argv[]) {
    TestConfig config;
    
    std::cout << "=== NeMo Cache-Aware Conformer Standalone Test ===" << std::endl;
    
    // Parse command line arguments
    if (!parseArguments(argc, argv, config)) {
        return 1;
    }
    
    try {
        // Display configuration
        std::cout << "\nConfiguration:" << std::endl;
        std::cout << "  NeMo model: " << config.nemo_model_path << std::endl;
        std::cout << "  Audio file: " << config.audio_file_path << std::endl;
        std::cout << "  VAD enabled: " << (config.enable_vad ? "yes" : "no") << std::endl;
        std::cout << "  Chunk size: " << config.chunk_size_ms << "ms" << std::endl;
        std::cout << "  Sample rate: " << config.sample_rate << "Hz" << std::endl;
        
        // Load audio file
        std::cout << "\nLoading audio file..." << std::endl;
        auto audio_data = loadAudioFile(config.audio_file_path);
        
        // Create NeMo pipeline
        std::cout << "\nInitializing NeMo pipeline..." << std::endl;
        auto pipeline = createNeMoPipeline(config.nemo_model_path, config.enable_vad);
        
        if (!pipeline) {
            std::cerr << "Failed to create NeMo pipeline" << std::endl;
            return 1;
        }
        
        std::cout << "NeMo pipeline initialized successfully!" << std::endl;
        
        // Calculate chunk parameters
        size_t chunk_samples = (config.sample_rate * config.chunk_size_ms) / 1000;
        size_t total_chunks = (audio_data.size() + chunk_samples - 1) / chunk_samples;
        
        std::cout << "\nProcessing parameters:" << std::endl;
        std::cout << "  Chunk samples: " << chunk_samples << std::endl;
        std::cout << "  Total chunks: " << total_chunks << std::endl;
        
        // Process audio in chunks
        std::cout << "\n=== Processing Audio ===" << std::endl;
        
        uint64_t total_latency = 0;
        uint64_t speech_chunks = 0;
        uint64_t silence_chunks = 0;
        uint64_t min_latency = UINT64_MAX;
        uint64_t max_latency = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < total_chunks; ++i) {
            // Extract chunk
            size_t start_sample = i * chunk_samples;
            size_t end_sample = std::min(start_sample + chunk_samples, audio_data.size());
            size_t current_chunk_size = end_sample - start_sample;
            
            std::vector<float> chunk_audio(current_chunk_size);
            for (size_t j = 0; j < current_chunk_size; ++j) {
                chunk_audio[j] = static_cast<float>(audio_data[start_sample + j]) / 32768.0f;
            }
            
            // Process chunk
            uint64_t timestamp_ms = (start_sample * 1000) / config.sample_rate;
            auto result = pipeline->processAudio(chunk_audio, timestamp_ms);
            
            // Update statistics
            total_latency += result.latency_ms;
            if (result.latency_ms < min_latency) min_latency = result.latency_ms;
            if (result.latency_ms > max_latency) max_latency = result.latency_ms;
            
            if (result.speech_detected) {
                speech_chunks++;
            } else {
                silence_chunks++;
            }
            
            // Display result
            if (config.verbose || !result.text.empty()) {
                std::cout << "[" << timestamp_ms << "ms] ";
                std::cout << "[" << (result.speech_detected ? "SPEECH" : "SILENCE") << "] ";
                std::cout << result.text;
                std::cout << " (conf: " << (result.confidence * 100.0) << "%)";
                std::cout << " [VAD:" << result.vad_latency_ms << "ms";
                std::cout << " FE:" << result.feature_latency_ms << "ms"; 
                std::cout << " MODEL:" << result.model_latency_ms << "ms";
                std::cout << " TOTAL:" << result.latency_ms << "ms]";
                std::cout << std::endl;
            }
            
            // Progress indicator
            if ((i + 1) % 50 == 0 || i == total_chunks - 1) {
                double progress = (double)(i + 1) / total_chunks * 100.0;
                std::cout << "Progress: " << (i + 1) << "/" << total_chunks 
                          << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Final statistics
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << "Total processing time: " << total_duration.count() << "ms" << std::endl;
        std::cout << "Audio duration: " << (audio_data.size() / 16000.0 * 1000.0) << "ms" << std::endl;
        std::cout << "Real-time factor: " << (total_duration.count() / (audio_data.size() / 16000.0 * 1000.0)) << std::endl;
        std::cout << "Speech chunks: " << speech_chunks << " (" << (speech_chunks * 100.0 / total_chunks) << "%)" << std::endl;
        std::cout << "Silence chunks: " << silence_chunks << " (" << (silence_chunks * 100.0 / total_chunks) << "%)" << std::endl;
        
        if (total_chunks > 0) {
            std::cout << "Average latency: " << (total_latency / total_chunks) << "ms" << std::endl;
            std::cout << "Latency range: " << min_latency << "-" << max_latency << "ms" << std::endl;
        }
        
        auto pipeline_stats = pipeline->getStats();
        std::cout << "\nPipeline Statistics:" << std::endl;
        std::cout << "  Total chunks processed: " << pipeline_stats.total_chunks_processed << std::endl;
        std::cout << "  Average VAD latency: " << pipeline_stats.avg_vad_latency_ms << "ms" << std::endl;
        std::cout << "  Average feature latency: " << pipeline_stats.avg_feature_latency_ms << "ms" << std::endl;
        std::cout << "  Average model latency: " << pipeline_stats.avg_model_latency_ms << "ms" << std::endl;
        std::cout << "  Real-time factor: " << pipeline_stats.real_time_factor << std::endl;
        
        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}