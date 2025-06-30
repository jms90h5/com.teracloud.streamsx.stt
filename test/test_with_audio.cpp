#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "../impl/include/NeMoCTCImpl.hpp"

// Simple WAV file reader for 16-bit mono files
bool readWavFile(const std::string& filename, std::vector<float>& audio_data, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    // Read WAV header (simplified - assumes standard 44-byte header)
    char header[44];
    file.read(header, 44);
    
    // Check "RIFF" and "WAVE" signatures
    if (std::memcmp(header, "RIFF", 4) != 0 || std::memcmp(header + 8, "WAVE", 4) != 0) {
        std::cerr << "Not a valid WAV file" << std::endl;
        return false;
    }
    
    // Extract sample rate (bytes 24-27)
    sample_rate = *reinterpret_cast<int*>(header + 24);
    
    // Extract bits per sample (bytes 34-35)
    short bits_per_sample = *reinterpret_cast<short*>(header + 34);
    if (bits_per_sample != 16) {
        std::cerr << "Only 16-bit WAV files are supported" << std::endl;
        return false;
    }
    
    // Read audio data
    std::vector<int16_t> raw_data;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        raw_data.push_back(sample);
    }
    
    // Convert to float [-1, 1]
    audio_data.clear();
    audio_data.reserve(raw_data.size());
    for (int16_t s : raw_data) {
        audio_data.push_back(static_cast<float>(s) / 32768.0f);
    }
    
    std::cout << "Loaded WAV file: " << filename << std::endl;
    std::cout << "  Sample rate: " << sample_rate << " Hz" << std::endl;
    std::cout << "  Duration: " << (float)audio_data.size() / sample_rate << " seconds" << std::endl;
    std::cout << "  Samples: " << audio_data.size() << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=== NeMo CTC Audio Test ===" << std::endl;
    
    // Default paths
    std::string model_path = "../opt/models/fastconformer_ctc_export/model.onnx";
    std::string tokens_path = "../opt/models/fastconformer_ctc_export/tokens.txt";
    std::string audio_path = "../samples/audio/librispeech_3sec.wav";
    
    // Allow audio file as command line argument
    if (argc > 1) {
        audio_path = argv[1];
    }
    
    // Initialize model
    NeMoCTCImpl nemo;
    std::cout << "\nInitializing model..." << std::endl;
    if (!nemo.initialize(model_path, tokens_path)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "\n" << nemo.getModelInfo() << std::endl;
    
    // Load audio file
    std::vector<float> audio_data;
    int sample_rate;
    
    std::cout << "\nLoading audio file..." << std::endl;
    if (!readWavFile(audio_path, audio_data, sample_rate)) {
        std::cerr << "Failed to load audio file" << std::endl;
        return 1;
    }
    
    // Check sample rate
    if (sample_rate != 16000) {
        std::cerr << "WARNING: Model expects 16kHz audio, got " << sample_rate << "Hz" << std::endl;
        // For simplicity, we'll proceed anyway - in production you'd resample
    }
    
    // Transcribe
    std::cout << "\nTranscribing..." << std::endl;
    std::string result = nemo.transcribe(audio_data);
    
    std::cout << "\n=== TRANSCRIPTION ===" << std::endl;
    std::cout << "\"" << result << "\"" << std::endl;
    std::cout << "===================" << std::endl;
    
    // If it's the librispeech sample, show expected result
    if (audio_path.find("librispeech_3sec.wav") != std::string::npos) {
        std::cout << "\nExpected: \"it was the first great song of his life\"" << std::endl;
    }
    
    return 0;
}