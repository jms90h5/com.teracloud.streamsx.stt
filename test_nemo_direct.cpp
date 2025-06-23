/**
 * Direct test of NeMoCTCImpl
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "NeMoCTCImpl.hpp"

int main() {
    try {
        std::cout << "=== Testing NeMoCTCImpl Directly ===" << std::endl;
        
        // Create NeMoCTCImpl instance
        NeMoCTCImpl nemo;
        
        // Initialize model
        bool init_result = nemo.initialize(
            "models/fastconformer_nemo_export/ctc_model.onnx",
            "models/fastconformer_nemo_export/tokens.txt"
        );
        
        if (!init_result) {
            std::cerr << "Failed to initialize model" << std::endl;
            return 1;
        }
        
        std::cout << "Model initialized successfully" << std::endl;
        
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
        
        // Convert to float
        std::vector<float> audio_float(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            audio_float[i] = audio_data[i] / 32768.0f;
        }
        
        // Transcribe
        std::cout << "\nTranscribing..." << std::endl;
        std::string result = nemo.transcribe(audio_float);
        
        std::cout << "\nTranscription: '" << result << "'" << std::endl;
        std::cout << "Expected: 'it was the first great sorrow of his life'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}