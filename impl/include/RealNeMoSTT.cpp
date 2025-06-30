#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "onnx_wrapper.hpp"
#include <cmath>

struct AudioData {
    std::vector<float> samples;
    int sample_rate;
    int channels;
};

// Load vocabulary from file
std::vector<std::string> loadVocabulary(const std::string& vocab_path) {
    std::vector<std::string> vocab;
    std::ifstream file(vocab_path);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Cannot open vocabulary file: " << vocab_path << std::endl;
        return vocab;
    }
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        vocab.push_back(line);
    }
    
    std::cout << "Loaded " << vocab.size() << " vocabulary tokens" << std::endl;
    return vocab;
}

// Simple CTC decoding
std::string ctcDecode(const std::vector<int>& tokens, const std::vector<std::string>& vocab) {
    std::string result;
    int prev_token = -1;
    
    for (int token : tokens) {
        if (token != prev_token && token > 0 && token < vocab.size()) {
            if (vocab[token] == "|") {
                result += " ";
            } else if (vocab[token] != "<pad>" && vocab[token] != "<s>" && 
                      vocab[token] != "</s>" && vocab[token] != "<unk>") {
                result += vocab[token];
            }
        }
        prev_token = token;
    }
    
    return result;
}

// Load WAV file (simple 16-bit PCM)
AudioData loadWavFile(const std::string& filename) {
    AudioData audio;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Cannot open audio file: " << filename << std::endl;
        return audio;
    }
    
    // Skip WAV header (44 bytes)
    file.seekg(44);
    
    // Read audio data as 16-bit samples
    std::vector<int16_t> raw_samples;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        raw_samples.push_back(sample);
    }
    
    // Convert to float and normalize
    audio.samples.reserve(raw_samples.size());
    for (int16_t raw : raw_samples) {
        audio.samples.push_back(static_cast<float>(raw) / 32768.0f);
    }
    
    audio.sample_rate = 16000; // Assume 16kHz
    audio.channels = 1;        // Assume mono
    
    std::cout << "Loaded audio: " << audio.samples.size() << " samples" << std::endl;
    return audio;
}

int main() {
    try {
        std::cout << "=== Real Wav2Vec2 Model Test ===" << std::endl;
        
        // Load vocabulary
        auto vocab = loadVocabulary("models/wav2vec2_vocab.txt");
        if (vocab.empty()) {
            std::cerr << "Failed to load vocabulary" << std::endl;
            return 1;
        }
        
        // Load audio
        auto audio = loadWavFile("test_data/audio/11-ibm-culture-2min-16k.wav");
        if (audio.samples.empty()) {
            std::cerr << "Failed to load audio" << std::endl;
            return 1;
        }
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        
        // Load model
        Ort::Session session(env, "models/wav2vec2_base.onnx", session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        
        std::cout << "Model loaded. Input: " << input_name.get() << ", Output: " << output_name.get() << std::endl;
        
        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(audio.samples.size())};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, audio.samples.data(), audio.samples.size(), 
            input_shape.data(), input_shape.size());
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        std::vector<const char*> input_names = {input_name.get()};
        std::vector<const char*> output_names = {output_name.get()};
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                         input_names.data(), &input_tensor, 1,
                                         output_names.data(), 1);
        
        // Process output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "Output shape: ";
        for (auto dim : output_shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
        
        // Apply argmax to get token predictions
        int time_steps = output_shape[1];
        int vocab_size = output_shape[2];
        std::vector<int> predicted_tokens;
        
        for (int t = 0; t < time_steps; t++) {
            int best_token = 0;
            float best_score = output_data[t * vocab_size];
            
            for (int v = 1; v < vocab_size; v++) {
                float score = output_data[t * vocab_size + v];
                if (score > best_score) {
                    best_score = score;
                    best_token = v;
                }
            }
            predicted_tokens.push_back(best_token);
        }
        
        // Decode tokens to text
        std::string transcript = ctcDecode(predicted_tokens, vocab);
        
        std::cout << "Raw tokens (first 20): ";
        for (int i = 0; i < std::min(20, static_cast<int>(predicted_tokens.size())); i++) {
            std::cout << predicted_tokens[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Transcript: '" << transcript << "'" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}