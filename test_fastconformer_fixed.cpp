/**
 * Test NVIDIA FastConformer with fixed 500-frame requirement
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>

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

// Simple feature extraction (energy-based for testing)
std::vector<std::vector<float>> extractFeatures(const std::vector<float>& audio, 
                                               int sample_rate = 16000,
                                               int frame_length_ms = 25,
                                               int frame_shift_ms = 10,
                                               int num_features = 80) {
    int frame_length = sample_rate * frame_length_ms / 1000;
    int frame_shift = sample_rate * frame_shift_ms / 1000;
    
    std::vector<std::vector<float>> features;
    
    for (size_t i = 0; i + frame_length <= audio.size(); i += frame_shift) {
        std::vector<float> frame_features(num_features);
        
        // Simple energy computation for each mel bin
        float energy = 0.0f;
        for (size_t j = 0; j < frame_length; j++) {
            energy += audio[i + j] * audio[i + j];
        }
        energy = std::log(energy + 1e-10f);
        
        // Distribute energy across mel bins with some variation
        for (int k = 0; k < num_features; k++) {
            frame_features[k] = energy + (k - 40) * 0.1f;
        }
        
        features.push_back(frame_features);
    }
    
    return features;
}

// Load vocabulary
std::vector<std::string> loadVocabulary(const std::string& vocab_path) {
    std::vector<std::string> vocab;
    std::ifstream file(vocab_path);
    std::string line;
    
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    
    return vocab;
}

// Simple greedy CTC decoding
std::string decodeCTC(const float* log_probs, int seq_len, int vocab_size, 
                      const std::vector<std::string>& vocab) {
    std::vector<int> tokens;
    int prev_token = -1;
    
    for (int t = 0; t < seq_len; t++) {
        // Find argmax
        int best_token = 0;
        float best_score = log_probs[t * vocab_size];
        
        for (int v = 1; v < vocab_size; v++) {
            if (log_probs[t * vocab_size + v] > best_score) {
                best_score = log_probs[t * vocab_size + v];
                best_token = v;
            }
        }
        
        // CTC decoding: skip blanks (token 0) and repeated tokens
        if (best_token != 0 && best_token != prev_token) {
            tokens.push_back(best_token);
        }
        prev_token = best_token;
    }
    
    // Convert tokens to string
    std::string result;
    for (int token : tokens) {
        if (token < vocab.size()) {
            result += vocab[token];
            if (vocab[token].length() > 1) {
                result += " ";
            }
        } else {
            result += "[" + std::to_string(token) + "]";
        }
    }
    
    return result;
}

int main() {
    std::cout << "=== Testing NVIDIA FastConformer (Fixed 500 frames) ===" << std::endl;
    
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastConformerTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_ctc_export/model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        std::cout << "Model loaded successfully" << std::endl;
        
        // Load vocabulary
        std::vector<std::string> vocab = loadVocabulary("models/fastconformer_ctc_export/tokens.txt");
        std::cout << "Vocabulary loaded: " << vocab.size() << " tokens" << std::endl;
        
        // Load test audio
        std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        std::vector<float> audio = loadWavFile(audio_file);
        
        if (audio.empty()) {
            std::cerr << "Failed to load audio file" << std::endl;
            return 1;
        }
        
        std::cout << "Audio loaded: " << audio.size() << " samples (" 
                  << (audio.size() / 16000.0) << " seconds)" << std::endl;
        
        // Extract features
        auto features = extractFeatures(audio);
        std::cout << "Features extracted: " << features.size() << " frames" << std::endl;
        
        // Model requires exactly 500 frames
        const int REQUIRED_FRAMES = 500;
        
        // Process audio in 500-frame chunks
        for (size_t chunk_start = 0; chunk_start < features.size(); chunk_start += REQUIRED_FRAMES) {
            // Prepare exactly 500 frames
            std::vector<float> chunk_data;
            chunk_data.reserve(REQUIRED_FRAMES * 80);
            
            int actual_frames = 0;
            for (int i = 0; i < REQUIRED_FRAMES; i++) {
                if (chunk_start + i < features.size()) {
                    // Use actual feature
                    for (float val : features[chunk_start + i]) {
                        chunk_data.push_back(val);
                    }
                    actual_frames++;
                } else {
                    // Pad with zeros
                    for (int j = 0; j < 80; j++) {
                        chunk_data.push_back(0.0f);
                    }
                }
            }
            
            std::cout << "\nProcessing chunk starting at frame " << chunk_start 
                      << " (" << actual_frames << " actual frames, padded to " 
                      << REQUIRED_FRAMES << ")" << std::endl;
            
            // Create input tensor
            std::vector<int64_t> input_shape = {1, REQUIRED_FRAMES, 80};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, chunk_data.data(), chunk_data.size(),
                input_shape.data(), input_shape.size());
            
            // Run inference
            const char* input_names[] = {"audio_signal"};
            const char* output_names[] = {"log_probs"};
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<Ort::Value> output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Get output
            float* log_probs = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            std::cout << "Output shape: [" << output_shape[0] << ", " 
                      << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
            std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
            
            // Decode
            std::string transcription = decodeCTC(log_probs, output_shape[1], 
                                                 output_shape[2], vocab);
            std::cout << "Transcription: \"" << transcription << "\"" << std::endl;
            
            // Only process first chunk for now
            if (chunk_start == 0) {
                std::cout << "\n(Processing only first chunk for this test)" << std::endl;
                break;
            }
        }
        
        std::cout << "\n✅ Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}