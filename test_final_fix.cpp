/**
 * Final test with correct processed_signal_length
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <onnxruntime_cxx_api.h>

// NPY loader
std::vector<float> loadNpyFile(const std::string& filename, std::vector<int64_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    char magic[6];
    file.read(magic, 6);
    
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);
    
    size_t shape_start = header.find("'shape': (") + 10;
    size_t shape_end = header.find(")", shape_start);
    std::string shape_str = header.substr(shape_start, shape_end - shape_start);
    
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.length()) {
        size_t comma = shape_str.find(",", pos);
        if (comma == std::string::npos) comma = shape_str.length();
        std::string dim_str = shape_str.substr(pos, comma - pos);
        if (!dim_str.empty() && dim_str != " ") {
            shape.push_back(std::stoll(dim_str));
        }
        pos = comma + 1;
    }
    
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    
    return data;
}

int main() {
    try {
        std::cout << "=== Testing with Correct processed_signal_length ===" << std::endl;
        
        // Load Python features
        std::vector<int64_t> feature_shape;
        auto features = loadNpyFile("python_features.npy", feature_shape);
        std::cout << "Loaded features shape: [" << feature_shape[0] << ", " << feature_shape[1] << "]" << std::endl;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get output names
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> output_names_vec;
        size_t num_outputs = session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names_vec.push_back(std::string(output_name.get()));
        }
        
        // Prepare input - features are [time, features], need [batch, features, time]
        int64_t n_frames = feature_shape[0];
        int64_t n_features = feature_shape[1];
        
        std::cout << "Input frames: " << n_frames << ", features: " << n_features << std::endl;
        
        // Transpose from [time, features] to [features, time]
        std::vector<float> transposed_features(features.size());
        for (int64_t t = 0; t < n_frames; t++) {
            for (int64_t f = 0; f < n_features; f++) {
                transposed_features[f * n_frames + t] = features[t * n_features + f];
            }
        }
        
        // Create input tensors
        std::vector<int64_t> input_shape = {1, n_features, n_frames};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, transposed_features.data(), transposed_features.size(),
            input_shape.data(), input_shape.size()
        );
        
        // CRITICAL: processed_signal_length should be the number of time frames!
        std::vector<int64_t> length_data = {n_frames};  // 874, not -1
        std::vector<int64_t> length_shape = {1};
        auto length_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()
        );
        
        std::cout << "\nRunning inference..." << std::endl;
        std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]" << std::endl;
        std::cout << "processed_signal_length: " << length_data[0] << std::endl;
        
        // Run inference
        const char* input_names[] = {"processed_signal", "processed_signal_length"};
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_vec) {
            output_names_cstr.push_back(name.c_str());
        }
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(length_tensor));
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                         input_names, input_tensors.data(), input_tensors.size(),
                                         output_names_cstr.data(), output_names_cstr.size());
        
        // Check outputs
        auto& logits = output_tensors[0];
        auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\nOutput shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << "]" << std::endl;
        
        if (output_tensors.size() > 1) {
            auto& encoded_lengths = output_tensors[1];
            int64_t* length_ptr = encoded_lengths.GetTensorMutableData<int64_t>();
            std::cout << "Encoded length: " << length_ptr[0] << std::endl;
        }
        
        // Get predictions
        float* output_data = logits.GetTensorMutableData<float>();
        size_t num_frames_out = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
        std::cout << "\nFirst 10 predictions: ";
        for (size_t i = 0; i < 10 && i < num_frames_out; i++) {
            int max_idx = 0;
            float max_val = output_data[i * vocab_size];
            for (size_t j = 1; j < vocab_size; j++) {
                if (output_data[i * vocab_size + j] > max_val) {
                    max_val = output_data[i * vocab_size + j];
                    max_idx = j;
                }
            }
            std::cout << max_idx << " ";
        }
        std::cout << std::endl;
        
        // Compare with Python logits
        std::vector<int64_t> py_shape;
        auto python_logits = loadNpyFile("python_logits.npy", py_shape);
        
        std::cout << "\n=== Comparing with Python ===" << std::endl;
        float total_diff = 0;
        int num_compared = 0;
        for (size_t i = 0; i < 10 && i < python_logits.size(); i++) {
            float cpp_val = output_data[i];
            float py_val = python_logits[i];
            float diff = std::abs(cpp_val - py_val);
            total_diff += diff;
            num_compared++;
            if (diff > 0.001) {
                std::cout << "  Position " << i << ": C++=" << cpp_val << ", Python=" << py_val << ", diff=" << diff << std::endl;
            }
        }
        std::cout << "Average difference: " << (total_diff / num_compared) << std::endl;
        
        // Decode
        std::vector<std::string> vocab;
        std::ifstream vocab_file("models/fastconformer_nemo_export/tokens.txt");
        std::string token;
        while (std::getline(vocab_file, token)) {
            vocab.push_back(token);
        }
        
        int prev_token = -1;
        std::string text;
        for (size_t i = 0; i < num_frames_out; i++) {
            int max_idx = 0;
            float max_val = output_data[i * vocab_size];
            for (size_t j = 1; j < vocab_size; j++) {
                if (output_data[i * vocab_size + j] > max_val) {
                    max_val = output_data[i * vocab_size + j];
                    max_idx = j;
                }
            }
            
            if (max_idx != 1024 && max_idx != prev_token && max_idx < vocab.size()) {
                std::string tok = vocab[max_idx];
                // Handle BPE tokens
                if (tok.length() >= 3 && 
                    static_cast<unsigned char>(tok[0]) == 0xE2 && 
                    static_cast<unsigned char>(tok[1]) == 0x96 && 
                    static_cast<unsigned char>(tok[2]) == 0x81) {
                    if (!text.empty()) text += " ";
                    text += tok.substr(3);
                } else {
                    text += tok;
                }
            }
            prev_token = max_idx;
        }
        
        std::cout << "\nTranscription: '" << text << "'" << std::endl;
        std::cout << "Expected: 'it was the first great sorrow of his life...'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}