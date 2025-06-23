/**
 * Test ONNX inference using features saved from Python
 * This will help identify if the issue is feature extraction or inference
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <onnxruntime_cxx_api.h>

// Simple NPY file reader for float32 arrays
std::vector<float> loadNpyFile(const std::string& filename, std::vector<int64_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read NPY header
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a valid NPY file");
    }
    
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);
    
    std::string header(header_len, ' ');
    file.read(&header[0], header_len);
    
    // Parse shape from header (simplified - assumes 2D)
    size_t shape_start = header.find("'shape': (") + 10;
    size_t shape_end = header.find(")", shape_start);
    std::string shape_str = header.substr(shape_start, shape_end - shape_start);
    
    // Extract dimensions
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
    
    // Calculate total size
    size_t total_size = 1;
    for (auto dim : shape) {
        total_size *= dim;
    }
    
    // Read data
    std::vector<float> data(total_size);
    file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(float));
    
    return data;
}

int main() {
    try {
        std::cout << "=== Testing ONNX with Python Features ===" << std::endl;
        
        // Load features from Python
        std::vector<int64_t> feature_shape;
        auto features = loadNpyFile("python_features.npy", feature_shape);
        std::cout << "Loaded features shape: [" << feature_shape[0] << ", " << feature_shape[1] << "]" << std::endl;
        
        // Calculate stats
        float min_val = *std::min_element(features.begin(), features.end());
        float max_val = *std::max_element(features.begin(), features.end());
        float mean_val = std::accumulate(features.begin(), features.end(), 0.0f) / features.size();
        std::cout << "Feature stats: min=" << min_val << ", max=" << max_val << ", mean=" << mean_val << std::endl;
        
        // Print first few values
        std::cout << "First 10 feature values: ";
        for (size_t i = 0; i < 10 && i < features.size(); i++) {
            std::cout << features[i] << " ";
        }
        std::cout << std::endl;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get model input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_outputs = session.GetOutputCount();
        std::cout << "Model has " << num_outputs << " outputs:" << std::endl;
        std::vector<std::string> output_names_vec;
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names_vec.push_back(std::string(output_name.get()));
            std::cout << "  Output " << i << ": " << output_names_vec.back() << std::endl;
        }
        
        // Prepare input - features are [time, features], need [batch, features, time]
        int64_t n_frames = feature_shape[0];
        int64_t n_features = feature_shape[1];
        
        // Transpose from [time, features] to [features, time]
        std::vector<float> transposed_features(features.size());
        for (int64_t t = 0; t < n_frames; t++) {
            for (int64_t f = 0; f < n_features; f++) {
                transposed_features[f * n_frames + t] = features[t * n_features + f];
            }
        }
        
        // Create input tensor [batch=1, features, time]
        std::vector<int64_t> input_shape = {1, n_features, n_frames};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, transposed_features.data(), transposed_features.size(),
            input_shape.data(), input_shape.size()
        );
        
        // Create length tensor
        std::vector<int64_t> length_data = {n_frames};
        std::vector<int64_t> length_shape = {1};
        auto length_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()
        );
        
        std::cout << "\nRunning inference..." << std::endl;
        std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]" << std::endl;
        
        // Run inference
        const char* input_names[] = {"processed_signal", "processed_signal_length"};
        const char* output_names[] = {output_names_vec[0].c_str()};
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(length_tensor));
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                         input_names, input_tensors.data(), input_tensors.size(),
                                         output_names, 1);
        
        // Get output
        auto& output_tensor = output_tensors[0];
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
        
        float* output_data = output_tensor.GetTensorMutableData<float>();
        size_t num_frames = output_shape[1];
        size_t vocab_size = output_shape[2];
        
        // Get predictions
        std::cout << "\nFirst 10 predictions: ";
        for (size_t i = 0; i < 10 && i < num_frames; i++) {
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
        
        // Simple CTC decode
        std::vector<std::string> vocab;
        std::ifstream vocab_file("models/fastconformer_nemo_export/tokens.txt");
        std::string token;
        while (std::getline(vocab_file, token)) {
            vocab.push_back(token);
        }
        
        int prev_token = -1;
        std::string text;
        for (size_t i = 0; i < num_frames; i++) {
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
                // Handle BPE tokens with ▁ prefix (UTF-8: \xE2\x96\x81)
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
        
        // Load and compare with Python logits
        std::vector<int64_t> logits_shape;
        auto python_logits = loadNpyFile("python_logits.npy", logits_shape);
        
        // Compare first few values
        std::cout << "\n=== Comparing with Python logits ===" << std::endl;
        bool match = true;
        for (size_t i = 0; i < 10; i++) {
            float cpp_val = output_data[i];
            float py_val = python_logits[i];
            float diff = std::abs(cpp_val - py_val);
            if (diff > 1e-4) {
                match = false;
                std::cout << "Mismatch at " << i << ": C++=" << cpp_val << ", Python=" << py_val << ", diff=" << diff << std::endl;
            }
        }
        if (match) {
            std::cout << "Logits match Python!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}