/**
 * Detailed test to understand ONNX inference differences
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <onnxruntime_cxx_api.h>

// Simple NPY loader (same as before)
std::vector<float> loadNpyFile(const std::string& filename, std::vector<int64_t>& shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
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
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get model info
        Ort::AllocatorWithDefaultOptions allocator;
        
        std::cout << "=== Model Information ===" << std::endl;
        
        // Input info
        size_t num_inputs = session.GetInputCount();
        std::cout << "Number of inputs: " << num_inputs << std::endl;
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "Input " << i << ": " << input_name.get() << " shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]" << std::endl;
        }
        
        // Output info
        size_t num_outputs = session.GetOutputCount();
        std::cout << "\nNumber of outputs: " << num_outputs << std::endl;
        std::vector<std::string> output_names_vec;
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names_vec.push_back(std::string(output_name.get()));
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "Output " << i << ": " << output_names_vec[i] << " shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << shape[j];
            }
            std::cout << "]" << std::endl;
        }
        
        // Test 1: With only processed_signal
        std::cout << "\n=== Test 1: Single input (processed_signal only) ===" << std::endl;
        {
            std::vector<int64_t> feature_shape;
            auto features = loadNpyFile("python_features.npy", feature_shape);
            
            // Transpose
            int64_t n_frames = feature_shape[0];
            int64_t n_features = feature_shape[1];
            std::vector<float> transposed(features.size());
            for (int64_t t = 0; t < n_frames; t++) {
                for (int64_t f = 0; f < n_features; f++) {
                    transposed[f * n_frames + t] = features[t * n_features + f];
                }
            }
            
            std::vector<int64_t> input_shape = {1, n_features, n_frames};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, transposed.data(), transposed.size(),
                input_shape.data(), input_shape.size()
            );
            
            try {
                const char* input_names[] = {"processed_signal"};
                std::vector<Ort::Value> input_tensors;
                input_tensors.push_back(std::move(input_tensor));
                
                std::vector<const char*> output_names_cstr;
                for (const auto& name : output_names_vec) {
                    output_names_cstr.push_back(name.c_str());
                }
                
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                                 input_names, input_tensors.data(), 1,
                                                 output_names_cstr.data(), output_names_cstr.size());
                
                std::cout << "Success with single input!" << std::endl;
                auto& logits = output_tensors[0];
                auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
                std::cout << "Output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "Failed with single input: " << e.what() << std::endl;
            }
        }
        
        // Test 2: With both inputs but length = -1
        std::cout << "\n=== Test 2: Both inputs with length = -1 ===" << std::endl;
        {
            std::vector<int64_t> feature_shape;
            auto features = loadNpyFile("python_features.npy", feature_shape);
            
            // Transpose
            int64_t n_frames = feature_shape[0];
            int64_t n_features = feature_shape[1];
            std::vector<float> transposed(features.size());
            for (int64_t t = 0; t < n_frames; t++) {
                for (int64_t f = 0; f < n_features; f++) {
                    transposed[f * n_frames + t] = features[t * n_features + f];
                }
            }
            
            std::vector<int64_t> input_shape = {1, n_features, n_frames};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, transposed.data(), transposed.size(),
                input_shape.data(), input_shape.size()
            );
            
            std::vector<int64_t> length_data = {-1};  // Try -1 for "use all"
            std::vector<int64_t> length_shape = {1};
            auto length_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, length_data.data(), 1,
                length_shape.data(), 1
            );
            
            const char* input_names[] = {"processed_signal", "processed_signal_length"};
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_tensor));
            input_tensors.push_back(std::move(length_tensor));
            
            std::vector<const char*> output_names_cstr;
            for (const auto& name : output_names_vec) {
                output_names_cstr.push_back(name.c_str());
            }
            
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                             input_names, input_tensors.data(), 2,
                                             output_names_cstr.data(), output_names_cstr.size());
            
            std::cout << "Run completed with length=-1" << std::endl;
            
            // Check encoded_lengths output
            if (output_tensors.size() > 1) {
                auto& encoded_lengths = output_tensors[1];
                int64_t* length_ptr = encoded_lengths.GetTensorMutableData<int64_t>();
                std::cout << "Encoded length: " << length_ptr[0] << std::endl;
            }
            
            // Check predictions
            auto& logits = output_tensors[0];
            float* output_data = logits.GetTensorMutableData<float>();
            auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
            size_t vocab_size = shape[2];
            
            std::cout << "First 10 predictions: ";
            for (size_t i = 0; i < 10; i++) {
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
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}