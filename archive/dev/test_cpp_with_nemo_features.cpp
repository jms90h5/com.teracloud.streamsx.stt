/**
 * Test C++ with exact NeMo features
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        std::cout << "=== Testing with Exact NeMo Features ===" << std::endl;
        
        // Load the exact NeMo features (first 125 frames)
        std::vector<float> model_input;
        {
            std::ifstream file("nemo_features_125.bin", std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open nemo_features_125.bin");
            }
            
            // 80 features * 125 frames = 10000 floats
            model_input.resize(10000);
            file.read(reinterpret_cast<char*>(model_input.data()), 10000 * sizeof(float));
            std::cout << "Loaded " << model_input.size() << " floats from NeMo features" << std::endl;
        }
        
        // Shape is [80, 125] already in correct format
        std::vector<int64_t> input_shape = {1, 80, 125};
        std::cout << "Input shape: [" << input_shape[0] << ", " << input_shape[1] << ", " << input_shape[2] << "]" << std::endl;
        
        // Print first 5 values to verify
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5; i++) {
            std::cout << model_input[i] << " ";
        }
        std::cout << std::endl;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Create tensors
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, model_input.data(), model_input.size(),
            input_shape.data(), input_shape.size()
        );
        
        std::vector<int64_t> length_data = {125};
        std::vector<int64_t> length_shape = {1};
        auto length_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), 1,
            length_shape.data(), 1
        );
        
        // Get output names
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> output_names_vec;
        size_t num_outputs = session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names_vec.push_back(std::string(output_name.get()));
        }
        
        // Run inference
        const char* input_names[] = {"processed_signal", "processed_signal_length"};
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_vec) {
            output_names_cstr.push_back(name.c_str());
        }
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(length_tensor));
        
        std::cout << "\nRunning inference..." << std::endl;
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                         input_names, input_tensors.data(), input_tensors.size(),
                                         output_names_cstr.data(), output_names_cstr.size());
        
        // Check outputs
        auto& logits = output_tensors[0];
        auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << "]" << std::endl;
        
        if (output_tensors.size() > 1) {
            auto& encoded_lengths = output_tensors[1];
            int64_t* length_ptr = encoded_lengths.GetTensorMutableData<int64_t>();
            std::cout << "Encoded length: " << length_ptr[0] << std::endl;
        }
        
        // Get predictions
        float* output_data = logits.GetTensorMutableData<float>();
        size_t num_frames = logits_shape[1];
        size_t vocab_size = logits_shape[2];
        
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
        
        // Decode
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
        std::cout << "Expected: 'it was the first great sorrow of his life'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}