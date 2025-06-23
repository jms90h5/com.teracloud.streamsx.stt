/**
 * Test C++ ONNX inference using features extracted by Python
 * This isolates the feature extraction issue from the inference issue
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <onnxruntime_cxx_api.h>

// Load binary features saved by Python
std::vector<float> loadBinaryFeatures(const std::string& filename, size_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_floats = file_size / sizeof(float);
    std::cout << "Loading " << num_floats << " floats from " << filename << std::endl;
    
    std::vector<float> features(num_floats);
    file.read(reinterpret_cast<char*>(features.data()), file_size);
    
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
                      const std::vector<std::string>& vocab, int blank_id) {
    std::vector<int> tokens;
    int prev_token = blank_id;
    
    std::cout << "\nFirst few frame predictions:" << std::endl;
    
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
        
        // Debug first few frames
        if (t < 10) {
            std::cout << "  Frame " << t << ": token " << best_token 
                      << " (score: " << best_score << ")";
            if (best_token < vocab.size()) {
                std::cout << " = " << vocab[best_token];
            }
            if (best_token == blank_id) {
                std::cout << " <blank>";
            }
            std::cout << std::endl;
        }
        
        // CTC decoding rules
        if (best_token == blank_id || best_token == prev_token) {
            prev_token = best_token;
            continue;
        }
        
        tokens.push_back(best_token);
        prev_token = best_token;
    }
    
    // Convert tokens to text (handle BPE tokens)
    std::string text;
    for (int token : tokens) {
        if (token < vocab.size()) {
            std::string tok = vocab[token];
            
            // Handle BPE tokens
            if (tok.find("##") == 0) {
                text += tok.substr(2);
            } else if (tok.find("▁") == 0) {
                text += " " + tok.substr(3);  // UTF-8 "▁" is 3 bytes
            } else {
                text += tok;
            }
        }
    }
    
    return text;
}

int main() {
    std::cout << "=== Test C++ ONNX with Python Features ===" << std::endl;
    
    // Load Python features (time x mel format)
    auto features = loadBinaryFeatures("python_features_time_mel.bin", 874 * 80);
    if (features.empty()) {
        std::cerr << "Failed to load features!" << std::endl;
        return 1;
    }
    
    // Verify features
    int n_frames = 874;
    int n_mels = 80;
    std::cout << "Loaded features: " << n_frames << " frames x " << n_mels << " mels" << std::endl;
    
    // Print first few values
    std::cout << "First 10 feature values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "  [" << i << "]: " << features[i] << std::endl;
    }
    
    // Truncate to 125 frames for model
    const int MODEL_FRAMES = 125;
    if (n_frames > MODEL_FRAMES) {
        std::cout << "\nTruncating to " << MODEL_FRAMES << " frames for model" << std::endl;
        features.resize(MODEL_FRAMES * n_mels);
        n_frames = MODEL_FRAMES;
    }
    
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Load model
    std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
    std::cout << "\nLoading model: " << model_path << std::endl;
    Ort::Session session(env, model_path.c_str(), session_options);
    
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    std::cout << "Model has " << num_inputs << " inputs" << std::endl;
    
    for (size_t i = 0; i < num_inputs; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "  Input " << i << ": " << input_name.get() << std::endl;
    }
    
    // Load vocabulary
    auto vocab = loadVocabulary("models/fastconformer_nemo_export/tokens.txt");
    std::cout << "\nVocabulary size: " << vocab.size() << std::endl;
    int blank_id = 1024;  // NeMo uses 1024 for blank
    
    // Create input tensors
    std::vector<int64_t> input_shape = {1, n_frames, n_mels};
    std::vector<int64_t> length_shape = {1};
    std::vector<int64_t> length_data = {n_frames};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Create tensors
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, features.data(), features.size(), 
        input_shape.data(), input_shape.size()));
    
    if (num_inputs > 1) {
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()));
    }
    
    // Get input/output names
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> input_names_allocated;
    for (size_t i = 0; i < num_inputs; i++) {
        input_names_allocated.push_back(session.GetInputNameAllocated(i, allocator));
        input_names.push_back(input_names_allocated.back().get());
    }
    
    std::vector<const char*> output_names = {"output", "output_lengths"};
    
    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                     input_names.data(), 
                                     input_tensors.data(), 
                                     input_tensors.size(),
                                     output_names.data(), 
                                     output_names.size());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << duration << " ms" << std::endl;
    
    // Get output
    float* log_probs = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] 
              << ", " << output_shape[2] << "]" << std::endl;
    
    if (output_tensors.size() > 1) {
        int64_t* lengths = output_tensors[1].GetTensorMutableData<int64_t>();
        std::cout << "Encoded length: " << lengths[0] << std::endl;
    }
    
    // Decode
    std::string transcription = decodeCTC(log_probs, output_shape[1], output_shape[2], 
                                         vocab, blank_id);
    
    std::cout << "\n=== Result ===" << std::endl;
    std::cout << "Transcription: '" << transcription << "'" << std::endl;
    std::cout << "\nExpected: 'it was the first great sorrow of his life...'" << std::endl;
    
    return 0;
}