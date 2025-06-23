#include "NeMoCTCImpl.hpp"
#include "NeMoCTCImplForward.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstring>

NeMoCTCImpl::NeMoCTCImpl() : initialized_(false), blank_id_(-1) {
}

NeMoCTCImpl::~NeMoCTCImpl() {
}

bool NeMoCTCImpl::initialize(const std::string& model_path, const std::string& tokens_path) {
    try {
        // Initialize ONNX Runtime
        env_ = std::make_unique<OnnxIsolated::Env>(ORT_LOGGING_LEVEL_WARNING, "NeMoCTC");
        session_options_ = std::make_unique<OnnxIsolated::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        memory_info_ = std::make_unique<OnnxIsolated::MemoryInfo>(
            OnnxIsolated::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
        
        // Load model
        session_ = std::make_unique<OnnxIsolated::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get model input/output info
        OnnxIsolated::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);
        input_shapes_.reserve(num_inputs);
        
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(input_name.get()));
            
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes_.push_back(input_shape);
        }
        
        // Output info
        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);
        output_shapes_.reserve(num_outputs);
        
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name.get()));
            
            auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes_.push_back(output_shape);
        }
        
        // Initialize ImprovedFbank feature extractor
        improved_fbank::FbankComputer::Options fbank_opts;
        fbank_opts.sample_rate = 16000;
        fbank_opts.num_mel_bins = 80;
        fbank_opts.frame_length_ms = 25;
        fbank_opts.frame_shift_ms = 10;
        fbank_opts.n_fft = 512;
        fbank_opts.apply_log = true;
        fbank_opts.dither = 1e-5f;
        fbank_opts.normalize_per_feature = false;  // FastConformer has normalize: NA
        
        fbank_computer_ = std::make_unique<improved_fbank::FbankComputer>(fbank_opts);
        std::cout << "Initialized ImprovedFbank feature extractor" << std::endl;
        
        // Load vocabulary
        if (!loadVocabulary(tokens_path)) {
            std::cerr << "Failed to load vocabulary from: " << tokens_path << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "✅ NeMo CTC model initialized successfully" << std::endl;
        std::cout << "Model inputs: " << num_inputs << ", outputs: " << num_outputs << std::endl;
        for (size_t i = 0; i < input_names_.size(); i++) {
            std::cout << "  Input " << i << ": " << input_names_[i] << " shape: [";
            for (size_t j = 0; j < input_shapes_[i].size(); j++) {
                if (j > 0) std::cout << ", ";
                std::cout << input_shapes_[i][j];
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "Vocabulary size: " << vocab_.size() << ", blank_id: " << blank_id_ << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing NeMo CTC model: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCTCImpl::loadVocabulary(const std::string& tokens_path) {
    std::cout << "Attempting to load vocabulary from: " << tokens_path << std::endl;
    
    std::ifstream file(tokens_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open tokens file: " << tokens_path << std::endl;
        std::cerr << "Error: " << strerror(errno) << std::endl;
        return false;
    }
    
    std::string line;
    int id = 0;
    
    // Read tokens - each line is a token, ID is line number
    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocab_[id] = line;
            id++;
        }
    }
    
    // NeMo uses blank token ID 1024
    blank_id_ = 1024;
    
    std::cout << "Loaded " << vocab_.size() << " tokens from vocabulary" << std::endl;
    
    if (vocab_.empty()) {
        std::cerr << "Vocabulary is empty after loading!" << std::endl;
        return false;
    }
    
    return true;
}

std::vector<float> NeMoCTCImpl::extractMelFeatures(const std::vector<float>& audio_samples) {
    // Use ImprovedFbank for feature extraction
    auto features_2d = fbank_computer_->computeFeatures(audio_samples);
    
    std::cout << "ImprovedFbank extracted " << features_2d.size() << " frames with " 
              << (features_2d.empty() ? 0 : features_2d[0].size()) << " features each" << std::endl;
    
    // Flatten to [time * features] format
    std::vector<float> mel_features;
    mel_features.reserve(features_2d.size() * (features_2d.empty() ? 0 : features_2d[0].size()));
    
    for (const auto& frame : features_2d) {
        mel_features.insert(mel_features.end(), frame.begin(), frame.end());
    }
    
    return mel_features;
}

std::string NeMoCTCImpl::transcribe(const std::vector<float>& audio_samples) {
    if (!initialized_) {
        return "ERROR: Model not initialized";
    }
    
    try {
        // Extract mel features
        auto mel_features = extractMelFeatures(audio_samples);
        
        // Calculate dimensions
        int n_mels = 80;
        int n_frames = mel_features.size() / n_mels;
        
        std::cout << "Mel features extracted: " << n_frames << " frames, " 
                  << n_mels << " features per frame" << std::endl;
        
        // Model expects exactly 125 frames - process in chunks
        const int EXPECTED_FRAMES = 125;
        
        if (n_frames < EXPECTED_FRAMES) {
            std::cout << "WARNING: Only " << n_frames << " frames, model expects " 
                      << EXPECTED_FRAMES << ". Padding with zeros." << std::endl;
            
            // Pad with zeros to reach 125 frames
            mel_features.resize(EXPECTED_FRAMES * n_mels, 0.0f);
            n_frames = EXPECTED_FRAMES;
        } else if (n_frames > EXPECTED_FRAMES) {
            std::cout << "Processing first " << EXPECTED_FRAMES << " frames out of " 
                      << n_frames << " total frames" << std::endl;
            
            // Truncate to 125 frames for now
            mel_features.resize(EXPECTED_FRAMES * n_mels);
            n_frames = EXPECTED_FRAMES;
        }
        
        // Debug: print first few feature values and save to file
        std::cout << "First 5 mel features (before reshape): ";
        for (size_t i = 0; i < 5 && i < mel_features.size(); i++) {
            std::cout << mel_features[i] << " ";
        }
        std::cout << std::endl;
        
        // Save features for debugging
        std::ofstream debug_file("cpp_features_debug.bin", std::ios::binary);
        if (debug_file.is_open()) {
            debug_file.write(reinterpret_cast<const char*>(mel_features.data()), 
                           mel_features.size() * sizeof(float));
            debug_file.close();
            std::cout << "Saved " << mel_features.size() << " features to cpp_features_debug.bin" << std::endl;
        }
        
        // Prepare input tensors
        // Model expects [batch, time, features] (mel_features is already in [time, features] format)
        std::vector<int64_t> audio_shape = {1, n_frames, n_mels};  // [batch, time, features]
        
        auto audio_tensor = OnnxIsolated::Value::CreateTensor<float>(
            *memory_info_, 
            mel_features.data(),  // Use mel_features directly (already [time, features])
            mel_features.size(),
            audio_shape.data(), 
            audio_shape.size()
        );
        
        // Prepare input/output arrays
        std::vector<OnnxIsolated::Value> input_tensors;
        input_tensors.push_back(std::move(audio_tensor));
        
        // Only pass length tensor if model expects it (num_inputs > 1)
        if (input_names_.size() > 1) {
            std::vector<int64_t> length_shape = {1};
            std::vector<int64_t> length_data = {n_frames};
            auto length_tensor = OnnxIsolated::Value::CreateTensor<int64_t>(
                *memory_info_,
                length_data.data(),
                length_data.size(),
                length_shape.data(),
                length_shape.size()
            );
            input_tensors.push_back(std::move(length_tensor));
        }
        
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session_->Run(
            OnnxIsolated::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );
        
        // Get output logits
        float* logits_data = output_tensors[0].GetTensorMutableData<float>();
        auto logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Convert to vector for decoding
        size_t logits_size = 1;
        for (auto dim : logits_shape) {
            logits_size *= dim;
        }
        
        std::vector<float> logits_vec(logits_data, logits_data + logits_size);
        
        // Decode CTC output
        return ctcDecode(logits_vec, logits_shape);
        
    } catch (const std::exception& e) {
        return "ERROR: " + std::string(e.what());
    }
}

std::string NeMoCTCImpl::ctcDecode(const std::vector<float>& logits, const std::vector<int64_t>& shape) {
    // Simple CTC decoding: argmax + remove consecutive duplicates + remove blanks
    
    int batch_size = shape[0];
    int time_steps = shape[1]; 
    int vocab_size = shape[2];
    
    std::cout << "CTC decode: batch_size=" << batch_size << ", time_steps=" << time_steps 
              << ", vocab_size=" << vocab_size << std::endl;
    
    std::string result;
    
    // Process first batch only
    int prev_token = -1;
    
    // Debug: Print first few time steps
    std::cout << "First 5 time steps predictions:" << std::endl;
    
    for (int t = 0; t < time_steps; t++) {
        // Find argmax for this time step
        int max_idx = 0;
        float max_val = logits[t * vocab_size];
        
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[t * vocab_size + v];
            if (val > max_val) {
                max_val = val;
                max_idx = v;
            }
        }
        
        // Debug output for first 5 frames
        if (t < 5) {
            std::cout << "  Frame " << t << ": max_idx=" << max_idx 
                      << " (blank=" << blank_id_ << "), max_val=" << max_val;
            if (vocab_.find(max_idx) != vocab_.end()) {
                std::cout << ", token='" << vocab_[max_idx] << "'";
            }
            std::cout << std::endl;
        }
        
        // Skip if same as previous token (remove consecutive duplicates)
        if (max_idx == prev_token) {
            continue;
        }
        
        // Skip blank token
        if (max_idx == blank_id_) {
            prev_token = max_idx;
            continue;
        }
        
        // Add token to result
        if (vocab_.find(max_idx) != vocab_.end()) {
            std::string token = vocab_[max_idx];
            
            // Handle SentencePiece tokens (▁ character is 0xE2 0x96 0x81 in UTF-8)
            if (token.length() >= 3 && 
                (unsigned char)token[0] == 0xE2 && 
                (unsigned char)token[1] == 0x96 && 
                (unsigned char)token[2] == 0x81) {
                result += " " + token.substr(3);  // Remove ▁ and add space
            } else {
                result += token;
            }
        }
        
        prev_token = max_idx;
    }
    
    // Trim leading/trailing spaces
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    
    return result;
}

std::string NeMoCTCImpl::getModelInfo() const {
    if (!initialized_) {
        return "Model not initialized";
    }
    
    std::stringstream info;
    info << "NeMo CTC Model Info:\n";
    info << "Inputs: " << input_names_.size() << "\n";
    for (size_t i = 0; i < input_names_.size(); i++) {
        info << "  " << input_names_[i] << ": [";
        for (size_t j = 0; j < input_shapes_[i].size(); j++) {
            if (j > 0) info << ", ";
            info << input_shapes_[i][j];
        }
        info << "]\n";
    }
    
    info << "Outputs: " << output_names_.size() << "\n";
    for (size_t i = 0; i < output_names_.size(); i++) {
        info << "  " << output_names_[i] << ": [";
        for (size_t j = 0; j < output_shapes_[i].size(); j++) {
            if (j > 0) info << ", ";
            info << output_shapes_[i][j];
        }
        info << "]\n";
    }
    
    info << "Vocabulary size: " << vocab_.size() << "\n";
    info << "Blank token ID: " << blank_id_;
    
    return info.str();
}
// Factory function implementation
std::unique_ptr<NeMoCTCImpl> createNeMoCTCImpl() {
    return std::make_unique<NeMoCTCImpl>();
}
