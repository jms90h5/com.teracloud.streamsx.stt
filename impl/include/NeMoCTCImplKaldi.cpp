#include "NeMoCTCImpl.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

NeMoCTCImpl::NeMoCTCImpl() : initialized_(false), blank_id_(1024) {}

NeMoCTCImpl::~NeMoCTCImpl() = default;

bool NeMoCTCImpl::initialize(const std::string& model_path, const std::string& tokens_path) {
    try {
        // Initialize ONNX Runtime
        env_ = std::make_unique<OnnxIsolated::Env>(ORT_LOGGING_LEVEL_WARNING, "NeMoCTC");
        
        session_options_ = std::make_unique<OnnxIsolated::SessionOptions>();
        session_options_->SetIntraOpNumThreads(4);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        session_ = std::make_unique<OnnxIsolated::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get model metadata
        OnnxIsolated::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();
        
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(input_name.get()));
            
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes_.push_back(tensor_info.GetShape());
        }
        
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name.get()));
            
            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes_.push_back(tensor_info.GetShape());
        }
        
        // Create memory info for tensor creation
        memory_info_ = std::make_unique<OnnxIsolated::MemoryInfo>(
            OnnxIsolated::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );
        
        // Initialize Kaldi feature extractor
        // Match NeMo's configuration exactly
        knf::FbankOptions fbank_opts;
        fbank_opts.frame_opts.samp_freq = 16000;
        fbank_opts.frame_opts.frame_length_ms = 25;
        fbank_opts.frame_opts.frame_shift_ms = 10;
        fbank_opts.frame_opts.dither = 1e-5f;
        fbank_opts.frame_opts.window_type = "hann";
        fbank_opts.frame_opts.remove_dc_offset = true;
        fbank_opts.frame_opts.preemph_coeff = 0.0f;  // No pre-emphasis
        fbank_opts.frame_opts.snip_edges = false;    // Important for matching frame count
        
        fbank_opts.mel_opts.num_bins = 80;
        fbank_opts.mel_opts.low_freq = 0;
        fbank_opts.mel_opts.high_freq = 8000;  // Nyquist for 16kHz
        
        fbank_opts.use_energy = false;
        fbank_opts.use_log_fbank = true;
        fbank_opts.use_power = true;  // Power spectrum
        
        // Note: We're NOT using kaldi's feature extractor directly,
        // but we'll use it to understand the correct parameters
        
        // For now, keep using ImprovedFbank but with corrected parameters
        improved_fbank::FbankComputer::Options options;
        options.sample_rate = 16000;
        options.num_mel_bins = 80;
        options.frame_length_ms = 25;
        options.frame_shift_ms = 10;
        options.n_fft = 512;
        options.low_freq = 0.0f;
        options.high_freq = 8000.0f;
        options.dither = 1e-5f;
        options.apply_log = true;
        options.normalize_per_feature = false;  // Model expects unnormalized features
        
        feature_extractor_ = std::make_unique<improved_fbank::FbankComputer>(options);
        
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
    // Use Kaldi native fbank for feature extraction
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = 16000;
    opts.frame_opts.frame_length_ms = 25;
    opts.frame_opts.frame_shift_ms = 10;
    opts.frame_opts.dither = 1e-5f;
    opts.frame_opts.window_type = "hann";
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.preemph_coeff = 0.0f;
    opts.frame_opts.snip_edges = false;
    
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.low_freq = 0;
    opts.mel_opts.high_freq = 8000;
    
    opts.use_energy = false;
    opts.use_log_fbank = true;
    opts.use_power = true;
    
    knf::FbankComputer fbank(opts);
    
    // Process audio in streaming fashion
    knf::OnlineFeatureInterface* online_fbank = new knf::OnlineFbank(opts);
    
    // Convert audio to the format expected by Kaldi
    std::vector<float> waveform = audio_samples;
    
    // Accept waveform
    online_fbank->AcceptWaveform(16000, waveform.data(), waveform.size());
    online_fbank->InputFinished();
    
    // Extract features
    int32_t num_frames = online_fbank->NumFramesReady();
    int32_t dim = online_fbank->Dim();
    
    std::cout << "Kaldi extracted " << num_frames << " frames with " << dim << " features each" << std::endl;
    
    // Get features in [features, time] format for model
    std::vector<float> features;
    features.resize(num_frames * dim);
    
    // Extract frame by frame and transpose to [features, time]
    for (int32_t t = 0; t < num_frames; t++) {
        std::vector<float> frame(dim);
        online_fbank->GetFrame(t, frame.data());
        
        // Place in [features, time] format
        for (int32_t f = 0; f < dim; f++) {
            features[f * num_frames + t] = frame[f];
        }
    }
    
    delete online_fbank;
    
    return features;
}

std::string NeMoCTCImpl::transcribe(const std::vector<float>& audio_samples) {
    if (!initialized_) {
        return "ERROR: Model not initialized";
    }
    
    try {
        // Extract mel features using Kaldi
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
        
        // Debug: print first few feature values
        std::cout << "First 5 mel features (Kaldi): ";
        for (size_t i = 0; i < 5 && i < mel_features.size(); i++) {
            std::cout << mel_features[i] << " ";
        }
        std::cout << std::endl;
        
        // Prepare input tensors
        // Model expects [batch, features, time] (mel_features is already in [features, time] format)
        std::vector<int64_t> audio_shape = {1, n_mels, n_frames};  // [batch, features, time]
        
        auto audio_tensor = OnnxIsolated::Value::CreateTensor<float>(
            *memory_info_, 
            mel_features.data(),  // Use mel_features directly (already [features, time])
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
    int prev_token_id = -1;
    
    // Debug: print first few predictions
    std::cout << "First 5 time steps predictions:" << std::endl;
    
    for (int t = 0; t < time_steps; t++) {
        // Find argmax for this time step
        int max_idx = 0;
        float max_val = logits[t * vocab_size];
        
        for (int v = 1; v < vocab_size; v++) {
            if (logits[t * vocab_size + v] > max_val) {
                max_val = logits[t * vocab_size + v];
                max_idx = v;
            }
        }
        
        if (t < 5) {
            std::cout << "  Frame " << t << ": max_idx=" << max_idx 
                      << " (blank=" << blank_id_ << "), max_val=" << max_val << std::endl;
        }
        
        // CTC decoding rules:
        // 1. Skip blanks
        // 2. Skip repeated tokens
        if (max_idx != blank_id_ && max_idx != prev_token_id) {
            if (vocab_.find(max_idx) != vocab_.end()) {
                std::string token = vocab_[max_idx];
                
                // Handle BPE tokens with UTF-8 prefix (▁ = \xe2\x96\x81)
                if (token.length() >= 3 && 
                    static_cast<unsigned char>(token[0]) == 0xe2 && 
                    static_cast<unsigned char>(token[1]) == 0x96 && 
                    static_cast<unsigned char>(token[2]) == 0x81) {
                    // This is a word boundary marker, add space before the word
                    if (!result.empty()) {
                        result += " ";
                    }
                    result += token.substr(3);  // Skip the ▁ prefix
                } else {
                    // Regular token, just append
                    result += token;
                }
            }
        }
        
        prev_token_id = max_idx;
    }
    
    return result;
}

std::string NeMoCTCImpl::getModelInfo() const {
    if (!initialized_) {
        return "Model not initialized";
    }
    
    std::stringstream ss;
    ss << "NeMo CTC Model Info:\n";
    ss << "  Inputs: " << input_names_.size() << "\n";
    for (size_t i = 0; i < input_names_.size(); i++) {
        ss << "    " << input_names_[i] << ": [";
        for (size_t j = 0; j < input_shapes_[i].size(); j++) {
            if (j > 0) ss << ", ";
            ss << input_shapes_[i][j];
        }
        ss << "]\n";
    }
    ss << "  Outputs: " << output_names_.size() << "\n";
    ss << "  Vocabulary size: " << vocab_.size() << "\n";
    ss << "  Blank ID: " << blank_id_ << "\n";
    
    return ss.str();
}