#include "NeMoCTCModel.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace onnx_stt {

NeMoCTCModel::NeMoCTCModel(const Config& config)
    : config_(config),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      dither_dist_(0.0f, config.dither) {
    std::cout << "NeMoCTCModel constructor - model path: " << config_.model_path << std::endl;
    std::cout << "NeMoCTCModel constructor - vocab path: " << config_.vocab_path << std::endl;
}

NeMoCTCModel::~NeMoCTCModel() = default;

bool NeMoCTCModel::initialize() {
    if (!loadModel()) {
        std::cerr << "Failed to load ONNX model" << std::endl;
        return false;
    }
    
    if (!loadVocabulary()) {
        std::cerr << "Failed to load vocabulary" << std::endl;
        return false;
    }
    
    // Initialize feature extractor
    improved_fbank::FbankComputer::Options fbank_opts;
    fbank_opts.sample_rate = config_.sample_rate;
    fbank_opts.num_mel_bins = config_.n_mels;
    fbank_opts.frame_length_ms = config_.window_size_ms;
    fbank_opts.frame_shift_ms = config_.window_stride_ms;
    fbank_opts.n_fft = config_.n_fft;
    fbank_opts.apply_log = true;  // NeMo expects log mel features
    fbank_opts.dither = config_.dither;
    fbank_opts.normalize_per_feature = false;  // NeMo has normalize: NA
    
    fbank_computer_ = std::make_unique<improved_fbank::FbankComputer>(fbank_opts);
    
    return true;
}

bool NeMoCTCModel::loadModel() {
    try {
        std::cout << "Loading model from path: " << config_.model_path << std::endl;
        
        // Create ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeMoCTC");
        
        // Configure session
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config_.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);
        
        // Get input/output names
        size_t num_inputs = session_->GetInputCount();
        size_t num_outputs = session_->GetOutputCount();
        
        std::cout << "Model loaded: " << num_inputs << " inputs, " << num_outputs << " outputs" << std::endl;
        
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_name_strings_.push_back(std::string(input_name.get()));
            input_names_.push_back(input_name_strings_.back().c_str());
            std::cout << "Input " << i << ": " << input_names_[i] << std::endl;
        }
        
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_name_strings_.push_back(std::string(output_name.get()));
            output_names_.push_back(output_name_strings_.back().c_str());
            std::cout << "Output " << i << ": " << output_names_[i] << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCTCModel::loadVocabulary() {
    try {
        std::ifstream file(config_.vocab_path);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            vocabulary_.push_back(line);
        }
        
        std::cout << "Loaded vocabulary: " << vocabulary_.size() << " tokens" << std::endl;
        return !vocabulary_.empty();
    } catch (const std::exception& e) {
        std::cerr << "Error loading vocabulary: " << e.what() << std::endl;
        return false;
    }
}

void NeMoCTCModel::addDither(std::vector<float>& audio) {
    if (config_.dither > 0) {
        for (auto& sample : audio) {
            sample += dither_dist_(generator_);
        }
    }
}

std::vector<std::vector<float>> NeMoCTCModel::extractFeatures(const std::vector<float>& audio) {
    // Use ImprovedFbank for proper mel-spectrogram extraction
    return fbank_computer_->computeFeatures(audio);
}

NeMoCTCModel::TranscriptionResult NeMoCTCModel::processFeatures(
    const std::vector<std::vector<float>>& features) {
    
    TranscriptionResult result;
    
    try {
        // Prepare input tensor - transpose from [frames, mels] to [mels, frames]
        size_t num_frames = features.size();
        size_t num_mels = config_.n_mels;
        std::vector<float> input_data(num_mels * num_frames);
        
        for (size_t i = 0; i < num_frames; i++) {
            for (size_t j = 0; j < num_mels; j++) {
                input_data[j * num_frames + i] = features[i][j];
            }
        }
        
        // Create input tensors
        std::vector<int64_t> signal_shape = {1, static_cast<int64_t>(num_mels), static_cast<int64_t>(num_frames)};
        std::vector<int64_t> length_shape = {1};
        std::vector<int64_t> length_data = {static_cast<int64_t>(num_frames)};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, input_data.data(), input_data.size(),
            signal_shape.data(), signal_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()));
        
        // Debug output names
        std::cout << "Running inference with " << output_names_.size() << " outputs:" << std::endl;
        for (size_t i = 0; i < output_names_.size(); i++) {
            std::cout << "  Output " << i << ": " << output_names_[i] << std::endl;
        }
        
        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_names_.data(), input_tensors.data(), input_tensors.size(),
                                          output_names_.data(), output_names_.size());
        
        // Process output
        auto& log_probs_tensor = output_tensors[0];
        auto& lengths_tensor = output_tensors[1];
        
        auto log_probs_shape = log_probs_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto output_length = lengths_tensor.GetTensorData<int64_t>()[0];
        
        // Convert to vector for decoding
        const float* log_probs_data = log_probs_tensor.GetTensorData<float>();
        std::vector<std::vector<float>> log_probs;
        
        for (int t = 0; t < output_length; t++) {
            std::vector<float> frame(log_probs_shape[2]);
            for (int v = 0; v < log_probs_shape[2]; v++) {
                frame[v] = log_probs_data[t * log_probs_shape[2] + v];
            }
            log_probs.push_back(frame);
        }
        
        // Decode
        result.text = greedyCTCDecode(log_probs);
        result.num_frames = output_length;
        
        // Calculate average confidence
        float total_confidence = 0.0f;
        for (const auto& frame : log_probs) {
            float max_prob = *std::max_element(frame.begin(), frame.end());
            total_confidence += std::exp(max_prob);
        }
        result.avg_confidence = total_confidence / log_probs.size();
        
    } catch (const std::exception& e) {
        std::cerr << "Error in processFeatures: " << e.what() << std::endl;
    }
    
    return result;
}

std::string NeMoCTCModel::greedyCTCDecode(const std::vector<std::vector<float>>& log_probs) {
    std::vector<int> tokens;
    int prev_token = config_.blank_id;
    
    for (const auto& frame : log_probs) {
        // Find argmax
        int best_token = std::distance(frame.begin(), 
                                      std::max_element(frame.begin(), frame.end()));
        
        // CTC decoding rules
        if (best_token != config_.blank_id && best_token != prev_token) {
            tokens.push_back(best_token);
        }
        prev_token = best_token;
    }
    
    return handleBPETokens(tokens);
}

std::string NeMoCTCModel::handleBPETokens(const std::vector<int>& tokens) {
    std::string text;
    
    for (int token : tokens) {
        if (token < static_cast<int>(vocabulary_.size())) {
            std::string tok = vocabulary_[token];
            
            // Handle BPE tokens with ▁ prefix (UTF-8: \xE2\x96\x81)
            if (tok.length() >= 3 && 
                static_cast<unsigned char>(tok[0]) == 0xE2 && 
                static_cast<unsigned char>(tok[1]) == 0x96 && 
                static_cast<unsigned char>(tok[2]) == 0x81) {
                // Add space and remove ▁ prefix
                if (!text.empty()) text += " ";
                text += tok.substr(3);
            } else {
                // Regular token, append directly
                text += tok;
            }
        }
    }
    
    return text;
}

NeMoCTCModel::TranscriptionResult NeMoCTCModel::processAudio(const std::vector<float>& audio) {
    auto features = extractFeatures(audio);
    return processFeatures(features);
}

} // namespace onnx_stt