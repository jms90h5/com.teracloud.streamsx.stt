#include "SileroVAD.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace onnx_stt {

// Silero VAD Implementation
SileroVAD::SileroVAD(const Config& config) : config_(config) {
    window_size_samples_ = (config_.window_size_ms * config_.sample_rate) / 1000;
    frame_shift_samples_ = (config_.frame_shift_ms * config_.sample_rate) / 1000;
}

bool SileroVAD::initialize(const Config& config) {
    config_ = config;
    window_size_samples_ = (config_.window_size_ms * config_.sample_rate) / 1000;
    frame_shift_samples_ = (config_.frame_shift_ms * config_.sample_rate) / 1000;
    
    try {
        // Initialize ONNX Runtime
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SileroVAD");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Try to load Silero VAD model
        std::string model_path = "../models/silero_vad.onnx";
        if (!loadModel(model_path)) {
            std::cerr << "Warning: Could not load Silero VAD model from " << model_path 
                     << ". VAD will be disabled." << std::endl;
            return false;
        }
        
        // Initialize state tensors (for LSTM layers)
        state_h_.resize(2 * 64, 0.0f); // 2 layers, 64 hidden units
        state_c_.resize(2 * 64, 0.0f);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Silero VAD: " << e.what() << std::endl;
        return false;
    }
}

bool SileroVAD::loadModel(const std::string& model_path) {
    try {
        // Try to create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get input/output metadata
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names and shapes
        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);
        input_shapes_.reserve(num_inputs);
        
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.emplace_back(input_name.get());
            
            auto input_shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes_.emplace_back(input_shape);
        }
        
        // Output names and shapes  
        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);
        output_shapes_.reserve(num_outputs);
        
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.emplace_back(output_name.get());
            
            auto output_shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes_.emplace_back(output_shape);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load Silero VAD model: " << e.what() << std::endl;
        return false;
    }
}

VADInterface::VADResult SileroVAD::processChunk(const int16_t* samples, 
                                               size_t num_samples, 
                                               uint64_t timestamp_ms) {
    std::vector<float> audio_float;
    convertInt16ToFloat(samples, num_samples, audio_float);
    return processChunk(audio_float, timestamp_ms);
}

VADInterface::VADResult SileroVAD::processChunk(const std::vector<float>& audio, 
                                               uint64_t timestamp_ms) {
    if (!session_) {
        // Fallback: assume all audio is speech if no model loaded
        return {true, 1.0f, timestamp_ms};
    }
    
    // Add to buffer
    audio_buffer_.insert(audio_buffer_.end(), audio.begin(), audio.end());
    
    VADResult result = {false, 0.0f, timestamp_ms};
    
    // Process complete windows
    while (audio_buffer_.size() >= static_cast<size_t>(window_size_samples_)) {
        std::vector<float> window(audio_buffer_.begin(), 
                                 audio_buffer_.begin() + window_size_samples_);
        
        auto window_result = runInference(window, timestamp_ms);
        
        // Use the latest result (could accumulate/average multiple windows)
        result = window_result;
        
        // Shift buffer by frame_shift_samples_
        audio_buffer_.erase(audio_buffer_.begin(), 
                           audio_buffer_.begin() + frame_shift_samples_);
    }
    
    return result;
}

VADInterface::VADResult SileroVAD::runInference(const std::vector<float>& audio_chunk, 
                                               uint64_t timestamp_ms) {
    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names_ptr;
        
        for (size_t i = 0; i < input_names_.size(); ++i) {
            input_names_ptr.push_back(input_names_[i].c_str());
        }
        
        // Input audio tensor [1, window_size]
        std::vector<int64_t> audio_shape = {1, static_cast<int64_t>(audio_chunk.size())};
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(audio_chunk.data()), 
            audio_chunk.size(), audio_shape.data(), audio_shape.size()));
        
        // State tensors (h and c for LSTM)
        std::vector<int64_t> state_shape = {2, 1, 64}; // [layers, batch, hidden]
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, state_h_.data(), state_h_.size(), 
            state_shape.data(), state_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, state_c_.data(), state_c_.size(), 
            state_shape.data(), state_shape.size()));
        
        // Prepare output tensors
        std::vector<const char*> output_names_ptr;
        for (const auto& name : output_names_) {
            output_names_ptr.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names_ptr.data(), input_tensors.data(), input_tensors.size(),
                                          output_names_ptr.data(), output_names_ptr.size());
        
        // Extract speech probability
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        float speech_prob = output_data[0];
        
        // Update states if available
        if (output_tensors.size() > 1) {
            float* new_h = output_tensors[1].GetTensorMutableData<float>();
            std::copy(new_h, new_h + state_h_.size(), state_h_.begin());
        }
        if (output_tensors.size() > 2) {
            float* new_c = output_tensors[2].GetTensorMutableData<float>();
            std::copy(new_c, new_c + state_c_.size(), state_c_.begin());
        }
        
        return {speech_prob > config_.speech_threshold, speech_prob, timestamp_ms};
        
    } catch (const std::exception& e) {
        std::cerr << "Silero VAD inference failed: " << e.what() << std::endl;
        return {true, 1.0f, timestamp_ms}; // Fallback: assume speech
    }
}

void SileroVAD::reset() {
    std::fill(state_h_.begin(), state_h_.end(), 0.0f);
    std::fill(state_c_.begin(), state_c_.end(), 0.0f);
    audio_buffer_.clear();
}

void SileroVAD::convertInt16ToFloat(const int16_t* samples, size_t num_samples, 
                                   std::vector<float>& output) {
    output.resize(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        output[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
}

// Energy VAD Implementation (fallback)
EnergyVAD::EnergyVAD(const Config& config) 
    : config_(config), energy_threshold_(0.01f), history_size_(10) {
    energy_history_.reserve(history_size_);
}

bool EnergyVAD::initialize(const Config& config) {
    config_ = config;
    energy_threshold_ = 0.01f;
    energy_history_.clear();
    return true;
}

VADInterface::VADResult EnergyVAD::processChunk(const int16_t* samples, 
                                               size_t num_samples, 
                                               uint64_t timestamp_ms) {
    std::vector<float> audio_float;
    audio_float.resize(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        audio_float[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    return processChunk(audio_float, timestamp_ms);
}

VADInterface::VADResult EnergyVAD::processChunk(const std::vector<float>& audio, 
                                               uint64_t timestamp_ms) {
    float energy = calculateEnergy(audio);
    updateThreshold(energy);
    
    bool is_speech = energy > energy_threshold_;
    float confidence = std::min(1.0f, energy / (energy_threshold_ * 2.0f));
    
    return {is_speech, confidence, timestamp_ms};
}

void EnergyVAD::reset() {
    energy_history_.clear();
    energy_threshold_ = 0.01f;
}

float EnergyVAD::calculateEnergy(const std::vector<float>& audio) {
    if (audio.empty()) return 0.0f;
    
    float sum_squares = 0.0f;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    return sum_squares / audio.size();
}

void EnergyVAD::updateThreshold(float energy) {
    energy_history_.push_back(energy);
    if (energy_history_.size() > history_size_) {
        energy_history_.erase(energy_history_.begin());
    }
    
    if (energy_history_.size() >= 3) {
        // Adaptive threshold based on recent energy levels
        float mean_energy = std::accumulate(energy_history_.begin(), 
                                          energy_history_.end(), 0.0f) / energy_history_.size();
        energy_threshold_ = mean_energy * 0.5f; // 50% of average
        energy_threshold_ = std::max(energy_threshold_, 0.001f); // Minimum threshold
    }
}

// Factory functions
std::unique_ptr<VADInterface> createSileroVAD(const VADInterface::Config& config) {
    auto vad = std::make_unique<SileroVAD>(config);
    if (vad->initialize(config)) {
        return vad;
    }
    return nullptr;
}

std::unique_ptr<VADInterface> createEnergyVAD(const VADInterface::Config& config) {
    auto vad = std::make_unique<EnergyVAD>(config);
    if (vad->initialize(config)) {
        return vad;
    }
    return nullptr;
}

} // namespace onnx_stt