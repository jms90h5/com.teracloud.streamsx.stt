#include "OnnxSTTImpl.hpp"
#include "NeMoCTCModel.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace onnx_stt {

OnnxSTTImpl::OnnxSTTImpl(const Config& config)
    : config_(config) {
    last_process_time_ = std::chrono::steady_clock::now();
}

OnnxSTTImpl::~OnnxSTTImpl() = default;

bool OnnxSTTImpl::initialize() {
    try {
        if (config_.model_type == Config::NEMO_CTC) {
            // Initialize NeMo CTC model
            std::cout << "Loading NeMo CTC model from: " << config_.encoder_onnx_path << std::endl;
            
            NeMoCTCModel::Config ctc_config;
            ctc_config.model_path = config_.encoder_onnx_path;
            ctc_config.vocab_path = config_.vocab_path;
            ctc_config.sample_rate = config_.sample_rate;
            ctc_config.n_mels = config_.num_mel_bins;
            ctc_config.window_size_ms = config_.frame_length_ms;
            ctc_config.window_stride_ms = config_.frame_shift_ms;
            ctc_config.blank_id = config_.blank_id;
            ctc_config.num_threads = config_.num_threads;
            
            std::cout << "Creating NeMoCTCModel with path: " << ctc_config.model_path << std::endl;
            nemo_ctc_model_ = std::make_unique<NeMoCTCModel>(ctc_config);
            
            if (!nemo_ctc_model_->initialize()) {
                std::cerr << "Failed to initialize NeMo CTC model" << std::endl;
                return false;
            }
            
            std::cout << "OnnxSTTImpl initialized with NeMo CTC model" << std::endl;
            
        } else {
            // Initialize NeMo cache-aware streaming Conformer model
            std::cout << "Loading NeMo cache-aware streaming Conformer from: " << config_.encoder_onnx_path << std::endl;
            
            // Configure NeMo model
            NeMoCacheAwareConformer::NeMoConfig nemo_config;
            nemo_config.model_path = config_.encoder_onnx_path;
            nemo_config.num_threads = config_.num_threads;
            nemo_config.feature_dim = config_.num_mel_bins;
            nemo_config.chunk_frames = 500;  // 500 frames for FastConformer cache-aware streaming
            nemo_config.vocab_path = config_.vocab_path;
            
            // Create NeMo model instance
            nemo_cache_model_ = std::make_unique<NeMoCacheAwareConformer>(nemo_config);
            
            // Initialize the model with proper ModelConfig
            ModelInterface::ModelConfig model_config;
            model_config.encoder_path = config_.encoder_onnx_path;
            model_config.vocab_path = config_.vocab_path;
            model_config.sample_rate = config_.sample_rate;
            
            if (!nemo_cache_model_->initialize(model_config)) {
                std::cerr << "Failed to initialize NeMo cache-aware model" << std::endl;
                return false;
            }
            
            // Initialize real feature extraction using ImprovedFbank
            improved_fbank::FbankComputer::Options fbank_opts;
            fbank_opts.sample_rate = config_.sample_rate;
            fbank_opts.num_mel_bins = config_.num_mel_bins;
            fbank_opts.frame_length_ms = config_.frame_length_ms;
            fbank_opts.frame_shift_ms = config_.frame_shift_ms;
            fbank_opts.n_fft = 512;
            fbank_opts.apply_log = true;
            fbank_opts.dither = 1e-5f;
            fbank_opts.normalize_per_feature = false;  // FastConformer has normalize: NA
            
            fbank_computer_ = std::make_unique<improved_fbank::FbankComputer>(fbank_opts);
            
            std::cout << "OnnxSTTImpl initialized with NeMo cache-aware streaming Conformer" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

OnnxSTTImpl::TranscriptionResult OnnxSTTImpl::processAudioChunk(
    const int16_t* samples, 
    size_t num_samples, 
    uint64_t timestamp_ms) {
    
    auto start_time = std::chrono::steady_clock::now();
    TranscriptionResult result;
    result.timestamp_ms = timestamp_ms;
    result.is_final = false;
    result.confidence = 0.0;
    
    try {
        // Stage 1: Voice Activity Detection (optional - for now process all audio)
        
        // Stage 2: Convert int16 to float and buffer
        std::vector<float> float_samples(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            float_samples[i] = samples[i] / 32768.0f;
        }
        
        // Add to audio buffer
        audio_buffer_.insert(audio_buffer_.end(), 
                           float_samples.begin(), 
                           float_samples.end());
        
        // Update stats
        stats_.total_audio_ms += (num_samples * 1000) / config_.sample_rate;
        
        if (config_.model_type == Config::NEMO_CTC) {
            // For CTC model, process in larger chunks or complete audio
            const size_t min_samples = config_.sample_rate / 10;  // At least 100ms
            
            if (audio_buffer_.size() >= min_samples) {
                // Process available audio
                auto ctc_result = nemo_ctc_model_->processAudio(audio_buffer_);
                
                result.text = ctc_result.text;
                result.confidence = ctc_result.avg_confidence;
                result.is_final = true;  // CTC processes complete utterances
                
                // Clear buffer after processing
                audio_buffer_.clear();
            }
        } else {
            // Process if we have enough samples for a NeMo chunk (500 frames * 160 samples/frame)
            // FastConformer model needs 500 frames to produce exactly 125 frames after subsampling factor 4
            const size_t samples_per_chunk = 500 * (config_.sample_rate * config_.frame_shift_ms / 1000);
            
            while (audio_buffer_.size() >= samples_per_chunk) {
                // Extract chunk
                std::vector<float> chunk(audio_buffer_.begin(), 
                                       audio_buffer_.begin() + samples_per_chunk);
                audio_buffer_.erase(audio_buffer_.begin(), 
                                  audio_buffer_.begin() + samples_per_chunk);
                
                // Stage 3: Real feature extraction using ImprovedFbank
                auto features_2d = fbank_computer_->computeFeatures(chunk);
                
                std::cout << "Extracted " << features_2d.size() << " feature frames for " 
                          << chunk.size() << " audio samples" << std::endl;
                
                // Stage 4: Speech recognition using NeMo cache-aware model
                auto nemo_result = nemo_cache_model_->processChunk(features_2d, timestamp_ms);
                
                // Update result
                result.text = nemo_result.text;
                result.confidence = nemo_result.confidence;
                result.is_final = nemo_result.is_final;
            }
        }
        
        // Calculate latency
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        result.latency_ms = duration;
        
        // Update stats
        stats_.total_processing_ms += duration;
        if (stats_.total_audio_ms > 0) {
            stats_.real_time_factor = 
                static_cast<double>(stats_.total_processing_ms) / stats_.total_audio_ms;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing audio chunk: " << e.what() << std::endl;
    }
    
    return result;
}

void OnnxSTTImpl::reset() {
    if (nemo_cache_model_) {
        nemo_cache_model_->reset();
    }
    // NeMo CTC model doesn't need reset
    audio_buffer_.clear();
    feature_buffer_.clear();
    stats_ = Stats{};
}

std::vector<float> OnnxSTTImpl::extractFeatures(const std::vector<float>& /*audio*/) {
    // ERROR: This class should not handle feature extraction!
    // Feature extraction must be done by STTPipeline using ImprovedFbank or Kaldifeat
    std::cerr << "ERROR: OnnxSTTImpl::extractFeatures should not be called!" << std::endl;
    return std::vector<float>();  // Return empty vector
}

bool OnnxSTTImpl::setupNeMoModel() {
    // This is handled in initialize()
    return true;
}

} // namespace onnx_stt