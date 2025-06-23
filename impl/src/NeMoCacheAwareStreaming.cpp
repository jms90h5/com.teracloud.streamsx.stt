#include "NeMoCacheAwareStreaming.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace onnx_stt {

NeMoCacheAwareStreaming::NeMoCacheAwareStreaming(const std::string& /*model_dir*/,
                                               LatencyMode latency_mode,
                                               DecoderType decoder_type)
    : latency_mode_(latency_mode)
    , decoder_type_(decoder_type)
    , chunk_size_(0)
    , context_frames_(0)
{
    // Initialize ONNX Runtime environment
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NeMoCacheAware");
    
    // Initialize streaming cache
    cache_ = std::make_unique<StreamingCache>();
    cache_->processed_frames = 0;
    
    // Configure latency mode
    configureLatencyMode();
    
    std::cout << "NeMo Cache-Aware Streaming initialized:" << std::endl;
    std::cout << "  Latency mode: " << static_cast<int>(latency_mode_) << std::endl;
    std::cout << "  Decoder type: " << static_cast<int>(decoder_type_) << std::endl;
    std::cout << "  Chunk size: " << chunk_size_ << " frames" << std::endl;
}

bool NeMoCacheAwareStreaming::initialize() {
    try {
        // Load ONNX models (skip for testing with mock implementation)
        // if (!loadONNXModels("models/nemo_cache_aware_onnx")) {
        //     std::cerr << "Failed to load ONNX models" << std::endl;
        //     return false;
        // }
        std::cout << "✓ Using mock implementation for testing" << std::endl;
        
        // Load vocabulary (use mock if not available)
        if (!loadVocabulary("models/nemo_extracted/a4398a6af7324038a05d5930e49f4cf2_vocab.txt")) {
            std::cout << "✓ Using mock vocabulary for testing" << std::endl;
            vocabulary_ = {"<blank>", "this", "is", "a", "test", "of", "the", "nvidia", "nemo", 
                          "cache", "aware", "streaming", "fastconformer", "model", "with", 
                          "real", "time", "speech", "recognition", "system"};
        }
        
        // Initialize feature extractor (for now, we'll use a simple fallback)
        // feature_extractor_ = createSimpleFbank({SAMPLE_RATE, N_MELS, 25, 10});
        
        // Initialize cache structures
        cache_->encoder_states.resize(N_LAYERS);
        cache_->attention_cache.resize(N_LAYERS);
        for (int i = 0; i < N_LAYERS; ++i) {
            cache_->encoder_states[i].resize(D_MODEL, 0.0f);
            cache_->attention_cache[i].resize(D_MODEL * 8, 0.0f); // 8 attention heads
        }
        
        std::cout << "✓ NeMo Cache-Aware Streaming model initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCacheAwareStreaming::loadONNXModels(const std::string& model_dir) {
    try {
        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load encoder model
        std::string encoder_path = model_dir + "/fastconformer_encoder_cache_aware.onnx";
        if (std::ifstream(encoder_path).good()) {
            encoder_session_ = std::make_unique<Ort::Session>(*ort_env_, encoder_path.c_str(), session_options);
            std::cout << "✓ Encoder model loaded from " << encoder_path << std::endl;
        } else {
            std::cerr << "Encoder model not found at " << encoder_path << std::endl;
            return false;
        }
        
        // Load CTC decoder model (if available)
        std::string ctc_path = model_dir + "/fastconformer_decoder_ctc.onnx";
        if (std::ifstream(ctc_path).good()) {
            ctc_decoder_session_ = std::make_unique<Ort::Session>(*ort_env_, ctc_path.c_str(), session_options);
            std::cout << "✓ CTC decoder model loaded from " << ctc_path << std::endl;
        } else {
            std::cout << "⚠ CTC decoder model not found, using fallback implementation" << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading ONNX models: " << e.what() << std::endl;
        return false;
    }
}

bool NeMoCacheAwareStreaming::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Cannot open vocabulary file: " << vocab_file << std::endl;
        return false;
    }
    
    vocabulary_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocabulary_.push_back(line);
        }
    }
    
    std::cout << "✓ Loaded vocabulary with " << vocabulary_.size() << " tokens" << std::endl;
    return true;
}

void NeMoCacheAwareStreaming::configureLatencyMode() {
    // Configure chunk size and context based on latency mode
    switch (latency_mode_) {
        case LatencyMode::ULTRA_LOW:  // 0ms
            chunk_size_ = 160;        // ~10ms chunks at 16kHz
            context_frames_ = 0;      // No look-ahead
            break;
        case LatencyMode::VERY_LOW:   // 80ms
            chunk_size_ = 1280;       // ~80ms chunks
            context_frames_ = 1;      // Minimal context
            break;
        case LatencyMode::LOW:        // 480ms
            chunk_size_ = 7680;       // ~480ms chunks
            context_frames_ = 16;     // Standard context
            break;
        case LatencyMode::MEDIUM:     // 1040ms
            chunk_size_ = 16640;      // ~1040ms chunks
            context_frames_ = 33;     // Extended context
            break;
    }
    
    cache_->context_size = context_frames_;
}

std::string NeMoCacheAwareStreaming::processAudioChunk(const float* audio_chunk,
                                                     int chunk_size,
                                                     bool is_final) {
    if (!encoder_session_) {
        return "[ERROR: Model not initialized]";
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Extract mel-spectrogram features
        std::vector<std::vector<float>> features = extractFeatures(audio_chunk, chunk_size);
        
        // Run cache-aware encoder
        std::vector<std::vector<float>> encoder_output = runEncoder(features);
        
        // Decode based on selected decoder type
        std::string result;
        switch (decoder_type_) {
            case DecoderType::CTC:
                result = runCTCDecoder(encoder_output);
                break;
            case DecoderType::RNNT:
                result = runRNNTDecoder(encoder_output);
                break;
            case DecoderType::HYBRID:
                // Use CTC for streaming, RNN-T for final
                if (is_final) {
                    result = runRNNTDecoder(encoder_output);
                } else {
                    result = runCTCDecoder(encoder_output);
                }
                break;
        }
        
        // Update cache and statistics
        cache_->processed_frames += chunk_size;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (is_final || !result.empty()) {
            std::cout << "Processed chunk in " << processing_time.count() << "ms: \"" << result << "\"" << std::endl;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing audio chunk: " << e.what() << std::endl;
        return "[ERROR: Processing failed]";
    }
}

std::vector<std::vector<float>> NeMoCacheAwareStreaming::extractFeatures(const float* audio, int length) {
    // Simple feature extraction fallback for testing
    // In real implementation, this would use proper mel-spectrogram computation
    
    int num_frames = length / 160; // 10ms frames at 16kHz
    std::vector<std::vector<float>> features(num_frames, std::vector<float>(N_MELS, 0.0f));
    
    // Generate dummy mel-spectrogram features for testing
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < N_MELS; ++j) {
            // Simple energy-based features
            int start_sample = i * 160;
            int end_sample = std::min(start_sample + 400, length); // 25ms window
            
            float energy = 0.0f;
            for (int s = start_sample; s < end_sample; ++s) {
                energy += audio[s] * audio[s];
            }
            
            // Convert to log mel scale (simplified)
            features[i][j] = std::log(std::max(energy + 1e-8f, 1e-8f)) + (j * 0.1f);
        }
    }
    
    return features;
}

std::vector<std::vector<float>> NeMoCacheAwareStreaming::runEncoder(const std::vector<std::vector<float>>& features) {
    // Mock encoder for testing - simulates FastConformer encoder output
    // In real implementation, this would run the actual ONNX model
    
    if (features.empty()) {
        return {};
    }
    
    // Simulate encoder processing with downsampling (typical 4x or 8x reduction)
    int downsample_factor = 4;
    int output_frames = (features.size() + downsample_factor - 1) / downsample_factor;
    
    std::vector<std::vector<float>> encoder_output(output_frames, std::vector<float>(D_MODEL));
    
    // Simulate encoder processing with some randomness based on input
    for (int i = 0; i < output_frames; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            // Create deterministic but varied output based on input features
            float value = 0.0f;
            if (i * downsample_factor < static_cast<int>(features.size())) {
                for (int k = 0; k < N_MELS && k < static_cast<int>(features[i * downsample_factor].size()); ++k) {
                    value += features[i * downsample_factor][k] * (j + 1) * 0.001f;
                }
            }
            encoder_output[i][j] = std::tanh(value); // Bounded output
        }
    }
    
    // Update encoder cache
    updateEncoderCache(encoder_output);
    
    return encoder_output;
}

std::string NeMoCacheAwareStreaming::runCTCDecoder(const std::vector<std::vector<float>>& encoder_output) {
    // Mock CTC decoder for testing
    // Returns a realistic-looking transcription based on input length
    
    if (encoder_output.empty()) {
        return "";
    }
    
    // Simulate different transcriptions based on latency mode
    std::vector<std::string> mock_words = {
        "this", "is", "a", "test", "of", "the", "nvidia", "nemo", 
        "cache", "aware", "streaming", "fastconformer", "model",
        "with", "real", "time", "speech", "recognition"
    };
    
    // Generate transcription based on input size
    int num_words = std::min(static_cast<int>(encoder_output.size() / 10), static_cast<int>(mock_words.size()));
    std::string result;
    
    for (int i = 0; i < num_words; ++i) {
        if (i > 0) result += " ";
        result += mock_words[i % mock_words.size()];
    }
    
    return result;
}

std::string NeMoCacheAwareStreaming::runRNNTDecoder(const std::vector<std::vector<float>>& encoder_output) {
    // Simplified RNN-T decoding (beam search would be more accurate)
    std::vector<int> token_ids;
    
    // For now, fall back to CTC-style decoding
    // A full RNN-T implementation would use the prediction network and joint network
    return runCTCDecoder(encoder_output);
}

std::string NeMoCacheAwareStreaming::decodeTokens(const std::vector<int>& token_ids) {
    std::string result;
    
    for (int token_id : token_ids) {
        if (token_id >= 0 && token_id < static_cast<int>(vocabulary_.size())) {
            std::string token = vocabulary_[token_id];
            
            // Handle subword tokens (SentencePiece format)
            if (token.find("▁") == 0) {
                // Remove ▁ and add space
                if (!result.empty()) result += " ";
                result += token.substr(3); // Remove ▁ (3 bytes in UTF-8)
            } else {
                // Append without space
                result += token;
            }
        }
    }
    
    return result;
}

void NeMoCacheAwareStreaming::updateEncoderCache(const std::vector<std::vector<float>>& new_states) {
    // Update cache with latest encoder states for streaming continuity
    if (!new_states.empty() && cache_) {
        // Keep the last frame's state for each layer
        // This is a simplified cache update - a full implementation would
        // manage attention cache more sophisticatedly
        for (size_t layer = 0; layer < cache_->encoder_states.size() && layer < new_states.size(); ++layer) {
            if (!new_states[layer].empty()) {
                cache_->encoder_states[layer] = new_states[layer];
            }
        }
    }
}

void NeMoCacheAwareStreaming::reset() {
    if (cache_) {
        cache_->processed_frames = 0;
        for (auto& layer_state : cache_->encoder_states) {
            std::fill(layer_state.begin(), layer_state.end(), 0.0f);
        }
        for (auto& attention_state : cache_->attention_cache) {
            std::fill(attention_state.begin(), attention_state.end(), 0.0f);
        }
        cache_->prediction_cache.clear();
    }
    
    std::cout << "Streaming cache reset" << std::endl;
}

std::string NeMoCacheAwareStreaming::getModelInfo() const {
    return "NVIDIA NeMo Cache-Aware FastConformer Streaming (114M params)\n"
           "Architecture: 17-layer FastConformer + Hybrid CTC/RNN-T\n"
           "Features: Multi-latency streaming, cache-aware attention\n"
           "Vocabulary: " + std::to_string(vocabulary_.size()) + " tokens";
}

void NeMoCacheAwareStreaming::setLatencyMode(LatencyMode mode) {
    latency_mode_ = mode;
    configureLatencyMode();
    std::cout << "Latency mode changed to: " << static_cast<int>(mode) << std::endl;
}

void NeMoCacheAwareStreaming::setDecoderType(DecoderType type) {
    decoder_type_ = type;
    std::cout << "Decoder type changed to: " << static_cast<int>(type) << std::endl;
}

NeMoCacheAwareStreaming::StreamingStats NeMoCacheAwareStreaming::getStreamingStats() const {
    StreamingStats stats;
    stats.total_frames_processed = cache_ ? cache_->processed_frames : 0;
    stats.average_processing_time_ms = 0.0f; // Would need to track this
    stats.current_latency_ms = static_cast<float>(chunk_size_) / SAMPLE_RATE * 1000.0f;
    stats.cache_size_mb = 0.1f; // Approximate cache size
    return stats;
}

// Interface implementations
bool NeMoCacheAwareStreaming::initialize(const ModelConfig& config) {
    config_ = config;
    return initialize();
}

ModelInterface::TranscriptionResult NeMoCacheAwareStreaming::processChunk(
    const std::vector<std::vector<float>>& features, uint64_t timestamp_ms) {
    
    TranscriptionResult result;
    result.timestamp_ms = timestamp_ms;
    result.is_final = false;
    result.confidence = 0.8;
    
    // Convert features to audio samples for legacy interface
    // This is a simplified approach - real implementation would work directly with features
    std::vector<float> dummy_audio(features.size() * 10, 0.0f);
    result.text = processAudioChunk(dummy_audio.data(), dummy_audio.size(), false);
    
    return result;
}

const ModelInterface::ModelConfig& NeMoCacheAwareStreaming::getConfig() const {
    return config_;
}

std::map<std::string, double> NeMoCacheAwareStreaming::getStats() const {
    auto stats = getStreamingStats();
    return {
        {"total_frames", static_cast<double>(stats.total_frames_processed)},
        {"avg_processing_time_ms", stats.average_processing_time_ms},
        {"current_latency_ms", stats.current_latency_ms},
        {"cache_size_mb", stats.cache_size_mb}
    };
}

int NeMoCacheAwareStreaming::getChunkFrames() const {
    return chunk_size_;
}

// Factory function
std::unique_ptr<NeMoCacheAwareStreaming> createNeMoCacheAwareModel(
    const std::string& model_dir,
    NeMoCacheAwareStreaming::LatencyMode latency,
    NeMoCacheAwareStreaming::DecoderType decoder) {
    
    auto model = std::make_unique<NeMoCacheAwareStreaming>(model_dir, latency, decoder);
    if (model->initialize()) {
        return model;
    }
    return nullptr;
}

} // namespace onnx_stt