#include "ZipformerRNNT.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <functional>
#include <thread>

namespace onnx_stt {

ZipformerRNNT::ZipformerRNNT(const Config& config)
    : config_(config)
    , env_(ORT_LOGGING_LEVEL_WARNING, "ZipformerRNNT")
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

ZipformerRNNT::~ZipformerRNNT() = default;

bool ZipformerRNNT::initialize() {
    try {
        // Configure session options
        session_options_.SetIntraOpNumThreads(config_.num_threads);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Load encoder
        std::cout << "Loading encoder from: " << config_.encoder_path << std::endl;
        encoder_ = std::make_unique<Ort::Session>(env_, config_.encoder_path.c_str(), session_options_);
        
        // Load decoder
        std::cout << "Loading decoder from: " << config_.decoder_path << std::endl;
        decoder_ = std::make_unique<Ort::Session>(env_, config_.decoder_path.c_str(), session_options_);
        
        // Load joiner
        std::cout << "Loading joiner from: " << config_.joiner_path << std::endl;
        joiner_ = std::make_unique<Ort::Session>(env_, config_.joiner_path.c_str(), session_options_);
        
        // Load tokens
        if (!loadTokens(config_.tokens_path)) {
            return false;
        }
        
        // Initialize cache state
        cache_state_.initialize(config_);
        
        // Initialize with empty hypothesis
        reset();
        
        std::cout << "ZipformerRNNT initialized successfully" << std::endl;
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

ZipformerRNNT::Result ZipformerRNNT::processChunk(const std::vector<float>& features) {
    Result result;
    
    try {
        // Run encoder with current chunk and caches
        auto encoder_out = runEncoder(features);
        
        // Perform beam search step
        beamSearchStep(encoder_out);
        
        // Get best hypothesis
        if (!hypotheses_.empty()) {
            const auto& best = hypotheses_[0];
            result.tokens = best.tokens;
            result.text = tokensToText(best.tokens);
            result.confidence = std::exp(best.score / best.tokens.size());
            result.is_final = false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing chunk: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<float> ZipformerRNNT::runEncoder(const std::vector<float>& features) {
    // Prepare inputs
    std::vector<Ort::Value> inputs;
    
    // Add audio features
    std::vector<int64_t> feature_shape = {1, config_.chunk_size, config_.feature_dim};
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(features.data()),
        features.size(),
        feature_shape.data(),
        feature_shape.size()
    ));
    
    // Add all cache tensors
    auto cache_tensors = cache_state_.toOnnxValues(memory_info_);
    for (auto& tensor : cache_tensors) {
        inputs.push_back(std::move(tensor));
    }
    
    // Input names (must match model exactly)
    std::vector<const char*> input_names = {
        "x",
        // Length caches
        "cached_len_0", "cached_len_1", "cached_len_2", "cached_len_3", "cached_len_4",
        // Average caches
        "cached_avg_0", "cached_avg_1", "cached_avg_2", "cached_avg_3", "cached_avg_4",
        // Key caches
        "cached_key_0", "cached_key_1", "cached_key_2", "cached_key_3", "cached_key_4",
        // Value caches
        "cached_val_0", "cached_val_1", "cached_val_2", "cached_val_3", "cached_val_4",
        // Value2 caches
        "cached_val2_0", "cached_val2_1", "cached_val2_2", "cached_val2_3", "cached_val2_4",
        // Conv1 caches
        "cached_conv1_0", "cached_conv1_1", "cached_conv1_2", "cached_conv1_3", "cached_conv1_4",
        // Conv2 caches
        "cached_conv2_0", "cached_conv2_1", "cached_conv2_2", "cached_conv2_3", "cached_conv2_4"
    };
    
    // Output names
    std::vector<const char*> output_names = {
        "encoder_out",
        // New cache values (same names with "new_" prefix)
        "new_cached_len_0", "new_cached_len_1", "new_cached_len_2", "new_cached_len_3", "new_cached_len_4",
        "new_cached_avg_0", "new_cached_avg_1", "new_cached_avg_2", "new_cached_avg_3", "new_cached_avg_4",
        "new_cached_key_0", "new_cached_key_1", "new_cached_key_2", "new_cached_key_3", "new_cached_key_4",
        "new_cached_val_0", "new_cached_val_1", "new_cached_val_2", "new_cached_val_3", "new_cached_val_4",
        "new_cached_val2_0", "new_cached_val2_1", "new_cached_val2_2", "new_cached_val2_3", "new_cached_val2_4",
        "new_cached_conv1_0", "new_cached_conv1_1", "new_cached_conv1_2", "new_cached_conv1_3", "new_cached_conv1_4",
        "new_cached_conv2_0", "new_cached_conv2_1", "new_cached_conv2_2", "new_cached_conv2_3", "new_cached_conv2_4"
    };
    
    // Run encoder
    auto outputs = encoder_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size()
    );
    
    // Extract encoder output
    float* encoder_data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t encoder_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    
    std::vector<float> encoder_out(encoder_data, encoder_data + encoder_size);
    
    // Update caches with new values
    cache_state_.updateFromOutputs(outputs);
    
    return encoder_out;
}

std::vector<float> ZipformerRNNT::runDecoder(const std::vector<int>& tokens,
                                            const std::vector<float>& /* state */) {
    // Decoder expects shape [batch, 2] where second dimension is:
    // [0] = current token
    // [1] = context (usually 0 for RNN-T)
    std::vector<int64_t> decoder_input(2);
    decoder_input[0] = tokens.empty() ? 0 : tokens.back();  // Current token
    decoder_input[1] = 0;  // Context
    
    std::vector<int64_t> token_shape = {1, 2};
    auto token_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        decoder_input.data(),
        decoder_input.size(),
        token_shape.data(),
        token_shape.size()
    );
    
    // Run decoder
    const char* input_names[] = {"y"};
    const char* output_names[] = {"decoder_out"};
    
    auto outputs = decoder_->Run(
        Ort::RunOptions{nullptr},
        input_names, &token_tensor, 1,
        output_names, 1
    );
    
    // Extract decoder output
    float* decoder_data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t decoder_size = shape[shape.size()-1];  // Last dimension is decoder output size
    
    return std::vector<float>(decoder_data, decoder_data + decoder_size);
}

std::vector<float> ZipformerRNNT::runJoiner(const std::vector<float>& encoder_out,
                                           const std::vector<float>& decoder_out) {
    // Joiner expects [batch, 512] for both inputs
    std::vector<int64_t> shape = {1, 512};
    
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(encoder_out.data()),
        encoder_out.size(),
        shape.data(),
        shape.size()
    ));
    
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(decoder_out.data()),
        decoder_out.size(),
        shape.data(),
        shape.size()
    ));
    
    // Run joiner
    const char* input_names[] = {"encoder_out", "decoder_out"};
    const char* output_names[] = {"logit"};
    
    auto outputs = joiner_->Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), inputs.size(),
        output_names, 1
    );
    
    // Extract logits
    float* logits = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t vocab_size = output_shape[output_shape.size()-1];
    
    return std::vector<float>(logits, logits + vocab_size);
}

void ZipformerRNNT::beamSearchStep(const std::vector<float>& encoder_out) {
    // Temporary storage for new hypotheses
    std::priority_queue<Hypothesis> candidates;
    
    // The encoder output is [batch, time, 512]
    // For joiner, we need to process each time step
    // For now, let's process the last frame (most recent)
    size_t encoder_dim = 512;
    size_t num_frames = encoder_out.size() / encoder_dim;
    
    // Extract last frame
    std::vector<float> last_encoder_frame(encoder_dim);
    if (num_frames > 0) {
        size_t offset = (num_frames - 1) * encoder_dim;
        std::copy(encoder_out.begin() + offset, 
                  encoder_out.begin() + offset + encoder_dim,
                  last_encoder_frame.begin());
    }
    
    // Process each current hypothesis
    for (const auto& hyp : hypotheses_) {
        // Run decoder with current hypothesis
        auto decoder_out = runDecoder(hyp.tokens, hyp.decoder_state);
        
        // Run joiner to get next token probabilities
        auto logits = runJoiner(last_encoder_frame, decoder_out);
        
        // Apply softmax to get probabilities
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        for (auto& logit : logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }
        for (auto& logit : logits) {
            logit = std::log(logit / sum_exp);  // Log probabilities
        }
        
        // Consider top-k tokens
        std::vector<std::pair<float, int>> token_scores;
        int vocab_size = std::min(static_cast<int>(tokens_.size()), static_cast<int>(logits.size()));
        for (int i = 0; i < vocab_size; ++i) {
            token_scores.push_back({logits[i], i});
        }
        
        std::sort(token_scores.begin(), token_scores.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Take only top-k tokens
        int k = std::min(10, static_cast<int>(token_scores.size()));
        token_scores.resize(k);
        
        // Create new hypotheses
        for (const auto& score_token : token_scores) {
            float score = score_token.first;
            int token_id = score_token.second;
            Hypothesis new_hyp = hyp;
            
            if (token_id != blank_id_) {
                // Non-blank token: add to sequence
                new_hyp.tokens.push_back(token_id);
            }
            
            new_hyp.score = hyp.score + score;
            new_hyp.decoder_state = decoder_out;  // Update decoder state
            
            candidates.push(new_hyp);
            if (candidates.size() > static_cast<size_t>(config_.beam_size * 2)) {
                candidates.pop();  // Keep only top candidates
            }
        }
    }
    
    // Select top beam_size hypotheses
    hypotheses_.clear();
    while (!candidates.empty() && hypotheses_.size() < static_cast<size_t>(config_.beam_size)) {
        hypotheses_.push_back(candidates.top());
        candidates.pop();
    }
    
    // Reverse to have best hypothesis first
    std::reverse(hypotheses_.begin(), hypotheses_.end());
}

std::string ZipformerRNNT::tokensToText(const std::vector<int>& tokens) {
    std::string text;
    
    for (int token_id : tokens) {
        if (token_id >= 0 && token_id < static_cast<int>(tokens_.size())) {
            const std::string& token = tokens_[token_id];
            
            // Handle BPE tokens (e.g., "▁" for word boundaries)
            if (token.find("▁") == 0) {
                if (!text.empty()) {
                    text += " ";
                }
                text += token.substr(3);  // Remove "▁"
            } else {
                text += token;
            }
        }
    }
    
    return text;
}

bool ZipformerRNNT::loadTokens(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open tokens file: " << path << std::endl;
        return false;
    }
    
    tokens_.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        // Format: "token id" or just "token"
        size_t space_pos = line.find_last_of(' ');
        if (space_pos != std::string::npos) {
            tokens_.push_back(line.substr(0, space_pos));
        } else {
            tokens_.push_back(line);
        }
    }
    
    // Find blank token (usually "<blk>" or similar)
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (tokens_[i] == "<blk>" || tokens_[i] == "<blank>") {
            blank_id_ = i;
            break;
        }
    }
    
    std::cout << "Loaded " << tokens_.size() << " tokens, blank_id=" << blank_id_ << std::endl;
    return true;
}

ZipformerRNNT::Result ZipformerRNNT::finalize() {
    Result result;
    
    if (!hypotheses_.empty()) {
        const auto& best = hypotheses_[0];
        result.tokens = best.tokens;
        result.text = tokensToText(best.tokens);
        result.confidence = std::exp(best.score / best.tokens.size());
        result.is_final = true;
    }
    
    return result;
}

void ZipformerRNNT::reset() {
    // Reset cache state
    cache_state_.initialize(config_);
    
    // Reset hypotheses with empty sequence
    hypotheses_.clear();
    Hypothesis empty_hyp;
    empty_hyp.tokens.clear();
    empty_hyp.score = 0.0f;
    empty_hyp.decoder_state.resize(config_.decoder_dim, 0.0f);
    hypotheses_.push_back(empty_hyp);
}

} // namespace onnx_stt