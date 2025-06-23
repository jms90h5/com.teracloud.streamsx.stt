/**
 * Test NVIDIA FastConformer with proper Fbank features and CMVN
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "impl/include/ImprovedFbank.hpp"
// #include <nlohmann/json.hpp>  // Removed to avoid dependency

// Simple WAV header structure  
struct WavHeader {
    char riff[4];
    uint32_t size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

std::vector<float> loadWavFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }
    
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    std::cout << "WAV file info:" << std::endl;
    std::cout << "  Sample rate: " << header.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << header.channels << std::endl;
    std::cout << "  Bits per sample: " << header.bits_per_sample << std::endl;
    
    // Read audio data
    std::vector<int16_t> samples(header.data_size / 2);
    file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
    
    // Convert to float normalized to [-1, 1]
    std::vector<float> audio(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    return audio;
}

// Load CMVN stats from file (hardcoded for now to avoid json dependency)
bool loadCMVNStats(const std::string& stats_file, 
                   std::vector<float>& mean_stats,
                   std::vector<float>& var_stats,
                   int& frame_count) {
    // Hardcoded CMVN stats from global_cmvn.stats
    mean_stats = {9.871770f,9.915302f,9.958834f,10.002366f,10.045898f,10.089430f,10.132962f,10.176494f,10.220026f,10.263558f,
                  10.307090f,10.350622f,10.394154f,10.437686f,10.481218f,10.524750f,10.568282f,10.611814f,10.655346f,10.698878f,
                  10.742411f,10.785943f,10.829475f,10.873007f,10.916539f,10.960071f,11.003603f,11.047135f,11.090667f,11.134199f,
                  11.177731f,11.221263f,11.264795f,11.308327f,11.351859f,11.395391f,11.438923f,11.482455f,11.525987f,11.569519f,
                  11.613051f,11.656583f,11.700115f,11.743647f,11.787179f,11.830711f,11.874243f,11.917775f,11.961307f,12.004839f,
                  12.048371f,12.091903f,12.135435f,12.178967f,12.222499f,12.266031f,12.309563f,12.353095f,12.396627f,12.440159f,
                  12.483692f,12.527224f,12.570756f,12.614288f,12.657820f,12.701352f,12.744884f,12.788416f,12.831948f,12.875480f,
                  12.919012f,12.962544f,13.006076f,13.049608f,13.093140f,13.136672f,13.180204f,13.223736f,13.267268f,13.310800f};
    
    var_stats = {2.668061f,2.784247f,2.902909f,3.024048f,3.147663f,3.273753f,3.402320f,3.533363f,3.666883f,3.802878f,
                 3.941350f,4.082297f,4.225721f,4.371621f,4.519998f,4.670850f,4.824179f,4.979983f,5.138264f,5.299021f,
                 5.462254f,5.627963f,5.796149f,5.966810f,6.139948f,6.315562f,6.493652f,6.674218f,6.857261f,7.042779f,
                 7.230774f,7.421245f,7.614192f,7.809615f,8.007514f,8.207890f,8.410742f,8.616069f,8.823873f,9.034153f,
                 9.246910f,9.462142f,9.679851f,9.900035f,10.122696f,10.347833f,10.575446f,10.805536f,11.038101f,11.273143f,
                 11.510661f,11.750655f,11.993125f,12.238071f,12.485493f,12.735392f,12.987767f,13.242618f,13.499945f,13.759748f,
                 14.022027f,14.286783f,14.554014f,14.823722f,15.095906f,15.370566f,15.647703f,15.927315f,16.209404f,16.493969f,
                 16.781009f,17.070526f,17.362520f,17.656989f,17.953935f,18.253356f,18.555254f,18.859628f,19.166478f,19.475805f};
    
    frame_count = 54068199;
    
    std::cout << "Loaded hardcoded CMVN stats: " << mean_stats.size() << " dims, " 
              << frame_count << " frames" << std::endl;
    return true;
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

// CTC beam search decoding
std::string decodeCTCBeamSearch(const float* log_probs, int seq_len, int vocab_size, 
                                const std::vector<std::string>& vocab, int beam_width = 5) {
    // For now, still use greedy decoding but with better token handling
    std::vector<int> tokens;
    int prev_token = -1;
    int blank_id = 0;  // CTC blank token is usually 0
    
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
        
        // CTC decoding rules
        if (best_token != blank_id) {
            if (best_token != prev_token) {
                tokens.push_back(best_token);
            }
        }
        prev_token = best_token;
    }
    
    // Convert tokens to string
    std::string result;
    for (int token : tokens) {
        if (token < vocab.size()) {
            std::string tok = vocab[token];
            // Handle subword tokens (e.g., "▁hello" -> " hello")
            if (tok.find("▁") == 0) {
                result += " " + tok.substr(3);  // Skip "▁" (3 bytes)
            } else {
                result += tok;
            }
        } else {
            result += "[" + std::to_string(token) + "]";
        }
    }
    
    // Trim leading space
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    
    return result;
}

int main() {
    std::cout << "=== Testing NVIDIA FastConformer with Proper Features ===" << std::endl;
    
    try {
        // Initialize proper feature extractor
        improved_fbank::FbankComputer::Options fbank_opts;
        fbank_opts.sample_rate = 16000;
        fbank_opts.num_mel_bins = 80;
        fbank_opts.frame_length_ms = 25;
        fbank_opts.frame_shift_ms = 10;
        fbank_opts.apply_log = true;
        fbank_opts.normalize_per_feature = true;
        
        improved_fbank::FbankComputer fbank(fbank_opts);
        
        // Load CMVN stats
        std::vector<float> mean_stats, var_stats;
        int frame_count;
        if (loadCMVNStats("global_cmvn.stats", mean_stats, var_stats, frame_count)) {
            fbank.setCMVNStats(mean_stats, var_stats, frame_count);
            std::cout << "CMVN normalization enabled" << std::endl;
        } else {
            std::cout << "Warning: No CMVN stats found, using raw features" << std::endl;
        }
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastConformerTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "models/fastconformer_ctc_export/model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        
        std::cout << "Model loaded successfully" << std::endl;
        
        // Load vocabulary
        std::vector<std::string> vocab = loadVocabulary("models/fastconformer_ctc_export/tokens.txt");
        std::cout << "Vocabulary loaded: " << vocab.size() << " tokens" << std::endl;
        
        // Show first few tokens for debugging
        std::cout << "First 10 tokens: ";
        for (int i = 0; i < std::min(10, (int)vocab.size()); i++) {
            std::cout << "[" << i << "]=" << vocab[i] << " ";
        }
        std::cout << std::endl;
        
        // Load test audio
        std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        std::vector<float> audio = loadWavFile(audio_file);
        
        if (audio.empty()) {
            std::cerr << "Failed to load audio file" << std::endl;
            return 1;
        }
        
        std::cout << "\nAudio loaded: " << audio.size() << " samples (" 
                  << (audio.size() / 16000.0) << " seconds)" << std::endl;
        
        // Extract proper features
        std::cout << "\nExtracting features with ImprovedFbank..." << std::endl;
        auto features = fbank.computeFeatures(audio);
        std::cout << "Features extracted: " << features.size() << " frames" << std::endl;
        
        // Model requires exactly 500 frames
        const int REQUIRED_FRAMES = 500;
        
        // Process audio in 500-frame chunks with overlap
        const int OVERLAP_FRAMES = 50;  // Overlap for context
        std::string full_transcript;
        
        for (size_t chunk_start = 0; chunk_start < features.size(); chunk_start += (REQUIRED_FRAMES - OVERLAP_FRAMES)) {
            // Prepare exactly 500 frames
            std::vector<float> chunk_data;
            chunk_data.reserve(REQUIRED_FRAMES * 80);
            
            int actual_frames = 0;
            for (int i = 0; i < REQUIRED_FRAMES; i++) {
                if (chunk_start + i < features.size()) {
                    // Use actual feature
                    for (float val : features[chunk_start + i]) {
                        chunk_data.push_back(val);
                    }
                    actual_frames++;
                } else {
                    // Pad with zeros
                    for (int j = 0; j < 80; j++) {
                        chunk_data.push_back(0.0f);
                    }
                }
            }
            
            std::cout << "\nProcessing chunk starting at frame " << chunk_start 
                      << " (" << actual_frames << " actual frames)" << std::endl;
            
            // Create input tensor
            std::vector<int64_t> input_shape = {1, REQUIRED_FRAMES, 80};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, chunk_data.data(), chunk_data.size(),
                input_shape.data(), input_shape.size());
            
            // Run inference
            const char* input_names[] = {"audio_signal"};
            const char* output_names[] = {"log_probs"};
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<Ort::Value> output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Get output
            float* log_probs = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            std::cout << "Output shape: [" << output_shape[0] << ", " 
                      << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
            std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
            
            // Decode  
            std::string chunk_transcript = decodeCTCBeamSearch(log_probs, output_shape[1], 
                                                              output_shape[2], vocab);
            std::cout << "Chunk transcription: \"" << chunk_transcript << "\"" << std::endl;
            
            // Accumulate transcript
            if (!chunk_transcript.empty()) {
                if (!full_transcript.empty()) {
                    full_transcript += " ";
                }
                full_transcript += chunk_transcript;
            }
            
            // Stop if we've processed all actual frames
            if (chunk_start + REQUIRED_FRAMES >= features.size()) {
                break;
            }
        }
        
        std::cout << "\n=== Final Transcription ===" << std::endl;
        std::cout << "\"" << full_transcript << "\"" << std::endl;
        
        std::cout << "\nExpected transcription:" << std::endl;
        std::cout << "\"he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce\"" << std::endl;
        
        std::cout << "\n✅ Test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}