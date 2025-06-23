/**
 * Comprehensive test for NeMo FastConformer ONNX model
 * 
 * Features:
 * - Proper mel-spectrogram extraction using kaldi-native-fbank
 * - Dynamic input handling
 * - CTC decoding with correct vocabulary
 * - Performance benchmarking
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <onnxruntime_cxx_api.h>
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

// JSON parsing (simple implementation)
#include <sstream>
#include <map>

// WAV file structure
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

// Model configuration
struct ModelConfig {
    std::string model_path;
    std::string vocab_path;
    std::string tokenizer_path;
    int sample_rate = 16000;
    int n_mels = 80;
    int vocab_size = 1024;
    int blank_id = 1024;
    
    // Feature extraction parameters
    float frame_shift_ms = 10.0f;
    float frame_length_ms = 25.0f;
    int n_fft = 512;
    std::string window_type = "hann";
};

// Load WAV file
std::vector<float> loadWavFile(const std::string& filename, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open WAV file: " + filename);
    }
    
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Validate WAV format
    if (std::string(header.riff, 4) != "RIFF" || std::string(header.wave, 4) != "WAVE") {
        throw std::runtime_error("Invalid WAV file format");
    }
    
    sample_rate = header.sample_rate;
    
    // Read audio data
    std::vector<int16_t> samples(header.data_size / 2);
    file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
    
    // Convert to float and normalize
    std::vector<float> audio(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    return audio;
}

// Add dither to audio
void addDither(std::vector<float>& audio, float dither = 1e-5f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, dither);
    
    for (auto& sample : audio) {
        sample += dist(gen);
    }
}

// Extract mel-spectrogram features using kaldi-native-fbank
std::vector<std::vector<float>> extractMelFeatures(const std::vector<float>& audio, 
                                                  const ModelConfig& config) {
    // Add dither to audio copy
    std::vector<float> dithered_audio = audio;
    addDither(dithered_audio, 1e-5f);
    
    // Configure feature extraction
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = config.sample_rate;
    opts.frame_opts.frame_shift_ms = config.frame_shift_ms;
    opts.frame_opts.frame_length_ms = config.frame_length_ms;
    opts.frame_opts.window_type = "hanning";  // Use Hann window as NeMo does
    opts.frame_opts.dither = 0.0f;  // We already added dither
    opts.mel_opts.num_bins = config.n_mels;
    opts.mel_opts.low_freq = 0.0f;
    opts.mel_opts.high_freq = config.sample_rate / 2.0f;
    opts.use_log_fbank = true;  // Natural log
    opts.use_energy = false;
    
    knf::OnlineFbank fbank(opts);
    
    // Process audio
    fbank.AcceptWaveform(config.sample_rate, dithered_audio.data(), dithered_audio.size());
    fbank.InputFinished();
    
    // Extract features
    int num_frames = fbank.NumFramesReady();
    std::vector<std::vector<float>> features;
    features.reserve(num_frames);
    
    for (int i = 0; i < num_frames; i++) {
        const float* frame = fbank.GetFrame(i);
        std::vector<float> feature_frame(frame, frame + config.n_mels);
        features.push_back(feature_frame);
    }
    
    return features;
}

// Load vocabulary
std::vector<std::string> loadVocabulary(const std::string& vocab_path) {
    std::vector<std::string> vocab;
    std::ifstream file(vocab_path);
    if (!file) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    
    return vocab;
}

// Simple greedy CTC decoding
std::string greedyCTCDecode(const std::vector<std::vector<float>>& log_probs,
                           const std::vector<std::string>& vocab,
                           int blank_id) {
    std::vector<int> tokens;
    int prev_token = blank_id;
    
    for (const auto& frame : log_probs) {
        // Find argmax
        int best_token = 0;
        float best_score = frame[0];
        for (size_t i = 1; i < frame.size(); i++) {
            if (frame[i] > best_score) {
                best_score = frame[i];
                best_token = i;
            }
        }
        
        // CTC decoding rules
        if (best_token != blank_id && best_token != prev_token) {
            tokens.push_back(best_token);
        }
        prev_token = best_token;
    }
    
    // Convert tokens to text (handle BPE tokens)
    std::string text;
    for (int token : tokens) {
        if (token < static_cast<int>(vocab.size())) {
            std::string tok = vocab[token];
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

// Load model configuration
ModelConfig loadConfig(const std::string& config_path) {
    ModelConfig config;
    
    // For simplicity, hardcode the paths relative to the config
    std::string base_dir = "models/fastconformer_nemo_export/";
    config.model_path = base_dir + "ctc_model.onnx";
    config.vocab_path = base_dir + "tokens.txt";
    config.tokenizer_path = base_dir + "tokenizer/tokenizer.model";
    
    // Other parameters from the exported config
    config.sample_rate = 16000;
    config.n_mels = 80;
    config.vocab_size = 1024;
    config.blank_id = 1024;
    config.frame_shift_ms = 10.0f;
    config.frame_length_ms = 25.0f;
    config.n_fft = 512;
    
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "=== NeMo FastConformer ONNX Test ===" << std::endl;
    
    try {
        // Load configuration
        ModelConfig config = loadConfig("models/fastconformer_nemo_export/config.json");
        std::cout << "Model: " << config.model_path << std::endl;
        std::cout << "Vocabulary: " << config.vocab_size << " tokens + blank" << std::endl;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastConformerTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::cout << "\nLoading model..." << std::endl;
        Ort::Session session(env, config.model_path.c_str(), session_options);
        
        // Print model info
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        std::cout << "Model inputs: " << num_inputs << ", outputs: " << num_outputs << std::endl;
        
        // Get input names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_0 = session.GetInputNameAllocated(0, allocator);
        auto input_name_1 = session.GetInputNameAllocated(1, allocator);
        auto output_name_0 = session.GetOutputNameAllocated(0, allocator);
        auto output_name_1 = session.GetOutputNameAllocated(1, allocator);
        
        // Load vocabulary
        std::cout << "\nLoading vocabulary..." << std::endl;
        std::vector<std::string> vocab = loadVocabulary(config.vocab_path);
        std::cout << "Loaded " << vocab.size() << " tokens" << std::endl;
        
        // Load test audio
        std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        std::cout << "\nLoading audio: " << audio_file << std::endl;
        std::cout << "Expected transcription: 'it was the first great song'" << std::endl;
        
        int audio_sample_rate;
        std::vector<float> audio = loadWavFile(audio_file, audio_sample_rate);
        float duration = audio.size() / static_cast<float>(audio_sample_rate);
        std::cout << "Audio: " << audio.size() << " samples, " << duration << " seconds" << std::endl;
        
        // Extract features
        std::cout << "\nExtracting mel-spectrogram features..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<float>> features = extractMelFeatures(audio, config);
        
        auto feature_time = std::chrono::high_resolution_clock::now();
        auto feature_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            feature_time - start_time).count();
        
        std::cout << "Extracted " << features.size() << " frames in " << feature_duration << " ms" << std::endl;
        
        // Prepare input tensors
        // Transpose features from [frames, mels] to [mels, frames]
        std::vector<float> input_data(config.n_mels * features.size());
        for (size_t i = 0; i < features.size(); i++) {
            for (int j = 0; j < config.n_mels; j++) {
                input_data[j * features.size() + i] = features[i][j];
            }
        }
        
        // Create input tensors
        std::vector<int64_t> input_shape = {1, config.n_mels, static_cast<int64_t>(features.size())};
        std::vector<int64_t> length_shape = {1};
        std::vector<int64_t> length_data = {static_cast<int64_t>(features.size())};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()));
        
        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        std::vector<const char*> input_names = {input_name_0.get(), input_name_1.get()};
        std::vector<const char*> output_names = {output_name_0.get(), output_name_1.get()};
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), input_tensors.data(), input_tensors.size(),
                                        output_names.data(), output_names.size());
        
        auto inference_time = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            inference_time - inference_start).count();
        
        std::cout << "Inference completed in " << inference_duration << " ms" << std::endl;
        
        // Get output
        auto& log_probs_tensor = output_tensors[0];
        auto& lengths_tensor = output_tensors[1];
        
        auto log_probs_shape = log_probs_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto output_length = lengths_tensor.GetTensorData<int64_t>()[0];
        
        std::cout << "Output shape: [" << log_probs_shape[0] << ", " 
                  << log_probs_shape[1] << ", " << log_probs_shape[2] << "]" << std::endl;
        std::cout << "Encoded length: " << output_length << std::endl;
        
        // Convert output to vector for decoding
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
        std::cout << "\nDecoding..." << std::endl;
        std::string transcription = greedyCTCDecode(log_probs, vocab, config.blank_id);
        
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Transcription: '" << transcription << "'" << std::endl;
        
        // Calculate metrics
        float rtf = (feature_duration + inference_duration) / (duration * 1000.0f);
        std::cout << "\nPerformance:" << std::endl;
        std::cout << "  Feature extraction: " << feature_duration << " ms" << std::endl;
        std::cout << "  Inference: " << inference_duration << " ms" << std::endl;
        std::cout << "  Total: " << (feature_duration + inference_duration) << " ms" << std::endl;
        std::cout << "  Real-time factor: " << rtf << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}