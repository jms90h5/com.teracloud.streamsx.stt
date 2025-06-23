/**
 * Simple test for NeMo FastConformer without external dependencies
 * Uses basic feature extraction for testing
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <onnxruntime_cxx_api.h>

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

// Load WAV file
std::vector<float> loadWavFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return {};
    }
    
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Read audio data
    std::vector<int16_t> samples(header.data_size / 2);
    file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
    
    // Convert to float
    std::vector<float> audio(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        audio[i] = samples[i] / 32768.0f;
    }
    
    std::cout << "Loaded audio: " << audio.size() << " samples, " 
              << header.sample_rate << " Hz" << std::endl;
    
    return audio;
}

// Add dither to audio
void addDither(std::vector<float>& audio, float dither = 1e-5f) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, dither);
    
    for (auto& sample : audio) {
        sample += distribution(generator);
    }
}

// Simple mel-like feature extraction (matching NeMo preprocessing)
std::vector<std::vector<float>> extractSimpleFeatures(const std::vector<float>& audio, 
                                                     int sample_rate = 16000,
                                                     int n_mels = 80,
                                                     float frame_shift_ms = 10.0f,
                                                     float frame_length_ms = 25.0f) {
    int frame_shift = sample_rate * frame_shift_ms / 1000;
    int frame_length = sample_rate * frame_length_ms / 1000;
    
    // Add dither to audio copy
    std::vector<float> dithered_audio = audio;
    addDither(dithered_audio);
    
    std::vector<std::vector<float>> features;
    
    // Hann window
    std::vector<float> window(frame_length);
    for (int i = 0; i < frame_length; i++) {
        window[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / (frame_length - 1));
    }
    
    for (size_t start = 0; start + frame_length <= dithered_audio.size(); start += frame_shift) {
        std::vector<float> frame_features(n_mels);
        
        // Apply window and compute energy
        float energy = 0.0f;
        for (int i = 0; i < frame_length; i++) {
            float windowed = dithered_audio[start + i] * window[i];
            energy += windowed * windowed;
        }
        
        // Create mel-like features (simplified) - NO NORMALIZATION
        float log_energy = std::log(energy + 1e-10f);
        
        // Simulate mel filterbank response (simplified)
        for (int mel = 0; mel < n_mels; mel++) {
            // Create variation across mel bins to simulate frequency response
            float mel_freq = (mel + 1.0f) / n_mels;
            float response = 1.0f - std::abs(mel_freq - 0.5f) * 2.0f;
            
            // Apply log scale (natural log, not log10)
            frame_features[mel] = log_energy - 2.0f + response * 2.0f;
        }
        
        features.push_back(frame_features);
    }
    
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

int main() {
    std::cout << "=== NeMo FastConformer Simple Test ===" << std::endl;
    
    try {
        // Configuration
        const std::string model_path = "models/fastconformer_nemo_export/ctc_model.onnx";
        const std::string vocab_path = "models/fastconformer_nemo_export/tokens.txt";
        const int n_mels = 80;
        const int blank_id = 1024;
        
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FastConformerTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        
        // Load model
        std::cout << "\nLoading model: " << model_path << std::endl;
        Ort::Session session(env, model_path.c_str(), session_options);
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto processed_signal_name = session.GetInputNameAllocated(0, allocator);
        auto processed_signal_length_name = session.GetInputNameAllocated(1, allocator);
        auto log_probs_name = session.GetOutputNameAllocated(0, allocator);
        auto encoded_lengths_name = session.GetOutputNameAllocated(1, allocator);
        
        std::cout << "Input names: " << processed_signal_name.get() 
                  << ", " << processed_signal_length_name.get() << std::endl;
        
        // Load vocabulary
        std::cout << "\nLoading vocabulary..." << std::endl;
        std::vector<std::string> vocab = loadVocabulary(vocab_path);
        std::cout << "Vocabulary size: " << vocab.size() << std::endl;
        
        // Load test audio
        std::string audio_file = "test_data/audio/librispeech-1995-1837-0001.wav";
        std::cout << "\nProcessing: " << audio_file << std::endl;
        std::cout << "Expected: 'it was the first great song'" << std::endl;
        
        std::vector<float> audio = loadWavFile(audio_file);
        if (audio.empty()) {
            std::cerr << "Failed to load audio" << std::endl;
            return 1;
        }
        
        // Extract features
        std::cout << "\nExtracting features..." << std::endl;
        auto features = extractSimpleFeatures(audio);
        std::cout << "Extracted " << features.size() << " frames" << std::endl;
        
        // Prepare input: transpose from [frames, mels] to [mels, frames]
        std::vector<float> input_data(n_mels * features.size());
        for (size_t i = 0; i < features.size(); i++) {
            for (int j = 0; j < n_mels; j++) {
                input_data[j * features.size() + i] = features[i][j];
            }
        }
        
        // Create input tensors
        std::vector<int64_t> signal_shape = {1, n_mels, static_cast<int64_t>(features.size())};
        std::vector<int64_t> length_shape = {1};
        std::vector<int64_t> length_data = {static_cast<int64_t>(features.size())};
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            signal_shape.data(), signal_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), length_data.size(),
            length_shape.data(), length_shape.size()));
        
        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        std::vector<const char*> input_names = {
            processed_signal_name.get(), 
            processed_signal_length_name.get()
        };
        std::vector<const char*> output_names = {
            log_probs_name.get(), 
            encoded_lengths_name.get()
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), input_tensors.data(), 2,
                                        output_names.data(), 2);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Inference time: " << duration << " ms" << std::endl;
        
        // Get outputs
        auto& log_probs_tensor = output_tensors[0];
        auto& lengths_tensor = output_tensors[1];
        
        const float* log_probs = log_probs_tensor.GetTensorData<float>();
        const int64_t* encoded_lengths = lengths_tensor.GetTensorData<int64_t>();
        int output_length = encoded_lengths[0];
        
        auto output_shape = log_probs_tensor.GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output shape: [" << output_shape[0] << ", " 
                  << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
        std::cout << "Encoded length: " << output_length << std::endl;
        
        // Decode
        std::string transcription = decodeCTC(log_probs, output_length, 
                                            output_shape[2], vocab, blank_id);
        
        std::cout << "\n=== Result ===" << std::endl;
        std::cout << "Transcription: '" << transcription << "'" << std::endl;
        
        // Show some token probabilities for debugging
        std::cout << "\nFirst few frame predictions:" << std::endl;
        for (int t = 0; t < std::min(5, output_length); t++) {
            int best = 0;
            float best_score = log_probs[t * output_shape[2]];
            for (int v = 1; v < output_shape[2]; v++) {
                if (log_probs[t * output_shape[2] + v] > best_score) {
                    best_score = log_probs[t * output_shape[2] + v];
                    best = v;
                }
            }
            std::cout << "  Frame " << t << ": token " << best 
                      << " (score: " << best_score << ")";
            if (best < static_cast<int>(vocab.size())) {
                std::cout << " = '" << vocab[best] << "'";
            } else if (best == blank_id) {
                std::cout << " = <blank>";
            }
            std::cout << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}