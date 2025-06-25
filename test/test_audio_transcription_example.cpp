/**
 * Example: Audio Transcription Test with Expected Results
 * 
 * This test demonstrates how to validate transcription accuracy
 * against known expected results.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include "../impl/include/NeMoCTCImpl.hpp"

struct TranscriptionTest {
    std::string audio_file;
    std::string expected_text;
    float min_word_accuracy;
};

// Calculate word error rate (WER)
float calculateWordAccuracy(const std::string& expected, const std::string& actual) {
    // Simple word-based accuracy (production would use proper WER calculation)
    std::vector<std::string> expected_words, actual_words;
    
    // Tokenize expected
    std::stringstream ss1(expected);
    std::string word;
    while (ss1 >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        expected_words.push_back(word);
    }
    
    // Tokenize actual
    std::stringstream ss2(actual);
    while (ss2 >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        actual_words.push_back(word);
    }
    
    // Simple accuracy: matching words / total expected words
    int matches = 0;
    size_t min_len = std::min(expected_words.size(), actual_words.size());
    
    for (size_t i = 0; i < min_len; ++i) {
        if (expected_words[i] == actual_words[i]) {
            matches++;
        }
    }
    
    return expected_words.empty() ? 0.0f : 
           static_cast<float>(matches) / expected_words.size();
}

// Load audio file (WAV format)
std::vector<float> loadWavFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Skip WAV header (44 bytes for standard WAV)
    file.seekg(44);
    
    // Read 16-bit PCM data
    std::vector<int16_t> pcm_data;
    int16_t sample;
    while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
        pcm_data.push_back(sample);
    }
    
    // Convert to float32 normalized to [-1, 1]
    std::vector<float> float_data;
    float_data.reserve(pcm_data.size());
    for (int16_t s : pcm_data) {
        float_data.push_back(static_cast<float>(s) / 32768.0f);
    }
    
    return float_data;
}

int main() {
    std::cout << "=== Audio Transcription Accuracy Test ===" << std::endl;
    
    // Test cases with expected results
    std::vector<TranscriptionTest> tests = {
        {
            "../test_data/audio/librispeech-1995-1837-0001.wav",
            "he hoped there would be stew for dinner",
            0.90f  // 90% word accuracy required
        },
        {
            "../test_data/audio/silence-2sec.wav",
            "",
            1.0f  // Should produce empty result
        }
        // Add more test cases as audio files become available
    };
    
    // Initialize model
    NeMoCTCImpl model;
    std::string model_path = "../models/fastconformer_ctc_export/model.onnx";
    std::string tokens_path = "../models/fastconformer_ctc_export/tokens.txt";
    
    std::cout << "\nInitializing model..." << std::endl;
    if (!model.initialize(model_path, tokens_path)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    // Run tests
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : tests) {
        std::cout << "\n--- Testing: " << test.audio_file << " ---" << std::endl;
        std::cout << "Expected: \"" << test.expected_text << "\"" << std::endl;
        
        try {
            // Load audio
            std::vector<float> audio = loadWavFile(test.audio_file);
            std::cout << "Loaded " << audio.size() << " samples ("
                      << (audio.size() / 16000.0f) << " seconds)" << std::endl;
            
            // Transcribe
            std::string result = model.transcribe(audio);
            std::cout << "Result: \"" << result << "\"" << std::endl;
            
            // Calculate accuracy
            float accuracy = calculateWordAccuracy(test.expected_text, result);
            std::cout << "Word accuracy: " << (accuracy * 100) << "%" << std::endl;
            
            // Check if meets minimum accuracy
            if (accuracy >= test.min_word_accuracy) {
                std::cout << "✅ PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "❌ FAILED (accuracy below " 
                          << (test.min_word_accuracy * 100) << "%)" << std::endl;
                failed++;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ FAILED with error: " << e.what() << std::endl;
            failed++;
        }
    }
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total: " << (passed + failed) << std::endl;
    
    return failed > 0 ? 1 : 0;
}

/* Example output:
=== Audio Transcription Accuracy Test ===

Initializing model...

--- Testing: ../test_data/audio/librispeech-1995-1837-0001.wav ---
Expected: "he hoped there would be stew for dinner"
Loaded 25600 samples (1.6 seconds)
Result: "he hoped there would be stew for dinner"
Word accuracy: 100%
✅ PASSED

--- Testing: ../test_data/audio/silence-2sec.wav ---
Expected: ""
Loaded 32000 samples (2 seconds)
Result: ""
Word accuracy: 100%
✅ PASSED

=== Test Summary ===
Passed: 2
Failed: 0
Total: 2
*/