#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include "StereoAudioSplitter.hpp"

// Simple WAV header structure
struct WavHeader {
    char riff[4];          // "RIFF"
    uint32_t fileSize;     // File size - 8
    char wave[4];          // "WAVE"
    char fmt[4];           // "fmt "
    uint32_t fmtSize;      // Format chunk size
    uint16_t format;       // Audio format (1 = PCM)
    uint16_t channels;     // Number of channels
    uint32_t sampleRate;   // Sample rate
    uint32_t byteRate;     // Byte rate
    uint16_t blockAlign;   // Block align
    uint16_t bitsPerSample;// Bits per sample
    char data[4];          // "data"
    uint32_t dataSize;     // Data chunk size
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <stereo_wav_file>" << std::endl;
        return 1;
    }
    
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return 1;
    }
    
    // Read WAV header
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Verify it's a WAV file
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "Not a valid WAV file" << std::endl;
        return 1;
    }
    
    std::cout << "WAV File Info:" << std::endl;
    std::cout << "  Channels: " << header.channels << std::endl;
    std::cout << "  Sample Rate: " << header.sampleRate << " Hz" << std::endl;
    std::cout << "  Bits/Sample: " << header.bitsPerSample << std::endl;
    std::cout << "  Data Size: " << header.dataSize << " bytes" << std::endl;
    
    if (header.channels != 2) {
        std::cerr << "Error: File is not stereo (has " << header.channels << " channels)" << std::endl;
        return 1;
    }
    
    // Read audio data
    std::vector<uint8_t> audioData(header.dataSize);
    file.read(reinterpret_cast<char*>(audioData.data()), header.dataSize);
    file.close();
    
    // Process based on format
    com::teracloud::streamsx::stt::StereoAudioSplitter::ChannelBuffers channels;
    com::teracloud::streamsx::stt::StereoAudioSplitter::SplitOptions options;
    options.sourceSampleRate = header.sampleRate;
    options.targetSampleRate = 0;  // No resampling
    
    try {
        if (header.bitsPerSample == 16) {
            const int16_t* pcmData = reinterpret_cast<const int16_t*>(audioData.data());
            size_t numSamples = audioData.size() / sizeof(int16_t);
            channels = com::teracloud::streamsx::stt::StereoAudioSplitter::splitInterleavedPCM16(
                pcmData, numSamples, options);
        } else if (header.bitsPerSample == 8) {
            // Assume u-law for 8-bit
            channels = com::teracloud::streamsx::stt::StereoAudioSplitter::splitG711uLaw(
                audioData.data(), audioData.size(), true);
        } else {
            std::cerr << "Unsupported bits per sample: " << header.bitsPerSample << std::endl;
            return 1;
        }
        
        std::cout << "\nChannel Separation Results:" << std::endl;
        std::cout << "  Left channel samples: " << channels.left.size() << std::endl;
        std::cout << "  Right channel samples: " << channels.right.size() << std::endl;
        
        // Calculate RMS for each channel
        float leftRMS = 0.0f, rightRMS = 0.0f;
        for (float sample : channels.left) {
            leftRMS += sample * sample;
        }
        for (float sample : channels.right) {
            rightRMS += sample * sample;
        }
        leftRMS = std::sqrt(leftRMS / channels.left.size());
        rightRMS = std::sqrt(rightRMS / channels.right.size());
        
        std::cout << "  Left channel RMS: " << leftRMS << std::endl;
        std::cout << "  Right channel RMS: " << rightRMS << std::endl;
        
        // Show first few samples
        std::cout << "\nFirst 10 samples:" << std::endl;
        for (size_t i = 0; i < 10 && i < channels.left.size(); ++i) {
            std::cout << "  [" << i << "] L: " << channels.left[i] 
                      << ", R: " << channels.right[i] << std::endl;
        }
        
        std::cout << "\nStereo audio successfully split!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing audio: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}