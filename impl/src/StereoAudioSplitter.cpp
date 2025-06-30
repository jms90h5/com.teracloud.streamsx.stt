#include "StereoAudioSplitter.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

// G.711 µ-law to PCM conversion
int16_t StereoAudioSplitter::ulawToPcm(uint8_t ulaw) {
    // µ-law decoding as per ITU-T G.711
    static const int16_t ulaw_table[256] = {
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
         -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
         -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
         -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
         -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
         -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
         -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
          -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
          -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
          -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
          -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
          -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
           -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
         32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
         23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
         15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
         11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
          7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
          5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
          3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
          2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
          1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
          1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
           876,    844,    812,    780,    748,    716,    684,    652,
           620,    588,    556,    524,    492,    460,    428,    396,
           372,    356,    340,    324,    308,    292,    276,    260,
           244,    228,    212,    196,    180,    164,    148,    132,
           120,    112,    104,     96,     88,     80,     72,     64,
            56,     48,     40,     32,     24,     16,      8,      0
    };
    
    return ulaw_table[ulaw];
}

// G.711 A-law to PCM conversion
int16_t StereoAudioSplitter::alawToPcm(uint8_t alaw) {
    // A-law decoding as per ITU-T G.711
    static const int16_t alaw_table[256] = {
         -5504,  -5248,  -6016,  -5760,  -4480,  -4224,  -4992,  -4736,
         -7552,  -7296,  -8064,  -7808,  -6528,  -6272,  -7040,  -6784,
         -2752,  -2624,  -3008,  -2880,  -2240,  -2112,  -2496,  -2368,
         -3776,  -3648,  -4032,  -3904,  -3264,  -3136,  -3520,  -3392,
        -22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
        -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
        -11008, -10496, -12032, -11520,  -8960,  -8448,  -9984,  -9472,
        -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
          -344,   -328,   -376,   -360,   -280,   -264,   -312,   -296,
          -472,   -456,   -504,   -488,   -408,   -392,   -440,   -424,
           -88,    -72,   -120,   -104,    -24,     -8,    -56,    -40,
          -216,   -200,   -248,   -232,   -152,   -136,   -184,   -168,
         -1376,  -1312,  -1504,  -1440,  -1120,  -1056,  -1248,  -1184,
         -1888,  -1824,  -2016,  -1952,  -1632,  -1568,  -1760,  -1696,
          -688,   -656,   -752,   -720,   -560,   -528,   -624,   -592,
          -944,   -912,  -1008,   -976,   -816,   -784,   -880,   -848,
          5504,   5248,   6016,   5760,   4480,   4224,   4992,   4736,
          7552,   7296,   8064,   7808,   6528,   6272,   7040,   6784,
          2752,   2624,   3008,   2880,   2240,   2112,   2496,   2368,
          3776,   3648,   4032,   3904,   3264,   3136,   3520,   3392,
         22016,  20992,  24064,  23040,  17920,  16896,  19968,  18944,
         30208,  29184,  32256,  31232,  26112,  25088,  28160,  27136,
         11008,  10496,  12032,  11520,   8960,   8448,   9984,   9472,
         15104,  14592,  16128,  15616,  13056,  12544,  14080,  13568,
           344,    328,    376,    360,    280,    264,    312,    296,
           472,    456,    504,    488,    408,    392,    440,    424,
            88,     72,    120,    104,     24,      8,     56,     40,
           216,    200,    248,    232,    152,    136,    184,    168,
          1376,   1312,   1504,   1440,   1120,   1056,   1248,   1184,
          1888,   1824,   2016,   1952,   1632,   1568,   1760,   1696,
           688,    656,    752,    720,    560,    528,    624,    592,
           944,    912,   1008,    976,    816,    784,    880,    848
    };
    
    return alaw_table[alaw];
}

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitInterleavedPCM16(
    const int16_t* interleavedData, 
    size_t numSamples,
    const SplitOptions& options) {
    
    if (numSamples % 2 != 0) {
        throw std::invalid_argument("Number of samples must be even for stereo data");
    }
    
    ChannelBuffers result;
    size_t numSamplesPerChannel = numSamples / 2;
    
    result.left.reserve(numSamplesPerChannel);
    result.right.reserve(numSamplesPerChannel);
    
    // Split interleaved samples: L R L R L R -> L L L, R R R
    for (size_t i = 0; i < numSamples; i += 2) {
        float leftSample = options.normalizeFloat ? 
            normalizeInt16(interleavedData[i]) : 
            static_cast<float>(interleavedData[i]);
            
        float rightSample = options.normalizeFloat ? 
            normalizeInt16(interleavedData[i + 1]) : 
            static_cast<float>(interleavedData[i + 1]);
        
        result.left.push_back(leftSample);
        result.right.push_back(rightSample);
    }
    
    // Apply resampling if needed (e.g., 8kHz to 16kHz for NeMo)
    if (options.targetSampleRate > 0 && 
        options.targetSampleRate != options.sourceSampleRate) {
        float factor = static_cast<float>(options.targetSampleRate) / 
                      static_cast<float>(options.sourceSampleRate);
        
        if (factor > 1.0f) {
            result.left = upsampleLinear(result.left, factor);
            result.right = upsampleLinear(result.right, factor);
        } else {
            // Downsampling not implemented yet
            throw std::runtime_error("Downsampling not yet implemented");
        }
    }
    
    return result;
}

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitInterleavedPCM8(
    const uint8_t* interleavedData,
    size_t numSamples,
    const SplitOptions& options) {
    
    if (numSamples % 2 != 0) {
        throw std::invalid_argument("Number of samples must be even for stereo data");
    }
    
    ChannelBuffers result;
    size_t numSamplesPerChannel = numSamples / 2;
    
    result.left.reserve(numSamplesPerChannel);
    result.right.reserve(numSamplesPerChannel);
    
    // Split interleaved 8-bit samples
    for (size_t i = 0; i < numSamples; i += 2) {
        float leftSample = options.normalizeFloat ? 
            normalizeUint8(interleavedData[i]) : 
            static_cast<float>(interleavedData[i] - 128);
            
        float rightSample = options.normalizeFloat ? 
            normalizeUint8(interleavedData[i + 1]) : 
            static_cast<float>(interleavedData[i + 1] - 128);
        
        result.left.push_back(leftSample);
        result.right.push_back(rightSample);
    }
    
    return result;
}

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitNonInterleaved(
    const int16_t* leftData,
    const int16_t* rightData,
    size_t numSamplesPerChannel,
    const SplitOptions& options) {
    
    ChannelBuffers result;
    result.left.reserve(numSamplesPerChannel);
    result.right.reserve(numSamplesPerChannel);
    
    // Process left channel
    for (size_t i = 0; i < numSamplesPerChannel; ++i) {
        float sample = options.normalizeFloat ? 
            normalizeInt16(leftData[i]) : 
            static_cast<float>(leftData[i]);
        result.left.push_back(sample);
    }
    
    // Process right channel
    for (size_t i = 0; i < numSamplesPerChannel; ++i) {
        float sample = options.normalizeFloat ? 
            normalizeInt16(rightData[i]) : 
            static_cast<float>(rightData[i]);
        result.right.push_back(sample);
    }
    
    return result;
}

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitG711uLaw(
    const uint8_t* g711Data,
    size_t numBytes,
    bool isInterleaved) {
    
    ChannelBuffers result;
    
    if (isInterleaved) {
        if (numBytes % 2 != 0) {
            throw std::invalid_argument("Number of bytes must be even for interleaved stereo");
        }
        
        size_t numSamplesPerChannel = numBytes / 2;
        result.left.reserve(numSamplesPerChannel);
        result.right.reserve(numSamplesPerChannel);
        
        // Decode interleaved µ-law: L R L R -> PCM L L L, R R R
        for (size_t i = 0; i < numBytes; i += 2) {
            int16_t leftPcm = ulawToPcm(g711Data[i]);
            int16_t rightPcm = ulawToPcm(g711Data[i + 1]);
            
            result.left.push_back(normalizeInt16(leftPcm));
            result.right.push_back(normalizeInt16(rightPcm));
        }
    } else {
        // Non-interleaved: first half is left, second half is right
        size_t numSamplesPerChannel = numBytes / 2;
        result.left.reserve(numSamplesPerChannel);
        result.right.reserve(numSamplesPerChannel);
        
        // Decode left channel
        for (size_t i = 0; i < numSamplesPerChannel; ++i) {
            int16_t pcm = ulawToPcm(g711Data[i]);
            result.left.push_back(normalizeInt16(pcm));
        }
        
        // Decode right channel
        for (size_t i = numSamplesPerChannel; i < numBytes; ++i) {
            int16_t pcm = ulawToPcm(g711Data[i]);
            result.right.push_back(normalizeInt16(pcm));
        }
    }
    
    return result;
}

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitG711aLaw(
    const uint8_t* g711Data,
    size_t numBytes,
    bool isInterleaved) {
    
    ChannelBuffers result;
    
    if (isInterleaved) {
        if (numBytes % 2 != 0) {
            throw std::invalid_argument("Number of bytes must be even for interleaved stereo");
        }
        
        size_t numSamplesPerChannel = numBytes / 2;
        result.left.reserve(numSamplesPerChannel);
        result.right.reserve(numSamplesPerChannel);
        
        // Decode interleaved A-law
        for (size_t i = 0; i < numBytes; i += 2) {
            int16_t leftPcm = alawToPcm(g711Data[i]);
            int16_t rightPcm = alawToPcm(g711Data[i + 1]);
            
            result.left.push_back(normalizeInt16(leftPcm));
            result.right.push_back(normalizeInt16(rightPcm));
        }
    } else {
        // Non-interleaved
        size_t numSamplesPerChannel = numBytes / 2;
        result.left.reserve(numSamplesPerChannel);
        result.right.reserve(numSamplesPerChannel);
        
        // Decode left channel
        for (size_t i = 0; i < numSamplesPerChannel; ++i) {
            int16_t pcm = alawToPcm(g711Data[i]);
            result.left.push_back(normalizeInt16(pcm));
        }
        
        // Decode right channel
        for (size_t i = numSamplesPerChannel; i < numBytes; ++i) {
            int16_t pcm = alawToPcm(g711Data[i]);
            result.right.push_back(normalizeInt16(pcm));
        }
    }
    
    return result;
}

std::vector<float> StereoAudioSplitter::upsampleLinear(
    const std::vector<float>& input,
    float factor) {
    
    if (factor < 1.0f) {
        throw std::invalid_argument("Upsampling factor must be >= 1.0");
    }
    
    if (factor == 1.0f) {
        return input;  // No resampling needed
    }
    
    size_t inputSize = input.size();
    size_t outputSize = static_cast<size_t>(inputSize * factor);
    std::vector<float> output;
    output.reserve(outputSize);
    
    // Simple linear interpolation
    for (size_t i = 0; i < outputSize; ++i) {
        float srcIndex = i / factor;
        size_t srcIndexInt = static_cast<size_t>(srcIndex);
        float fraction = srcIndex - srcIndexInt;
        
        if (srcIndexInt >= inputSize - 1) {
            // Last sample
            output.push_back(input.back());
        } else {
            // Linear interpolation between two samples
            float sample1 = input[srcIndexInt];
            float sample2 = input[srcIndexInt + 1];
            float interpolated = sample1 * (1.0f - fraction) + sample2 * fraction;
            output.push_back(interpolated);
        }
    }
    
    return output;
}

std::vector<float> StereoAudioSplitter::resample(
    const std::vector<float>& input,
    int inputRate,
    int outputRate) {
    
    if (inputRate == outputRate) {
        return input;  // No resampling needed
    }
    
    float factor = static_cast<float>(outputRate) / static_cast<float>(inputRate);
    
    if (factor > 1.0f) {
        // Upsampling
        return upsampleLinear(input, factor);
    } else {
        // Downsampling - not implemented yet
        throw std::runtime_error("Downsampling not yet implemented");
    }
}

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com