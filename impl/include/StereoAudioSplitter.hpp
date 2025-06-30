#ifndef STEREO_AUDIO_SPLITTER_HPP
#define STEREO_AUDIO_SPLITTER_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

/**
 * Utility class for splitting stereo audio into separate channels.
 * Supports various stereo formats and telephony codecs.
 */
class StereoAudioSplitter {
public:
    /**
     * Container for separated audio channels
     */
    struct ChannelBuffers {
        std::vector<float> left;
        std::vector<float> right;
    };
    
    /**
     * Options for audio splitting and processing
     */
    struct SplitOptions {
        bool normalizeFloat;      // Convert to [-1.0, 1.0] range
        bool applyDithering;      // Apply dithering for bit depth conversion
        int targetSampleRate;     // Target sample rate (0 = no resampling)
        int sourceSampleRate;     // Source sample rate
        
        // Constructor with defaults
        SplitOptions() : 
            normalizeFloat(true),
            applyDithering(false),
            targetSampleRate(0),
            sourceSampleRate(8000) {}
    };
    
    /**
     * Split interleaved stereo PCM16 data into separate channels
     * @param interleavedData Pointer to interleaved audio data (L,R,L,R,...)
     * @param numSamples Total number of samples (must be even)
     * @param options Processing options
     * @return Separated channel buffers
     */
    static ChannelBuffers splitInterleavedPCM16(
        const int16_t* interleavedData, 
        size_t numSamples,
        const SplitOptions& options = SplitOptions());
    
    /**
     * Split interleaved stereo PCM8 data into separate channels
     * @param interleavedData Pointer to interleaved 8-bit audio data
     * @param numSamples Total number of samples (must be even)
     * @param options Processing options
     * @return Separated channel buffers
     */
    static ChannelBuffers splitInterleavedPCM8(
        const uint8_t* interleavedData,
        size_t numSamples,
        const SplitOptions& options = SplitOptions());
    
    /**
     * Split non-interleaved stereo data into separate channels
     * @param leftData Pointer to left channel data
     * @param rightData Pointer to right channel data
     * @param numSamplesPerChannel Number of samples per channel
     * @param options Processing options
     * @return Separated channel buffers
     */
    static ChannelBuffers splitNonInterleaved(
        const int16_t* leftData,
        const int16_t* rightData,
        size_t numSamplesPerChannel,
        const SplitOptions& options = SplitOptions());
        
    /**
     * Split G.711 µ-law stereo audio into PCM channels
     * @param g711Data Pointer to G.711 µ-law encoded data
     * @param numBytes Number of bytes in the data
     * @param isInterleaved True if data is interleaved stereo
     * @return Separated PCM channel buffers
     */
    static ChannelBuffers splitG711uLaw(
        const uint8_t* g711Data,
        size_t numBytes,
        bool isInterleaved = true);
    
    /**
     * Split G.711 A-law stereo audio into PCM channels
     * @param g711Data Pointer to G.711 A-law encoded data
     * @param numBytes Number of bytes in the data
     * @param isInterleaved True if data is interleaved stereo
     * @return Separated PCM channel buffers
     */
    static ChannelBuffers splitG711aLaw(
        const uint8_t* g711Data,
        size_t numBytes,
        bool isInterleaved = true);
        
    /**
     * Resample audio data to a different sample rate
     * @param input Input audio samples
     * @param inputRate Input sample rate in Hz
     * @param outputRate Output sample rate in Hz
     * @return Resampled audio data
     */
    static std::vector<float> resample(
        const std::vector<float>& input,
        int inputRate,
        int outputRate);
        
    /**
     * Simple linear interpolation resampling (for upsampling only)
     * @param input Input audio samples
     * @param factor Upsampling factor (must be >= 1.0)
     * @return Resampled audio data
     */
    static std::vector<float> upsampleLinear(
        const std::vector<float>& input,
        float factor);
        
private:
    // G.711 µ-law to PCM conversion
    static int16_t ulawToPcm(uint8_t ulaw);
    
    // G.711 A-law to PCM conversion
    static int16_t alawToPcm(uint8_t alaw);
    
    // Normalize int16 to float [-1.0, 1.0]
    static inline float normalizeInt16(int16_t sample) {
        return sample / 32768.0f;
    }
    
    // Normalize uint8 to float [-1.0, 1.0]
    static inline float normalizeUint8(uint8_t sample) {
        return (sample - 128) / 128.0f;
    }
};

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com

#endif // STEREO_AUDIO_SPLITTER_HPP