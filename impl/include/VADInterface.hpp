#ifndef VAD_INTERFACE_HPP
#define VAD_INTERFACE_HPP

#include <vector>
#include <memory>
#include <cstdint>

namespace onnx_stt {

/**
 * Voice Activity Detection interface
 * Determines whether audio segments contain speech or silence/noise
 */
class VADInterface {
public:
    struct Config {
        int sample_rate = 16000;
        int window_size_ms = 64;       // Window size in milliseconds
        int frame_shift_ms = 16;       // Frame shift in milliseconds
        float speech_threshold = 0.5;  // Threshold for speech detection
    };
    
    struct VADResult {
        bool is_speech;
        float confidence;               // Confidence score (0.0 to 1.0)
        uint64_t timestamp_ms;         // Timestamp of the audio segment
    };
    
    virtual ~VADInterface() = default;
    
    // Initialize VAD with configuration
    virtual bool initialize(const Config& config) = 0;
    
    // Process audio chunk and return VAD result
    virtual VADResult processChunk(const int16_t* samples, 
                                  size_t num_samples, 
                                  uint64_t timestamp_ms) = 0;
    
    // Process audio chunk (float format)
    virtual VADResult processChunk(const std::vector<float>& audio, 
                                  uint64_t timestamp_ms) = 0;
    
    // Reset internal state
    virtual void reset() = 0;
    
    // Get configuration
    virtual const Config& getConfig() const = 0;
};

/**
 * Factory function to create VAD instances
 */
std::unique_ptr<VADInterface> createSileroVAD(const VADInterface::Config& config);
std::unique_ptr<VADInterface> createEnergyVAD(const VADInterface::Config& config);

} // namespace onnx_stt

#endif // VAD_INTERFACE_HPP