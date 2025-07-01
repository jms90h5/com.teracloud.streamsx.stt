#ifndef STT_BACKEND_ADAPTER_HPP
#define STT_BACKEND_ADAPTER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

/**
 * Audio chunk for processing
 */
struct AudioChunk {
    const void* data;           // Raw audio data
    size_t size;                // Size in bytes
    std::string encoding;       // Audio encoding (pcm16, ulaw, etc.)
    int sampleRate;            // Sample rate in Hz
    int channels;              // Number of channels
    int bitsPerSample;         // Bits per sample
    uint64_t timestamp;        // Timestamp in milliseconds
    
    // Channel information
    int channelNumber;         // 0-based channel index (-1 for mono/mixed)
    std::string channelRole;   // "caller", "agent", etc.
    
    // Optional metadata
    std::map<std::string, std::string> metadata;
};

/**
 * Word timing information
 */
struct WordTiming {
    std::string word;
    double startTime;
    double endTime;
    double confidence;
};

/**
 * Speaker information
 */
struct SpeakerInfo {
    int speakerId;
    std::string speakerLabel;
    double confidence;
};

/**
 * Transcription result
 */
struct TranscriptionResult {
    std::string text;
    double confidence;
    bool isFinal;
    std::vector<WordTiming> wordTimings;
    std::vector<SpeakerInfo> speakers;
    uint64_t startTime;
    uint64_t endTime;
    std::string detectedLanguage;
    std::map<std::string, std::string> metadata;
    std::vector<std::string> alternatives;
    
    // Error information if transcription failed
    bool hasError = false;
    std::string errorMessage;
    std::string errorCode;
};

/**
 * Transcription options
 */
struct TranscriptionOptions {
    std::string languageCode = "en-US";
    bool enableWordTimings = false;
    bool enablePunctuation = true;
    bool enableSpeakerLabels = false;
    bool enableProfanityFilter = false;
    int maxAlternatives = 1;
    std::map<std::string, std::string> customOptions;
};

/**
 * Backend capabilities
 */
struct BackendCapabilities {
    bool supportsStreaming = false;
    bool supportsWordTimings = false;
    bool supportsSpeakerLabels = false;
    bool supportsCustomModels = false;
    std::vector<std::string> supportedLanguages;
    std::vector<std::string> supportedEncodings;
    int minSampleRate = 0;     // 0 = flexible
    int maxSampleRate = 0;     // 0 = flexible
    int maxChannels = 1;
    std::map<std::string, std::string> features;
};

/**
 * Base configuration for all backends
 */
struct BackendConfig {
    std::string backendType;
    std::map<std::string, std::string> parameters;
    std::map<std::string, std::string> credentials;
    
    // Helper methods
    std::string getString(const std::string& key, 
                         const std::string& defaultValue = "") const {
        auto it = parameters.find(key);
        return (it != parameters.end()) ? it->second : defaultValue;
    }
    
    int getInt(const std::string& key, int defaultValue = 0) const {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {}
        }
        return defaultValue;
    }
    
    bool getBool(const std::string& key, bool defaultValue = false) const {
        auto it = parameters.find(key);
        if (it != parameters.end()) {
            return (it->second == "true" || it->second == "1");
        }
        return defaultValue;
    }
};

/**
 * Abstract base class for STT backend adapters
 */
class STTBackendAdapter {
public:
    virtual ~STTBackendAdapter() = default;
    
    /**
     * Initialize the backend with configuration
     * @param config Backend-specific configuration
     * @return true if initialization successful
     */
    virtual bool initialize(const BackendConfig& config) = 0;
    
    /**
     * Process an audio chunk
     * @param audio Audio data to process
     * @param options Transcription options
     * @return Transcription result (may be partial)
     */
    virtual TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) = 0;
    
    /**
     * Finalize processing and get final results
     * @return Final transcription result
     */
    virtual TranscriptionResult finalize() = 0;
    
    /**
     * Reset the backend state for new transcription
     */
    virtual void reset() = 0;
    
    /**
     * Get backend capabilities
     * @return Capabilities structure
     */
    virtual BackendCapabilities getCapabilities() const = 0;
    
    /**
     * Check if backend is healthy and ready
     * @return true if backend is operational
     */
    virtual bool isHealthy() const = 0;
    
    /**
     * Get backend type identifier
     * @return Backend type string (e.g., "nemo", "watson")
     */
    virtual std::string getBackendType() const = 0;
    
    /**
     * Get current backend status/statistics
     * @return Map of status key-value pairs
     */
    virtual std::map<std::string, std::string> getStatus() const {
        return {{"healthy", isHealthy() ? "true" : "false"}};
    }
    
    /**
     * Set log level for this backend
     * @param level Log level (DEBUG, INFO, WARN, ERROR)
     */
    virtual void setLogLevel(const std::string& level) {}
};

/**
 * Factory for creating backend adapters
 */
class STTBackendFactory {
public:
    /**
     * Create a backend adapter
     * @param backendType Type of backend ("nemo", "watson", etc.)
     * @param config Backend configuration
     * @return Unique pointer to backend adapter
     */
    static std::unique_ptr<STTBackendAdapter> createBackend(
        const std::string& backendType,
        const BackendConfig& config);
    
    /**
     * Get list of available backends
     * @return Vector of backend type identifiers
     */
    static std::vector<std::string> getAvailableBackends();
    
    /**
     * Check if a backend is available
     * @param backendType Backend type to check
     * @return true if backend is available
     */
    static bool isBackendAvailable(const std::string& backendType);
    
    /**
     * Register a custom backend factory function
     * @param backendType Backend type identifier
     * @param factoryFunc Factory function
     */
    using FactoryFunc = std::function<std::unique_ptr<STTBackendAdapter>(const BackendConfig&)>;
    static void registerBackend(const std::string& backendType, FactoryFunc factoryFunc);
};

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com

#endif // STT_BACKEND_ADAPTER_HPP