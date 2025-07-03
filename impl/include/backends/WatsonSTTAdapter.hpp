#ifndef WATSON_STT_ADAPTER_HPP
#define WATSON_STT_ADAPTER_HPP

#include "STTBackendAdapter.hpp"
#include <memory>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

/**
 * Watson-specific configuration
 */
struct WatsonConfig : public BackendConfig {
    std::string apiKey;
    std::string serviceUrl = "wss://api.us-south.speech-to-text.watson.cloud.ibm.com";
    std::string model = "en-US_BroadbandModel";
    std::string acousticCustomizationId;
    std::string languageCustomizationId;
    bool smartFormatting = true;
    bool profanityFilter = false;
    bool enableSpeakerLabels = false;
    float speechDetectorSensitivity = 0.5f;
    float backgroundAudioSuppression = 0.5f;
    
    // Parse from BackendConfig
    void parseConfig(const BackendConfig& config);
};

/**
 * Adapter for IBM Watson Speech to Text service
 * 
 * This adapter provides WebSocket-based streaming transcription
 * using the Watson STT cloud service.
 * 
 * NOTE: This is a placeholder implementation. Full WebSocket
 * support will be added in Phase 3 of the migration.
 */
class WatsonSTTAdapter : public STTBackendAdapter {
private:
    // Configuration
    WatsonConfig config_;
    
    // Connection state
    std::atomic<bool> connected_;
    std::atomic<bool> listening_;
    
    // Transcription state
    struct TranscriptionState {
        std::string sessionId;
        std::string accumulatedText;
        uint64_t startTime;
        uint64_t currentTime;
        double totalConfidence;
        int numResults;
        
        // Channel information
        int channelNumber;
        std::string channelRole;
        
        void reset() {
            sessionId.clear();
            accumulatedText.clear();
            startTime = 0;
            currentTime = 0;
            totalConfidence = 0.0;
            numResults = 0;
            channelNumber = -1;
            channelRole.clear();
        }
    };
    
    TranscriptionState state_;
    mutable std::mutex stateMutex_;
    
    // Result queue for async processing
    std::queue<TranscriptionResult> resultQueue_;
    std::mutex queueMutex_;
    
    // Helper methods
    bool connect();
    void disconnect();
    bool authenticate();
    TranscriptionResult createResult(const std::string& text,
                                   double confidence,
                                   bool isFinal);
    
public:
    WatsonSTTAdapter();
    virtual ~WatsonSTTAdapter();
    
    // STTBackendAdapter interface implementation
    bool initialize(const BackendConfig& config) override;
    
    TranscriptionResult processAudio(
        const AudioChunk& audio,
        const TranscriptionOptions& options) override;
    
    TranscriptionResult finalize() override;
    
    void reset() override;
    
    BackendCapabilities getCapabilities() const override;
    
    bool isHealthy() const override;
    
    std::string getBackendType() const override {
        return "watson";
    }
    
    std::map<std::string, std::string> getStatus() const override;
    
    void setLogLevel(const std::string& level) override;
};

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com

#endif // WATSON_STT_ADAPTER_HPP