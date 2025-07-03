#ifndef NEMO_STT_ADAPTER_HPP
#define NEMO_STT_ADAPTER_HPP

#include "STTBackendAdapter.hpp"
#include <memory>
#include <vector>
#include <mutex>

// Forward declaration
class NeMoCTCImpl;

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

/**
 * NeMo-specific configuration
 */
struct NeMoConfig : public BackendConfig {
    std::string modelPath;
    std::string vocabPath;
    std::string cmvnFile = "none";
    int numThreads = 4;
    std::string provider = "CPU";  // or "CUDA"
    bool enableCache = true;
    int blankId = 1024;
    
    // Parse from BackendConfig
    void parseConfig(const BackendConfig& config);
};

/**
 * Adapter for NVIDIA NeMo STT models
 * 
 * This adapter wraps the existing NeMoCTCModel implementation
 * to provide the unified backend interface.
 */
class NeMoSTTAdapter : public STTBackendAdapter {
private:
    // NeMo CTC model implementation
    std::unique_ptr<NeMoCTCImpl> model_;
    
    // Configuration
    NeMoConfig config_;
    
    // Current transcription state
    struct TranscriptionState {
        std::string accumulatedText;
        uint64_t startTime;
        uint64_t currentTime;
        double totalConfidence;
        int numSegments;
        bool isActive;
        
        // Channel information
        int channelNumber;
        std::string channelRole;
        
        void reset() {
            accumulatedText.clear();
            startTime = 0;
            currentTime = 0;
            totalConfidence = 0.0;
            numSegments = 0;
            isActive = false;
            channelNumber = -1;
            channelRole.clear();
        }
    };
    
    TranscriptionState state_;
    mutable std::mutex stateMutex_;
    
    // Audio buffer for buffering if needed
    std::vector<float> audioBuffer_;
    
    // Helper methods
    TranscriptionResult createResult(const std::string& text, 
                                   double confidence,
                                   bool isFinal);
    
public:
    NeMoSTTAdapter();
    virtual ~NeMoSTTAdapter();
    
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
        return "nemo";
    }
    
    std::map<std::string, std::string> getStatus() const override;
    
    void setLogLevel(const std::string& level) override;
};

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com

#endif // NEMO_STT_ADAPTER_HPP