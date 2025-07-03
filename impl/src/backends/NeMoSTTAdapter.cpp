#include "backends/NeMoSTTAdapter.hpp"
#include "NeMoCTCImpl.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <mutex>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

// NeMoConfig implementation
void NeMoConfig::parseConfig(const BackendConfig& config) {
    backendType = config.backendType;
    parameters = config.parameters;
    credentials = config.credentials;
    
    // Parse NeMo-specific parameters
    modelPath = getString("modelPath");
    vocabPath = getString("vocabPath");
    cmvnFile = getString("cmvnFile", "none");
    numThreads = getInt("numThreads", 4);
    provider = getString("provider", "CPU");
    enableCache = getBool("enableCache", true);
    blankId = getInt("blankId", 1024);
}

// NeMoSTTAdapter implementation
NeMoSTTAdapter::NeMoSTTAdapter() 
    : model_(nullptr) {
    state_.reset();
}

NeMoSTTAdapter::~NeMoSTTAdapter() {
    // Model cleanup handled by unique_ptr
}

bool NeMoSTTAdapter::initialize(const BackendConfig& config) {
    try {
        std::cerr << "NeMoSTTAdapter::initialize() called" << std::endl;
        
        // Parse configuration
        config_.parseConfig(config);
        
        std::cerr << "NeMoSTTAdapter: modelPath = " << config_.modelPath << std::endl;
        std::cerr << "NeMoSTTAdapter: vocabPath = " << config_.vocabPath << std::endl;
        
        // Validate required parameters
        if (config_.modelPath.empty()) {
            std::cerr << "NeMoSTTAdapter: modelPath is required" << std::endl;
            return false;
        }
        
        if (config_.vocabPath.empty()) {
            std::cerr << "NeMoSTTAdapter: vocabPath is required" << std::endl;
            return false;
        }
        
        // Create NeMoCTCImpl instance
        std::cerr << "NeMoSTTAdapter: Creating NeMoCTC instance..." << std::endl;
        model_ = std::make_unique<NeMoCTCImpl>();
        
        if (!model_) {
            std::cerr << "NeMoSTTAdapter: Failed to create NeMoCTCImpl instance" << std::endl;
            return false;
        }
        
        std::cerr << "NeMoSTTAdapter: NeMoCTC instance created, initializing..." << std::endl;
        
        // Initialize the model with model path and vocab path
        bool success = model_->initialize(config_.modelPath, config_.vocabPath);
        
        if (!success) {
            std::cerr << "NeMoSTTAdapter: Failed to initialize NeMo model" << std::endl;
            model_.reset();
            return false;
        }
        
        std::cerr << "NeMoSTTAdapter: Model initialized successfully" << std::endl;
        
        std::cout << "NeMoSTTAdapter: Successfully initialized with model: " 
                  << config_.modelPath << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "NeMoSTTAdapter: Initialization error: " << e.what() << std::endl;
        model_.reset();
        return false;
    }
}

TranscriptionResult NeMoSTTAdapter::processAudio(
    const AudioChunk& audio,
    const TranscriptionOptions& options) {
    
    TranscriptionResult result;
    
    if (!model_) {
        result.hasError = true;
        result.errorCode = "NOT_INITIALIZED";
        result.errorMessage = "NeMo model not initialized";
        return result;
    }
    
    try {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Update state with channel information
        if (!state_.isActive) {
            state_.isActive = true;
            state_.startTime = audio.timestamp;
            state_.channelNumber = audio.channelNumber;
            state_.channelRole = audio.channelRole;
        }
        state_.currentTime = audio.timestamp;
        
        // Validate audio format
        if (audio.encoding != "pcm16") {
            result.hasError = true;
            result.errorCode = "INVALID_ENCODING";
            result.errorMessage = "Only pcm16 encoding is supported";
            return result;
        }
        
        if (audio.sampleRate != 16000) {
            result.hasError = true;
            result.errorCode = "INVALID_SAMPLE_RATE";
            result.errorMessage = "Only 16000 Hz sample rate is supported";
            return result;
        }
        
        if (audio.channels != 1) {
            result.hasError = true;
            result.errorCode = "INVALID_CHANNELS";
            result.errorMessage = "Only mono audio is supported";
            return result;
        }
        
        // Convert audio to float format as expected by NeMoCTCImpl
        const int16_t* pcmData = static_cast<const int16_t*>(audio.data);
        size_t numSamples = audio.size / sizeof(int16_t);
        
        // Debug: Log audio chunk details
        std::cerr << "NeMoSTTAdapter::processAudio - numSamples=" << numSamples 
                  << ", timestamp=" << audio.timestamp << std::endl;
        
        // Convert int16 to float and buffer the audio
        for (size_t i = 0; i < numSamples; i++) {
            audioBuffer_.push_back(pcmData[i] / 32768.0f);
        }
        
        // For now, we'll transcribe on each chunk
        // In a real streaming implementation, we'd buffer and transcribe at intervals
        if (audioBuffer_.size() > 0) {
            std::string transcription = model_->transcribe(audioBuffer_);
            
            // Debug: Log transcription result
            std::cerr << "NeMoSTTAdapter::processAudio - transcription='" << transcription << "'" << std::endl;
            
            // Only update if we got new text
            if (!transcription.empty() && transcription != state_.accumulatedText) {
                state_.accumulatedText = transcription;
                state_.totalConfidence = 0.95;  // NeMoCTCImpl doesn't provide confidence
                state_.numSegments++;
                std::cerr << "NeMoSTTAdapter::processAudio - accumulated text: '" 
                          << state_.accumulatedText << "'" << std::endl;
            }
        }
        
        // Create result from current accumulated text
        result = createResult(state_.accumulatedText, state_.totalConfidence, false);
        
        // NeMo doesn't support these features yet
        result.wordTimings.clear();
        result.speakers.clear();
        result.alternatives.clear();
        
    } catch (const std::exception& e) {
        result.hasError = true;
        result.errorCode = "PROCESSING_ERROR";
        result.errorMessage = std::string("Error processing audio: ") + e.what();
    }
    
    return result;
}

TranscriptionResult NeMoSTTAdapter::finalize() {
    std::cerr << "NeMoSTTAdapter::finalize() called" << std::endl;
    
    if (!model_) {
        TranscriptionResult result;
        result.hasError = true;
        result.errorCode = "NOT_INITIALIZED";
        result.errorMessage = "NeMo model not initialized";
        return result;
    }
    
    try {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Do a final transcription with all buffered audio
        if (audioBuffer_.size() > 0) {
            std::string finalTranscription = model_->transcribe(audioBuffer_);
            if (!finalTranscription.empty()) {
                state_.accumulatedText = finalTranscription;
                state_.totalConfidence = 0.95;
            }
        }
        
        std::cerr << "NeMoSTTAdapter::finalize() - accumulated text: '" 
                  << state_.accumulatedText << "', segments: " << state_.numSegments << std::endl;
        
        // Calculate average confidence
        double avgConfidence = state_.numSegments > 0 ? 
            state_.totalConfidence / state_.numSegments : 0.95;
        
        // Create final result with complete text
        auto result = createResult(state_.accumulatedText, avgConfidence, true);
        
        std::cerr << "NeMoSTTAdapter::finalize() - returning text: '" 
                  << result.text << "'" << std::endl;
        
        // Reset state for next transcription
        state_.reset();
        audioBuffer_.clear();
        
        return result;
        
    } catch (const std::exception& e) {
        TranscriptionResult result;
        result.hasError = true;
        result.errorCode = "FINALIZATION_ERROR";
        result.errorMessage = std::string("Error finalizing transcription: ") + e.what();
        return result;
    }
}

void NeMoSTTAdapter::reset() {
    std::lock_guard<std::mutex> lock(stateMutex_);
    state_.reset();
    // NeMoCTCImpl doesn't have a reset method - it's stateless
    audioBuffer_.clear();
}

BackendCapabilities NeMoSTTAdapter::getCapabilities() const {
    BackendCapabilities caps;
    
    caps.supportsStreaming = true;
    caps.supportsWordTimings = false;  // Not yet implemented
    caps.supportsSpeakerLabels = false;  // Not supported
    caps.supportsCustomModels = true;
    
    // NeMo currently supports English
    caps.supportedLanguages = {"en-US", "en-GB", "en-IN", "en-AU"};
    
    // Supported audio formats
    caps.supportedEncodings = {"pcm16"};
    
    // Sample rate requirements
    caps.minSampleRate = 16000;
    caps.maxSampleRate = 16000;  // NeMo models typically trained on 16kHz
    
    caps.maxChannels = 1;  // Mono audio only
    
    // Additional features
    caps.features["provider"] = config_.provider;
    caps.features["modelType"] = "NEMO_CTC";
    caps.features["requiresVocab"] = "true";
    
    return caps;
}

bool NeMoSTTAdapter::isHealthy() const {
    return model_ != nullptr;
}

std::map<std::string, std::string> NeMoSTTAdapter::getStatus() const {
    std::map<std::string, std::string> status;
    
    status["healthy"] = isHealthy() ? "true" : "false";
    status["backend"] = "nemo";
    status["model"] = config_.modelPath;
    status["provider"] = config_.provider;
    
    std::lock_guard<std::mutex> lock(stateMutex_);
    status["active"] = state_.isActive ? "true" : "false";
    
    if (state_.isActive) {
        status["segments"] = std::to_string(state_.numSegments);
        status["duration_ms"] = std::to_string(state_.currentTime - state_.startTime);
    }
    
    return status;
}

void NeMoSTTAdapter::setLogLevel(const std::string& level) {
    // Could implement log level control here
    std::cout << "NeMoSTTAdapter: Log level set to " << level << std::endl;
}

// Helper methods
TranscriptionResult NeMoSTTAdapter::createResult(
    const std::string& text,
    double confidence,
    bool isFinal) {
    
    TranscriptionResult result;
    
    result.text = text;
    result.confidence = confidence;
    result.isFinal = isFinal;
    result.startTime = state_.startTime;
    result.endTime = state_.currentTime;
    
    // Set channel information
    result.metadata["channelNumber"] = std::to_string(state_.channelNumber);
    result.metadata["channelRole"] = state_.channelRole;
    
    // Set backend information
    result.metadata["backend"] = "nemo";
    result.metadata["model"] = config_.modelPath;
    
    // Language is always English for NeMo
    result.detectedLanguage = "en-US";
    
    return result;
}

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com