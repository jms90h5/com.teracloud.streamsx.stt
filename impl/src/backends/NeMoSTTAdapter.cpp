#include "backends/NeMoSTTAdapter.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>

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
        // Parse configuration
        config_.parseConfig(config);
        
        // Validate required parameters
        if (config_.modelPath.empty()) {
            std::cerr << "NeMoSTTAdapter: modelPath is required" << std::endl;
            return false;
        }
        
        if (config_.vocabPath.empty()) {
            std::cerr << "NeMoSTTAdapter: vocabPath is required" << std::endl;
            return false;
        }
        
        // Create NeMo model instance
        model_ = std::make_unique<NeMoCTCModel>();
        
        // Initialize the model
        bool success = model_->initialize(
            config_.modelPath,
            config_.vocabPath,
            config_.cmvnFile,
            "NEMO_CTC",  // modelType
            config_.blankId,
            config_.numThreads,
            config_.provider
        );
        
        if (!success) {
            std::cerr << "NeMoSTTAdapter: Failed to initialize NeMo model" << std::endl;
            model_.reset();
            return false;
        }
        
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
        
        // Preprocess audio if needed
        std::vector<float> processedAudio = preprocessAudio(audio);
        
        if (processedAudio.empty()) {
            result.hasError = true;
            result.errorCode = "INVALID_AUDIO";
            result.errorMessage = "Failed to preprocess audio";
            return result;
        }
        
        // Process through NeMo model
        bool isFinal = false;
        auto transcription = model_->processAudioChunk(
            processedAudio.data(),
            processedAudio.size(),
            isFinal
        );
        
        // Update accumulated text if we got a result
        if (!transcription.text.empty()) {
            if (!state_.accumulatedText.empty()) {
                state_.accumulatedText += " ";
            }
            state_.accumulatedText += transcription.text;
            state_.totalConfidence += transcription.confidence;
            state_.numSegments++;
        }
        
        // Create result
        result = createResult(transcription.text, transcription.confidence, false);
        
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
    if (!model_) {
        TranscriptionResult result;
        result.hasError = true;
        result.errorCode = "NOT_INITIALIZED";
        result.errorMessage = "NeMo model not initialized";
        return result;
    }
    
    try {
        std::lock_guard<std::mutex> lock(stateMutex_);
        
        // Get final transcription from model
        auto finalTranscription = model_->finalize();
        
        // Add to accumulated text if needed
        if (!finalTranscription.text.empty()) {
            if (!state_.accumulatedText.empty()) {
                state_.accumulatedText += " ";
            }
            state_.accumulatedText += finalTranscription.text;
            state_.totalConfidence += finalTranscription.confidence;
            state_.numSegments++;
        }
        
        // Calculate average confidence
        double avgConfidence = state_.numSegments > 0 ? 
            state_.totalConfidence / state_.numSegments : 0.0;
        
        // Create final result with complete text
        auto result = createResult(state_.accumulatedText, avgConfidence, true);
        
        // Reset state for next transcription
        state_.reset();
        
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
    if (model_) {
        model_->reset();
    }
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
    return model_ && model_->isInitialized();
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
std::vector<float> NeMoSTTAdapter::preprocessAudio(const AudioChunk& audio) {
    std::vector<float> result;
    
    // Check encoding
    if (audio.encoding != "pcm16") {
        std::cerr << "NeMoSTTAdapter: Unsupported encoding: " << audio.encoding 
                  << " (only pcm16 supported)" << std::endl;
        return result;
    }
    
    // Check sample rate
    if (audio.sampleRate != 16000) {
        std::cerr << "NeMoSTTAdapter: Unsupported sample rate: " << audio.sampleRate 
                  << " (only 16000 Hz supported)" << std::endl;
        return result;
    }
    
    // Check channels
    if (audio.channels != 1) {
        std::cerr << "NeMoSTTAdapter: Only mono audio supported, got " 
                  << audio.channels << " channels" << std::endl;
        return result;
    }
    
    // Convert PCM16 to float
    const int16_t* pcmData = static_cast<const int16_t*>(audio.data);
    size_t numSamples = audio.size / sizeof(int16_t);
    
    result.reserve(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        result.push_back(pcmData[i] / 32768.0f);
    }
    
    return result;
}

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