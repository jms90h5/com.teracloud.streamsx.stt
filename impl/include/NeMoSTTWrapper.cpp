#include "NeMoSTTWrapper.hpp"
#include <algorithm>
#include <iostream>

namespace com::teracloud::streams::stt {

NeMoSTTWrapper::NeMoSTTWrapper()
    : sampleRate_(16000),
      chunkDurationMs_(5000),  // 5 seconds default
      minSpeechDurationMs_(500), // 0.5 seconds minimum
      initialized_(false) {
    impl_ = std::make_unique<NeMoSTTImpl>();
}

NeMoSTTWrapper::~NeMoSTTWrapper() = default;

bool NeMoSTTWrapper::initialize(const std::string& modelPath) {
    if (!impl_->initialize(modelPath)) {
        std::cerr << "Failed to initialize NeMo model: " << impl_->getLastError() << std::endl;
        return false;
    }
    
    initialized_ = true;
    samplesPerChunk_ = (sampleRate_ * chunkDurationMs_) / 1000;
    return true;
}

void NeMoSTTWrapper::processAudioChunk(const float* data, size_t samples, int sampleRate) {
    if (!initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(bufferMutex_);
    
    // If sample rate differs, we'll need to resample
    // For now, we'll just store and let NeMo handle resampling
    sampleRate_ = sampleRate;
    samplesPerChunk_ = (sampleRate_ * chunkDurationMs_) / 1000;
    
    // Add samples to buffer
    audioBuffer_.insert(audioBuffer_.end(), data, data + samples);
}

bool NeMoSTTWrapper::getTranscription(std::string& transcription) {
    if (!initialized_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(bufferMutex_);
    
    // Check if we have enough samples
    size_t minSamples = (sampleRate_ * minSpeechDurationMs_) / 1000;
    if (audioBuffer_.size() < minSamples) {
        return false;
    }
    
    // Process up to chunk size
    size_t samplesToProcess = std::min(audioBuffer_.size(), samplesPerChunk_);
    
    // Copy samples to vector
    std::vector<float> audioData(audioBuffer_.begin(), 
                                  audioBuffer_.begin() + samplesToProcess);
    
    // Remove processed samples from buffer
    audioBuffer_.erase(audioBuffer_.begin(), 
                       audioBuffer_.begin() + samplesToProcess);
    
    // Transcribe
    transcription = impl_->transcribe(audioData, sampleRate_);
    
    // Check if it's an error
    if (transcription.find("Error:") == 0) {
        std::cerr << "Transcription error: " << transcription << std::endl;
        return false;
    }
    
    return !transcription.empty();
}

void NeMoSTTWrapper::reset() {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    audioBuffer_.clear();
}

} // namespace com::teracloud::streams::stt