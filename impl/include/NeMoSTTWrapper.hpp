#ifndef NEMO_STT_WRAPPER_HPP
#define NEMO_STT_WRAPPER_HPP

#include "NeMoSTTImpl.hpp"
#include <memory>
#include <deque>
#include <mutex>

namespace com::teracloud::streams::stt {

class NeMoSTTWrapper {
public:
    NeMoSTTWrapper();
    ~NeMoSTTWrapper();
    
    // Initialize with model path
    bool initialize(const std::string& modelPath);
    
    // Process audio chunk
    void processAudioChunk(const float* data, size_t samples, int sampleRate);
    
    // Get transcription if available
    bool getTranscription(std::string& transcription);
    
    // Reset the buffer
    void reset();
    
    // Set parameters
    void setChunkDurationMs(int ms) { chunkDurationMs_ = ms; }
    void setMinSpeechDurationMs(int ms) { minSpeechDurationMs_ = ms; }
    
private:
    std::unique_ptr<NeMoSTTImpl> impl_;
    std::deque<float> audioBuffer_;
    std::mutex bufferMutex_;
    
    int sampleRate_;
    int chunkDurationMs_;
    int minSpeechDurationMs_;
    size_t samplesPerChunk_;
    
    bool initialized_;
};

} // namespace com::teracloud::streams::stt

#endif // NEMO_STT_WRAPPER_HPP