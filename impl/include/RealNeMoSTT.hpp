#ifndef REAL_NEMO_STT_HPP
#define REAL_NEMO_STT_HPP

#include <string>
#include <vector>

/**
 * Real NeMo Speech-to-Text implementation using wav2vec2 ONNX model
 */
class RealNeMoSTT {
public:
    /**
     * Initialize the speech recognition engine
     * @param modelPath Path to the ONNX model file
     * @param vocabPath Path to the vocabulary file
     */
    RealNeMoSTT(const std::string& modelPath, const std::string& vocabPath);
    
    /**
     * Destructor
     */
    ~RealNeMoSTT();
    
    /**
     * Process audio data and return transcript
     * @param audioData Raw 16-bit PCM audio data
     * @param sampleCount Number of samples in the audio data
     * @return Transcribed text
     */
    std::string processAudio(const int16_t* audioData, size_t sampleCount);
    
    /**
     * Check if the model is ready for use
     * @return true if initialized successfully
     */
    bool isReady() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // REAL_NEMO_STT_HPP