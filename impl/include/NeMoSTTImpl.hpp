#ifndef NEMO_STT_IMPL_HPP
#define NEMO_STT_IMPL_HPP

#include <string>
#include <memory>
#include <vector>
#include <Python.h>

namespace com::teracloud::streams::stt {

class NeMoSTTImpl {
public:
    NeMoSTTImpl();
    ~NeMoSTTImpl();
    
    // Initialize the NeMo model
    bool initialize(const std::string& modelPath);
    
    // Process audio data and return transcription
    std::string transcribe(const std::vector<float>& audioData, int sampleRate);
    
    // Get the last error message
    std::string getLastError() const { return lastError_; }
    
private:
    PyObject* pModule_;
    PyObject* pModel_;
    PyObject* pTranscribeFunc_;
    std::string modelPath_;
    std::string lastError_;
    bool initialized_;
    
    // Python environment management
    static bool pythonInitialized_;
    static int instanceCount_;
    
    // Helper to set error message
    void setError(const std::string& error);
    
    // Helper to check and print Python errors
    bool checkPythonError();
};

} // namespace com::teracloud::streams::stt

#endif // NEMO_STT_IMPL_HPP