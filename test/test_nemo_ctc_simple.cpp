#include <iostream>
#include "../impl/include/NeMoCTCImpl.hpp"

int main() {
    std::cout << "=== Simple NeMo CTC Test ===" << std::endl;
    
    NeMoCTCImpl nemo;
    
    std::string model_path = "../models/fastconformer_ctc_export/model.onnx";
    std::string tokens_path = "../models/fastconformer_ctc_export/tokens.txt";
    
    std::cout << "Initializing model..." << std::endl;
    if (!nemo.initialize(model_path, tokens_path)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "\nModel info:" << std::endl;
    std::cout << nemo.getModelInfo() << std::endl;
    
    // Test with simple audio
    std::vector<float> test_audio(16000 * 2); // 2 seconds of silence
    std::cout << "\nTranscribing 2 seconds of silence..." << std::endl;
    std::string result = nemo.transcribe(test_audio);
    std::cout << "Result: '" << result << "'" << std::endl;
    
    return 0;
}