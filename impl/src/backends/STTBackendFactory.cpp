#include "backends/STTBackendAdapter.hpp"
#include "backends/NeMoSTTAdapter.hpp"
// Future includes:
// #include "backends/WatsonSTTAdapter.hpp"
// #include "backends/GoogleSTTAdapter.hpp"

#include <unordered_map>
#include <functional>
#include <algorithm>
#include <iostream>

namespace com {
namespace teracloud {
namespace streamsx {
namespace stt {

// Static registry for backend factories
static std::unordered_map<std::string, STTBackendFactory::FactoryFunc> s_backendRegistry;

// Register built-in backends
static void registerBuiltinBackends() {
    static bool registered = false;
    if (!registered) {
        // Register NeMo backend
        s_backendRegistry["nemo"] = [](const BackendConfig& config) {
            return std::make_unique<NeMoSTTAdapter>();
        };
        
        // Future backends will be registered here:
        // s_backendRegistry["watson"] = [](const BackendConfig& config) {
        //     return std::make_unique<WatsonSTTAdapter>();
        // };
        
        registered = true;
    }
}

std::unique_ptr<STTBackendAdapter> STTBackendFactory::createBackend(
    const std::string& backendType,
    const BackendConfig& config) {
    
    registerBuiltinBackends();
    
    // Convert to lowercase for case-insensitive matching
    std::string lowerType = backendType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    auto it = s_backendRegistry.find(lowerType);
    if (it == s_backendRegistry.end()) {
        std::cerr << "STTBackendFactory: Unknown backend type: " << backendType << std::endl;
        return nullptr;
    }
    
    try {
        // Create the backend
        auto backend = it->second(config);
        
        // Initialize it
        if (backend && !backend->initialize(config)) {
            std::cerr << "STTBackendFactory: Failed to initialize backend: " 
                      << backendType << std::endl;
            return nullptr;
        }
        
        return backend;
        
    } catch (const std::exception& e) {
        std::cerr << "STTBackendFactory: Error creating backend " << backendType 
                  << ": " << e.what() << std::endl;
        return nullptr;
    }
}

std::vector<std::string> STTBackendFactory::getAvailableBackends() {
    registerBuiltinBackends();
    
    std::vector<std::string> backends;
    backends.reserve(s_backendRegistry.size());
    
    for (const auto& pair : s_backendRegistry) {
        backends.push_back(pair.first);
    }
    
    std::sort(backends.begin(), backends.end());
    return backends;
}

bool STTBackendFactory::isBackendAvailable(const std::string& backendType) {
    registerBuiltinBackends();
    
    std::string lowerType = backendType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    return s_backendRegistry.find(lowerType) != s_backendRegistry.end();
}

void STTBackendFactory::registerBackend(
    const std::string& backendType,
    FactoryFunc factoryFunc) {
    
    registerBuiltinBackends();
    
    std::string lowerType = backendType;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    s_backendRegistry[lowerType] = factoryFunc;
    
    std::cout << "STTBackendFactory: Registered backend: " << backendType << std::endl;
}

} // namespace stt
} // namespace streamsx
} // namespace teracloud
} // namespace com