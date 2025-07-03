#include "backends/STTBackendAdapter.hpp"
#include "backends/NeMoSTTAdapter.hpp"
#include "backends/WatsonSTTAdapter.hpp"
// Future includes:
// #include "backends/GoogleSTTAdapter.hpp"
// #include "backends/AzureSTTAdapter.hpp"

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
        
        // Register Watson backend (placeholder for now)
        s_backendRegistry["watson"] = [](const BackendConfig& config) {
            return std::make_unique<WatsonSTTAdapter>();
        };
        
        // Future backends will be registered here:
        // s_backendRegistry["google"] = [](const BackendConfig& config) {
        //     return std::make_unique<GoogleSTTAdapter>();
        // };
        
        registered = true;
    }
}

std::unique_ptr<STTBackendAdapter> STTBackendFactory::createBackend(
    const std::string& backendType,
    const BackendConfig& config) {
    
    registerBuiltinBackends();
    
    // Debug output
    std::cerr << "STTBackendFactory: Requested backend type: '" << backendType << "'" << std::endl;
    std::cerr << "STTBackendFactory: Available backends: ";
    for (const auto& kv : s_backendRegistry) {
        std::cerr << "'" << kv.first << "' ";
    }
    std::cerr << std::endl;
    
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
        std::cerr << "STTBackendFactory: Creating backend instance..." << std::endl;
        auto backend = it->second(config);
        
        if (!backend) {
            std::cerr << "STTBackendFactory: Factory function returned nullptr" << std::endl;
            return nullptr;
        }
        
        std::cerr << "STTBackendFactory: Backend instance created, initializing..." << std::endl;
        
        // Initialize it
        if (!backend->initialize(config)) {
            std::cerr << "STTBackendFactory: Failed to initialize backend: " 
                      << backendType << std::endl;
            return nullptr;
        }
        
        std::cerr << "STTBackendFactory: Backend initialized successfully" << std::endl;
        return backend;
        
    } catch (const std::exception& e) {
        std::cerr << "STTBackendFactory: Exception caught - Error creating backend " << backendType 
                  << ": " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "STTBackendFactory: Unknown exception caught while creating backend " 
                  << backendType << std::endl;
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