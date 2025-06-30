# Makefile for building the ONNX STT implementation library

# ONNX Runtime location
# First check if it's in the toolkit deps
ONNX_LOCAL := ../lib/onnxruntime
ifneq ($(wildcard $(ONNX_LOCAL)/lib/libonnxruntime.so),)
    ONNXRUNTIME_ROOT := $(ONNX_LOCAL)
else
    # Otherwise check standard locations
    ifneq ($(wildcard /usr/local/lib/libonnxruntime.so),)
        ONNXRUNTIME_ROOT := /usr/local
    else ifneq ($(wildcard /usr/lib/libonnxruntime.so),)
        ONNXRUNTIME_ROOT := /usr
    else
        $(error ONNX Runtime not found. Run setup_onnx_runtime.sh first)
    endif
endif

# Compiler settings
CXX := g++
CXXFLAGS := -O3 -DNDEBUG

# Source files - ONNX implementation with VAD, feature extraction, cache management, pipeline, and NeMo models
SOURCES = src/OnnxSTTImpl.cpp src/OnnxSTTInterface.cpp src/ZipformerRNNT.cpp src/SileroVAD.cpp src/KaldifeatExtractor.cpp src/CacheManager.cpp src/STTPipeline.cpp src/NeMoCacheAwareConformer.cpp src/NeMoCacheAwareStreaming.cpp src/ModelFactory.cpp src/ImprovedFbank.cpp src/ImprovedFbankAdapter.cpp src/NeMoCTCModel.cpp src/StereoAudioSplitter.cpp
# Additional source in include directory
INCLUDE_SOURCES = include/NeMoCTCImpl.cpp

# Build directory
BUILD_DIR = build

# Object files (in build directory)
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_DIR)/%.o) $(INCLUDE_SOURCES:include/%.cpp=$(BUILD_DIR)/%.o)

# Target library
LIB = lib/libs2t_impl.so

# Compiler flags
CXXFLAGS += -fPIC -std=c++14 -Wall -Wextra
CXXFLAGS += -Iinclude
CXXFLAGS += -I$(ONNXRUNTIME_ROOT)/include

# Linker flags 
LDFLAGS += -shared
LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib
LDFLAGS += -lonnxruntime
LDFLAGS += -Llib
LDFLAGS += -lkaldi-native-fbank-core
LDFLAGS += -ldl
LDFLAGS += -Wl,-rpath,'$$ORIGIN'
LDFLAGS += -Wl,-rpath,'$$ORIGIN/../lib'
LDFLAGS += -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib

# Build rules
all: $(LIB)

$(LIB): $(OBJECTS) | lib
	$(CXX) $(LDFLAGS) -o $@ $^

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

lib:
	mkdir -p lib

# Compile source files to build directory
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile include files to build directory
$(BUILD_DIR)/%.o: include/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(LIB)

.PHONY: all clean