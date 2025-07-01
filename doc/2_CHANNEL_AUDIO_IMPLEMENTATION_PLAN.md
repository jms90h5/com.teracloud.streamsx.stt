# 2-Channel Audio Support Implementation Plan

## Implementation Status

**Last Updated**: June 30, 2025

### Completed Items:
1. ✅ **C++ Channel Separation Library** (`StereoAudioSplitter`)
   - Interleaved PCM16/PCM8 splitting
   - G.711 µ-law and A-law codec support
   - Linear upsampling (8kHz → 16kHz)
   - Tested with provided 2-channel telephony WAV file

2. ✅ **SPL Type Definitions** (`Types.spl`)
   - ChannelMetadata for channel identification
   - StereoAudioChunk for pre-split audio
   - ChannelAudioStream for single channel data
   - ChannelTranscription for results with channel info
   - Note: Changed 'timestamp' to 'audioTimestamp' to avoid SPL keyword conflict

3. ✅ **Basic Testing**
   - C++ test program created and verified
   - Successfully processed 289-second stereo WAV file (RTP-Audio-narrowband-2channel-G711.wav)
   - Confirmed correct channel separation: Left RMS=0.026, Right RMS=0.047

4. ✅ **AudioChannelSplitter Operator** (partial)
   - XML model and code generation templates created
   - Location: `com.teracloud.streamsx.stt/AudioChannelSplitter/`
   - Issue: Complex code generation needs simplification

5. ✅ **StereoFileAudioSource Composite** (partial)
   - Created composite operator for reading stereo files
   - Location: `com.teracloud.streamsx.stt/StereoFileAudioSource.spl`
   - Issue: Blocked by AudioChannelSplitter compilation

6. ✅ **TwoChannelBasicTest Sample** (partial)
   - Created test application structure
   - Location: `samples/TwoChannelBasicTest.spl`
   - Issue: Blocked by AudioChannelSplitter compilation

### Current Status:
- ✅ Core C++ functionality is working correctly
- ✅ SPL AudioChannelSplitter operator fully functional
- ✅ TwoChannelBasicTest compiles and runs successfully
- ✅ Tested with 289-second stereo WAV file - proper channel separation confirmed

### Completed Architecture:
- **C++ Library**: StereoAudioSplitter handles PCM16/8, G.711 µ-law/A-law
- **SPL Operator**: AudioChannelSplitter splits stereo input into two channel streams
- **Composite Operators**: StereoFileAudioSource for convenient file reading
- **Type System**: ChannelAudioStream with metadata for channel identification

### Test Results:
- Processed 1158 tuples per channel (4.6MB each)
- Correct timestamps (250ms increments)
- Proper channel role assignment (caller/agent)
- Channel 0 (caller): RMS = 0.026
- Channel 1 (agent): RMS = 0.047

### Next Steps:
- Create full 2-channel transcription demo with NeMoSTT
- Add support for real-time streaming scenarios
- Performance optimization and benchmarking

## Overview

This document provides a detailed implementation plan for adding 2-channel (stereo) audio support to the com.teracloud.streamsx.stt toolkit. The implementation follows **Option A**: Channel separation at the operator level, similar to IBM STT Gateway's approach.

## Implementation Approach: Option A - Operator-Level Channel Separation

### Key Principles
1. Each audio channel is processed by a separate STT operator instance
2. Channels are split early in the processing pipeline
3. Channel identity (caller/agent) is preserved throughout
4. Parallel processing for optimal performance
5. Support for telephony-specific formats (8kHz, G.711)

## Progress Tracking Checklist

### Phase 1: Core Channel Separation ✓ Check when complete
- [x] Create `StereoAudioSplitter` C++ class
- [x] Implement interleaved stereo splitting
- [x] Implement non-interleaved stereo splitting
- [x] Add G.711 codec support (µ-law and A-law)
- [x] Create unit tests for splitter
- [x] Document C++ API

### Phase 2: SPL Types and Schemas
- [x] Define `ChannelMetadata` type
- [x] Define `StereoAudioChunk` type
- [x] Define `ChannelAudioStream` type
- [x] Create type documentation
- [x] Add to toolkit namespace

### Phase 3: AudioChannelSplitter Operator
- [ ] Create operator directory structure
- [ ] Write operator model XML
- [ ] Implement C++ operator code
- [ ] Create code generation templates
- [ ] Add operator tests
- [ ] Write operator documentation

### Phase 4: StereoFileAudioSource Operator
- [ ] Create operator structure
- [ ] Implement stereo WAV file reading
- [ ] Add channel role assignment
- [ ] Support configurable block sizes
- [ ] Test with provided samples
- [ ] Document usage

### Phase 5: Integration and Testing
- [ ] Create sample applications
- [ ] Test with RTP-Audio-narrowband-2channel-G711.wav
- [ ] Performance benchmarking
- [ ] Memory usage optimization
- [ ] Documentation updates

## Detailed Implementation Steps

### Step 1: Create C++ Channel Separation Library ✅ COMPLETED

#### 1.1 Create Header File ✅
Location: `impl/include/StereoAudioSplitter.hpp`

```cpp
#ifndef STEREO_AUDIO_SPLITTER_HPP
#define STEREO_AUDIO_SPLITTER_HPP

#include <vector>
#include <cstdint>
#include <memory>

namespace com::teracloud::streamsx::stt {

class StereoAudioSplitter {
public:
    struct ChannelBuffers {
        std::vector<float> left;
        std::vector<float> right;
    };
    
    struct SplitOptions {
        bool normalizeFloat = true;  // Convert to [-1.0, 1.0]
        bool applyDithering = false; // For bit depth conversion
        int targetSampleRate = 0;    // 0 = no resampling
    };
    
    // Split interleaved stereo PCM data
    static ChannelBuffers splitInterleavedPCM16(
        const int16_t* interleavedData, 
        size_t numSamples,
        const SplitOptions& options = {});
    
    // Split interleaved stereo PCM data (8-bit)
    static ChannelBuffers splitInterleavedPCM8(
        const uint8_t* interleavedData,
        size_t numSamples,
        const SplitOptions& options = {});
    
    // Split non-interleaved stereo data
    static ChannelBuffers splitNonInterleaved(
        const int16_t* leftData,
        const int16_t* rightData,
        size_t numSamplesPerChannel,
        const SplitOptions& options = {});
        
    // G.711 µ-law stereo to PCM
    static ChannelBuffers splitG711uLaw(
        const uint8_t* g711Data,
        size_t numBytes,
        bool isInterleaved = true);
    
    // G.711 A-law stereo to PCM  
    static ChannelBuffers splitG711aLaw(
        const uint8_t* g711Data,
        size_t numBytes,
        bool isInterleaved = true);
        
    // Utility: Resample audio (for 8kHz to 16kHz conversion)
    static std::vector<float> resample(
        const std::vector<float>& input,
        int inputRate,
        int outputRate);
        
private:
    // G.711 decoding tables
    static const int16_t ULAW_TO_PCM[256];
    static const int16_t ALAW_TO_PCM[256];
    
    // Helper methods
    static int16_t ulawToPcm(uint8_t ulaw);
    static int16_t alawToPcm(uint8_t alaw);
};

} // namespace

#endif // STEREO_AUDIO_SPLITTER_HPP
```

#### 1.2 Create Implementation File
Location: `impl/src/StereoAudioSplitter.cpp`

```cpp
#include "StereoAudioSplitter.hpp"
#include <algorithm>
#include <cmath>

namespace com::teracloud::streamsx::stt {

// G.711 µ-law decoding table
const int16_t StereoAudioSplitter::ULAW_TO_PCM[256] = {
    // Table values here...
    // Implementation will include full lookup table
};

// G.711 A-law decoding table  
const int16_t StereoAudioSplitter::ALAW_TO_PCM[256] = {
    // Table values here...
    // Implementation will include full lookup table
};

StereoAudioSplitter::ChannelBuffers 
StereoAudioSplitter::splitInterleavedPCM16(
    const int16_t* interleavedData, 
    size_t numSamples,
    const SplitOptions& options) {
    
    ChannelBuffers result;
    size_t numSamplesPerChannel = numSamples / 2;
    
    result.left.reserve(numSamplesPerChannel);
    result.right.reserve(numSamplesPerChannel);
    
    // Split interleaved samples: L R L R L R -> L L L, R R R
    for (size_t i = 0; i < numSamples; i += 2) {
        float leftSample = interleavedData[i] / 32768.0f;
        float rightSample = interleavedData[i + 1] / 32768.0f;
        
        result.left.push_back(leftSample);
        result.right.push_back(rightSample);
    }
    
    // Apply resampling if needed
    if (options.targetSampleRate > 0 && options.targetSampleRate != 8000) {
        // TODO: Implement resampling
    }
    
    return result;
}

// Additional implementations...

} // namespace
```

### Step 2: Define SPL Types ✅ COMPLETED

#### 2.1 Create Types SPL File ✅
Location: `com.teracloud.streamsx.stt/Types.spl`

```spl
namespace com.teracloud.streamsx.stt;

/**
 * Metadata for identifying audio channels in multi-channel streams
 */
type ChannelMetadata = tuple<
    int32 channelNumber,      // 0-based channel index
    rstring channelRole,      // "caller", "agent", "left", "right", "unknown"
    rstring phoneNumber,      // Optional phone number for telephony
    rstring speakerId,        // Optional speaker identifier
    map<rstring,rstring> additionalMetadata
>;

/**
 * Stereo audio chunk with separate channel data
 */
type StereoAudioChunk = tuple<
    blob leftChannelData,     // PCM audio data for left channel
    blob rightChannelData,    // PCM audio data for right channel
    uint64 timestamp,         // Timestamp in milliseconds
    int32 sampleRate,         // Sample rate (e.g., 8000, 16000)
    int32 bitsPerSample,      // Bits per sample (8, 16, 24, 32)
    rstring encoding          // "pcm", "ulaw", "alaw"
>;

/**
 * Single channel audio stream with metadata
 */
type ChannelAudioStream = tuple<
    blob audioData,           // PCM audio data for single channel
    uint64 timestamp,         // Timestamp in milliseconds
    ChannelMetadata channelInfo,
    int32 sampleRate,
    int32 bitsPerSample
>;

/**
 * Transcription result with channel information
 */
type ChannelTranscription = tuple<
    rstring text,             // Transcribed text
    float64 confidence,       // Confidence score
    boolean isFinal,          // Is this a final transcription
    ChannelMetadata channelInfo,
    uint64 startTime,         // Start time of transcribed segment
    uint64 endTime           // End time of transcribed segment
>;
```

### Step 3: AudioChannelSplitter Operator

#### 3.1 Operator Model XML
Location: `com.teracloud.streamsx.stt/AudioChannelSplitter/AudioChannelSplitter.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<operatorModel xmlns="http://www.ibm.com/xmlns/prod/streams/spl/operator">
  <cppOperatorModel>
    <context>
      <description>
        Splits stereo audio input into separate left and right channel streams.
        Supports various stereo formats including interleaved PCM and G.711 codecs.
      </description>
      <customLiterals>
        <enumeration>
          <name>StereoFormat</name>
          <value>interleaved</value>
          <value>nonInterleaved</value>
          <value>separateStreams</value>
        </enumeration>
        <enumeration>
          <name>AudioEncoding</name>
          <value>pcm</value>
          <value>ulaw</value>
          <value>alaw</value>
        </enumeration>
      </customLiterals>
      <libraryDependencies>
        <library>
          <cmn:description>Stereo audio processing library</cmn:description>
          <cmn:managedLibrary>
            <cmn:lib>s2t_impl</cmn:lib>
            <cmn:libPath>../../impl/lib</cmn:libPath>
            <cmn:includePath>../../impl/include</cmn:includePath>
          </cmn:managedLibrary>
        </library>
      </libraryDependencies>
    </context>
    <parameters>
      <parameter>
        <name>stereoFormat</name>
        <description>Format of the input stereo audio</description>
        <optional>true</optional>
        <type>StereoFormat</type>
        <cardinality>1</cardinality>
      </parameter>
      <parameter>
        <name>encoding</name>
        <description>Audio encoding type</description>
        <optional>true</optional>
        <type>AudioEncoding</type>
        <cardinality>1</cardinality>
      </parameter>
      <parameter>
        <name>leftChannelRole</name>
        <description>Role identifier for left channel</description>
        <optional>true</optional>
        <type>rstring</type>
        <cardinality>1</cardinality>
      </parameter>
      <parameter>
        <name>rightChannelRole</name>
        <description>Role identifier for right channel</description>
        <optional>true</optional>
        <type>rstring</type>
        <cardinality>1</cardinality>
      </parameter>
    </parameters>
    <inputPorts>
      <inputPortSet>
        <description>Stereo audio input stream</description>
        <windowingMode>NonWindowed</windowingMode>
        <cardinality>1</cardinality>
        <optional>false</optional>
      </inputPortSet>
    </inputPorts>
    <outputPorts>
      <outputPortSet>
        <description>Left channel output</description>
        <windowPunctuationOutputMode>Preserving</windowPunctuationOutputMode>
        <cardinality>1</cardinality>
        <optional>false</optional>
      </outputPortSet>
      <outputPortSet>
        <description>Right channel output</description>
        <windowPunctuationOutputMode>Preserving</windowPunctuationOutputMode>
        <cardinality>1</cardinality>
        <optional>false</optional>
      </outputPortSet>
    </outputPorts>
  </cppOperatorModel>
</operatorModel>
```

#### 3.2 Operator Header Template
Location: `com.teracloud.streamsx.stt/AudioChannelSplitter/AudioChannelSplitter_h.cgt`

```perl
<%
    my $leftChannelRole = $model->getParameterByName("leftChannelRole");
    my $rightChannelRole = $model->getParameterByName("rightChannelRole");
%>

#include <StereoAudioSplitter.hpp>
#include <memory>

<%SPL::CodeGen::headerPrologue($model);%>

class MY_OPERATOR : public MY_BASE_OPERATOR 
{
public:
    MY_OPERATOR();
    virtual ~MY_OPERATOR();
    
    void allPortsReady();
    void prepareToShutdown();
    
    void process(Tuple const & tuple, uint32_t port);
    void process(Punctuation const & punct, uint32_t port);
    
private:
    // Channel roles
    std::string leftChannelRole_;
    std::string rightChannelRole_;
    
    // Audio format parameters
    bool isInterleaved_;
    std::string encoding_;
    
    // Processing state
    uint64_t currentTimestamp_;
    int32_t sampleRate_;
    int32_t bitsPerSample_;
    
    // Helper methods
    void processStereoChunk(const blob& audioData);
    void outputChannels(const StereoAudioSplitter::ChannelBuffers& channels);
};

<%SPL::CodeGen::headerEpilogue($model);%>
```

### Step 4: StereoFileAudioSource Operator

#### 4.1 Operator SPL Implementation
Location: `com.teracloud.streamsx.stt/StereoFileAudioSource.spl`

```spl
namespace com.teracloud.streamsx.stt;

/**
 * Reads stereo audio files and outputs separate channel streams.
 * Supports WAV files with various sample rates and bit depths.
 * 
 * @input none
 * @output LeftChannel - Audio stream for left channel
 * @output RightChannel - Audio stream for right channel
 * 
 * @param filename - Path to stereo audio file
 * @param blockSize - Size of audio chunks in bytes
 * @param leftChannelRole - Role identifier for left channel (default: "caller")
 * @param rightChannelRole - Role identifier for right channel (default: "agent")
 */
public composite StereoFileAudioSource(output LeftChannel, RightChannel) {
    param
        expression<rstring> $filename;
        expression<uint32> $blockSize: 8000u;  // Default 0.5s at 8kHz stereo
        expression<rstring> $leftChannelRole: "caller";
        expression<rstring> $rightChannelRole: "agent";
        
    type
        static LeftOutputType = ChannelAudioStream;
        static RightOutputType = ChannelAudioStream;
        
    graph
        // Read stereo file as binary blocks
        stream<blob data> RawAudio = FileSource() {
            param
                file: $filename;
                format: block;
                blockSize: $blockSize;
                initDelay: 0.0;
        }
        
        // Parse WAV header and split channels
        (stream<LeftOutputType> LeftChannel;
         stream<RightOutputType> RightChannel) = AudioChannelSplitter(RawAudio) {
            param
                stereoFormat: interleaved;
                encoding: pcm;
                leftChannelRole: $leftChannelRole;
                rightChannelRole: $rightChannelRole;
        }
}
```

### Step 5: Sample Applications

#### 5.1 Basic 2-Channel Transcription
Location: `samples/TwoChannelTranscription.spl`

```spl
use com.teracloud.streamsx.stt::*;

/**
 * Demonstrates 2-channel audio transcription with channel separation.
 * Processes caller and agent audio independently.
 */
composite TwoChannelTranscription {
    param
        expression<rstring> $audioFile: 
            getSubmissionTimeValue("audioFile", 
                "audio/RTP-Audio-narrowband-2channel-G711.wav");
                
    graph
        // Read stereo audio file and split channels
        (stream<ChannelAudioStream> CallerAudio;
         stream<ChannelAudioStream> AgentAudio) = StereoFileAudioSource() {
            param
                filename: $audioFile;
                blockSize: 16000u;  // 1 second at 8kHz stereo
                leftChannelRole: "caller";
                rightChannelRole: "agent";
        }
        
        // Transcribe caller channel
        stream<rstring transcription> CallerText = NeMoSTT(CallerAudio) {
            param
                modelPath: getEnvironmentVariable("STREAMS_STT_MODEL_PATH");
                sampleRate: 8000;  // Telephony audio
        }
        
        // Transcribe agent channel
        stream<rstring transcription> AgentText = NeMoSTT(AgentAudio) {
            param
                modelPath: getEnvironmentVariable("STREAMS_STT_MODEL_PATH");
                sampleRate: 8000;
        }
        
        // Display results with channel identification
        () as CallerDisplay = Custom(CallerText) {
            logic
                state: {
                    mutable rstring fullTranscript = "";
                }
                onTuple CallerText: {
                    printStringLn("[CALLER] " + transcription);
                    fullTranscript = fullTranscript + " " + transcription;
                }
                onPunct CallerText: {
                    if (currentPunct() == Sys.FinalMarker) {
                        printStringLn("\n[CALLER COMPLETE] " + fullTranscript);
                    }
                }
        }
        
        () as AgentDisplay = Custom(AgentText) {
            logic
                state: {
                    mutable rstring fullTranscript = "";
                }
                onTuple AgentText: {
                    printStringLn("[AGENT] " + transcription);
                    fullTranscript = fullTranscript + " " + transcription;
                }
                onPunct AgentText: {
                    if (currentPunct() == Sys.FinalMarker) {
                        printStringLn("\n[AGENT COMPLETE] " + fullTranscript);
                    }
                }
        }
        
        // Save transcriptions to separate files
        () as CallerFile = FileSink(CallerText) {
            param
                file: "caller_transcript.txt";
                format: txt;
        }
        
        () as AgentFile = FileSink(AgentText) {
            param
                file: "agent_transcript.txt";
                format: txt;
        }
}
```

### Step 6: Testing Plan

#### 6.1 Unit Tests for Channel Splitter
Location: `test/test_stereo_splitter.cpp`

```cpp
#include <gtest/gtest.h>
#include "StereoAudioSplitter.hpp"

using namespace com::teracloud::streamsx::stt;

TEST(StereoAudioSplitterTest, SplitInterleavedPCM16) {
    // Test data: interleaved L R L R
    int16_t testData[] = {1000, -1000, 2000, -2000, 3000, -3000};
    
    auto result = StereoAudioSplitter::splitInterleavedPCM16(
        testData, 6);
    
    ASSERT_EQ(result.left.size(), 3);
    ASSERT_EQ(result.right.size(), 3);
    
    // Check left channel
    EXPECT_NEAR(result.left[0], 1000.0f/32768.0f, 0.0001f);
    EXPECT_NEAR(result.left[1], 2000.0f/32768.0f, 0.0001f);
    EXPECT_NEAR(result.left[2], 3000.0f/32768.0f, 0.0001f);
    
    // Check right channel
    EXPECT_NEAR(result.right[0], -1000.0f/32768.0f, 0.0001f);
    EXPECT_NEAR(result.right[1], -2000.0f/32768.0f, 0.0001f);
    EXPECT_NEAR(result.right[2], -3000.0f/32768.0f, 0.0001f);
}

TEST(StereoAudioSplitterTest, SplitG711uLaw) {
    // Test G.711 µ-law decoding
    uint8_t ulawData[] = {0xFF, 0x00, 0x7F, 0x80};  // Test values
    
    auto result = StereoAudioSplitter::splitG711uLaw(
        ulawData, 4);
    
    ASSERT_EQ(result.left.size(), 2);
    ASSERT_EQ(result.right.size(), 2);
    
    // Verify decoding produced valid PCM values
    EXPECT_NE(result.left[0], 0.0f);
    EXPECT_NE(result.right[0], 0.0f);
}
```

#### 6.2 Integration Test Script
Location: `test/test_2channel_integration.sh`

```bash
#!/bin/bash

# Test 2-channel audio processing

echo "=== 2-Channel Audio Integration Test ==="

# Test with provided sample
TEST_FILE="../OneDrive_2_6-30-2025/RTP-Audio-narrowband-2channel-G711.wav"

if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

# Build the sample
echo "Building TwoChannelTranscription..."
sc -M TwoChannelTranscription -t ../

# Run the test
echo "Running transcription test..."
./output/bin/standalone -d . audioFile="$TEST_FILE"

# Check outputs
if [ -f "caller_transcript.txt" ] && [ -f "agent_transcript.txt" ]; then
    echo "SUCCESS: Both channel transcripts generated"
    echo "Caller transcript:"
    cat caller_transcript.txt
    echo "Agent transcript:"
    cat agent_transcript.txt
else
    echo "ERROR: Missing transcript files"
    exit 1
fi
```

### Step 7: Build Integration

#### 7.1 Update Makefile
Location: `impl/Makefile` (additions)

```makefile
# Add stereo audio sources
SOURCES += src/StereoAudioSplitter.cpp

# Add test for stereo splitter
test-stereo: $(LIB)
	g++ -o test_stereo test/test_stereo_splitter.cpp \
	    -L./lib -ls2t_impl -lgtest -lpthread \
	    -I./include -I/usr/include/gtest
	./test_stereo
```

### Recovery Points

If implementation is interrupted, resume from:

1. **After C++ library**: Continue with SPL types (Step 2)
2. **After types defined**: Continue with AudioChannelSplitter operator (Step 3)
3. **After splitter operator**: Continue with StereoFileAudioSource (Step 4)
4. **After file source**: Continue with sample applications (Step 5)
5. **After samples**: Continue with testing (Step 6)

### Known Issues and Solutions

1. **WAV Header Parsing**: Initial FileSource reads include WAV header
   - Solution: Skip first 44 bytes or implement proper WAV parser

2. **8kHz to 16kHz Resampling**: NeMo models trained on 16kHz
   - Solution: Implement resampling in StereoAudioSplitter

3. **G.711 Codec Tables**: Need complete lookup tables
   - Solution: Use ITU-T G.711 specification tables

4. **Channel Synchronization**: Channels may drift in real-time
   - Solution: Add timestamp-based synchronization

### Next Steps After Implementation

1. Performance optimization for real-time processing
2. Add support for more audio formats (MP3, Opus)
3. Implement channel role auto-detection
4. Add Voice Gateway integration
5. Create monitoring dashboard for multi-channel streams

## Conclusion

This plan provides a complete roadmap for implementing 2-channel audio support. The modular approach allows for incremental development and testing, with clear recovery points if work is interrupted.