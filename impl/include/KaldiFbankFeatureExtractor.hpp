#ifndef KALDI_FBANK_FEATURE_EXTRACTOR_HPP
#define KALDI_FBANK_FEATURE_EXTRACTOR_HPP

#include <vector>
#include <memory>
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

/**
 * Feature extractor using kaldi-native-fbank library
 * Provides librosa-compatible mel spectrograms for NeMo models
 */
class KaldiFbankFeatureExtractor {
public:
    KaldiFbankFeatureExtractor();
    ~KaldiFbankFeatureExtractor() = default;

    // Extract mel spectrogram features for NeMo models
    std::vector<float> extractMelSpectrogram(const std::vector<float>& audio_data);
    
    // Get expected number of frames for given audio length
    int getNumFrames(int audio_length) const;

private:
    std::unique_ptr<knf::OnlineFbank> fbank_;
    knf::FbankOptions fbank_opts_;
    
    void setupFbankOptions();
};

#endif // KALDI_FBANK_FEATURE_EXTRACTOR_HPP