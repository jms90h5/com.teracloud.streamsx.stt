#include "KaldiFbankFeatureExtractor.hpp"
#include <iostream>
#include <cmath>

KaldiFbankFeatureExtractor::KaldiFbankFeatureExtractor() {
    setupFbankOptions();
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
}

void KaldiFbankFeatureExtractor::setupFbankOptions() {
    // Configure to match librosa/NeMo parameters exactly
    fbank_opts_.frame_opts.samp_freq = 16000.0f;         // 16kHz sample rate
    fbank_opts_.frame_opts.frame_length_ms = 25.0f;     // 400 samples = 25ms window
    fbank_opts_.frame_opts.frame_shift_ms = 10.0f;      // 160 samples = 10ms hop
    fbank_opts_.frame_opts.dither = 0.0f;               // No dithering for consistency
    fbank_opts_.frame_opts.preemph_coeff = 0.0f;        // No pre-emphasis
    fbank_opts_.frame_opts.remove_dc_offset = false;    // Don't remove DC
    fbank_opts_.frame_opts.window_type = "hanning";     // Hann window like librosa
    fbank_opts_.frame_opts.round_to_power_of_two = true; // Use power-of-2 FFT size
    fbank_opts_.frame_opts.snip_edges = false;         // Use center padding like librosa
    
    // Mel filterbank options
    fbank_opts_.mel_opts.num_bins = 80;                 // 80 mel bins
    fbank_opts_.mel_opts.low_freq = 0.0f;               // 0 Hz low cutoff
    fbank_opts_.mel_opts.high_freq = 8000.0f;           // 8kHz high cutoff (Nyquist for 16kHz)
    fbank_opts_.mel_opts.vtln_low = 100.0f;
    fbank_opts_.mel_opts.vtln_high = -500.0f;
    
    // Energy and log options
    fbank_opts_.use_energy = false;                     // Don't add energy feature
    fbank_opts_.energy_floor = 0.0f;
    fbank_opts_.raw_energy = true;
    fbank_opts_.htk_compat = false;                     // Use slaney normalization like librosa
    fbank_opts_.use_log_fbank = true;                   // Apply log transform
    fbank_opts_.use_power = true;                       // Use power spectrum
    
    std::cout << "Kaldi Fbank configured for NeMo/librosa compatibility:" << std::endl;
    std::cout << "  Sample rate: " << fbank_opts_.frame_opts.samp_freq << " Hz" << std::endl;
    std::cout << "  Frame length: " << fbank_opts_.frame_opts.frame_length_ms << " ms" << std::endl;
    std::cout << "  Frame shift: " << fbank_opts_.frame_opts.frame_shift_ms << " ms" << std::endl;
    std::cout << "  Mel bins: " << fbank_opts_.mel_opts.num_bins << std::endl;
    std::cout << "  Frequency range: " << fbank_opts_.mel_opts.low_freq 
              << " - " << fbank_opts_.mel_opts.high_freq << " Hz" << std::endl;
}

std::vector<float> KaldiFbankFeatureExtractor::extractMelSpectrogram(const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        std::cerr << "Warning: Empty audio data" << std::endl;
        return {};
    }
    
    std::cout << "Extracting features from " << audio_data.size() << " audio samples" << std::endl;
    
    // Reset the feature extractor for new audio
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
    
    // Feed audio data to feature extractor
    fbank_->AcceptWaveform(fbank_opts_.frame_opts.samp_freq, audio_data.data(), audio_data.size());
    
    // Signal end of input
    fbank_->InputFinished();
    
    // Get number of frames ready
    int32_t num_frames = fbank_->NumFramesReady();
    int32_t feature_dim = fbank_->Dim();
    
    std::cout << "Kaldi extracted " << num_frames << " frames x " << feature_dim << " features" << std::endl;
    
    if (num_frames == 0) {
        std::cerr << "Warning: No frames extracted" << std::endl;
        return {};
    }
    
    // Extract all features and transpose to match Python/NeMo format
    std::vector<float> features(num_frames * feature_dim);
    
    // Kaldi gives [time, features], but we need [features, time] like Python
    for (int32_t t = 0; t < num_frames; t++) {
        const float* frame_data = fbank_->GetFrame(t);
        
        for (int32_t f = 0; f < feature_dim; f++) {
            // Transpose: [time, features] -> [features, time]
            int kaldi_idx = t * feature_dim + f;    // [time, features]
            int python_idx = f * num_frames + t;    // [features, time]
            features[python_idx] = frame_data[f];
        }
    }
    
    // Debug: Check feature range
    if (!features.empty()) {
        float min_val = features[0];
        float max_val = features[0];
        for (float val : features) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        std::cout << "Feature range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "Expected: [-15.03, 3.06] (like Python)" << std::endl;
        
        // Show first few features
        std::cout << "First 5 features: ";
        for (int i = 0; i < 5 && i < features.size(); i++) {
            std::cout << features[i] << " ";
        }
        std::cout << std::endl;
    }
    
    return features;
}

int KaldiFbankFeatureExtractor::getNumFrames(int audio_length) const {
    // Calculate expected number of frames
    // This matches librosa's calculation with center=True (default)
    float sample_rate = fbank_opts_.frame_opts.samp_freq;
    float frame_shift_ms = fbank_opts_.frame_opts.frame_shift_ms;
    float frame_length_ms = fbank_opts_.frame_opts.frame_length_ms;
    
    int frame_shift_samples = static_cast<int>(sample_rate * frame_shift_ms / 1000.0f);
    int frame_length_samples = static_cast<int>(sample_rate * frame_length_ms / 1000.0f);
    
    // With center padding (like librosa center=True)
    int pad_length = frame_length_samples / 2;
    int padded_length = audio_length + 2 * pad_length;
    int num_frames = 1 + (padded_length - frame_length_samples) / frame_shift_samples;
    
    return num_frames;
}