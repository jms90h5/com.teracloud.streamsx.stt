#include "KaldiFbankFeatureExtractor.hpp"
#include <iostream>
#include <cmath>

KaldiFbankFeatureExtractor::KaldiFbankFeatureExtractor() {
    setupFbankOptions();
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
    
    std::cout << "Kaldi Fbank configured for NeMo/librosa compatibility:" << std::endl;
    std::cout << "  Sample rate: " << fbank_opts_.frame_opts.samp_freq << " Hz" << std::endl;
    std::cout << "  Frame length: " << fbank_opts_.frame_opts.frame_length_ms << " ms" << std::endl;
    std::cout << "  Frame shift: " << fbank_opts_.frame_opts.frame_shift_ms << " ms" << std::endl;
    std::cout << "  Mel bins: " << fbank_opts_.mel_opts.num_bins << std::endl;
    std::cout << "  Frequency range: " << fbank_opts_.mel_opts.low_freq 
              << " - " << fbank_opts_.mel_opts.high_freq << " Hz" << std::endl;
}

void KaldiFbankFeatureExtractor::setupFbankOptions() {
    // Configure for NeMo compatibility (matching librosa mel spectrogram)
    fbank_opts_.frame_opts.samp_freq = 16000;
    fbank_opts_.frame_opts.frame_length_ms = 25.0;  // 400 samples
    fbank_opts_.frame_opts.frame_shift_ms = 10.0;   // 160 samples
    fbank_opts_.frame_opts.dither = 1e-5;  // Add dither
    fbank_opts_.frame_opts.remove_dc_offset = true;
    fbank_opts_.frame_opts.window_type = "hann";  // Hanning window
    fbank_opts_.frame_opts.preemph_coeff = 0.0;   // No pre-emphasis for NeMo
    
    // Mel filterbank configuration matching NeMo
    fbank_opts_.mel_opts.num_bins = 80;
    fbank_opts_.mel_opts.low_freq = 0.0;
    fbank_opts_.mel_opts.high_freq = 8000.0;  // Nyquist for 16kHz
    fbank_opts_.mel_opts.vtln_low = 100.0;
    fbank_opts_.mel_opts.vtln_high = -500.0;
    
    // Energy configuration
    fbank_opts_.use_energy = false;
    fbank_opts_.energy_floor = 1.0;
    fbank_opts_.raw_energy = true;
    fbank_opts_.htk_compat = false;
    fbank_opts_.use_log_fbank = true;  // Return log mel features
    fbank_opts_.use_power = true;      // Use power spectrum
}

std::vector<float> KaldiFbankFeatureExtractor::extractMelSpectrogram(const std::vector<float>& audio_data) {
    // Process audio through Kaldi fbank
    fbank_->AcceptWaveform(16000, audio_data.data(), audio_data.size());
    fbank_->InputFinished();
    
    int num_frames = fbank_->NumFramesReady();
    int num_mels = fbank_opts_.mel_opts.num_bins;
    
    std::vector<float> mel_features;
    mel_features.reserve(num_frames * num_mels);
    
    // Extract features frame by frame
    for (int i = 0; i < num_frames; ++i) {
        const float* frame = fbank_->GetFrame(i);
        mel_features.insert(mel_features.end(), frame, frame + num_mels);
    }
    
    // Reset for next utterance
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
    
    return mel_features;
}

int KaldiFbankFeatureExtractor::getNumFrames(int audio_length) const {
    // Calculate number of frames based on frame shift
    int frame_shift = static_cast<int>(fbank_opts_.frame_opts.frame_shift_ms * 
                                      fbank_opts_.frame_opts.samp_freq / 1000);
    int frame_length = static_cast<int>(fbank_opts_.frame_opts.frame_length_ms * 
                                        fbank_opts_.frame_opts.samp_freq / 1000);
    
    if (audio_length < frame_length) {
        return 0;
    }
    
    return 1 + (audio_length - frame_length) / frame_shift;
}