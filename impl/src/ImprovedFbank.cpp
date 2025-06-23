#include "../include/ImprovedFbank.hpp"
#include "../include/kaldi-native-fbank/csrc/rfft.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>

namespace improved_fbank {

FbankComputer::FbankComputer(const Options& opts) : opts_(opts), cmvn_available_(false) {
    // Convert ms to samples
    frame_length_samples_ = (opts_.sample_rate * opts_.frame_length_ms) / 1000;
    frame_shift_samples_ = (opts_.sample_rate * opts_.frame_shift_ms) / 1000;
    
    initializeWindow();
    initializeMelFilterbank();
    
    std::cout << "ImprovedFbank initialized:" << std::endl;
    std::cout << "  Sample rate: " << opts_.sample_rate << " Hz" << std::endl;
    std::cout << "  Frame length: " << frame_length_samples_ << " samples (" << opts_.frame_length_ms << "ms)" << std::endl;
    std::cout << "  Frame shift: " << frame_shift_samples_ << " samples (" << opts_.frame_shift_ms << "ms)" << std::endl;
    std::cout << "  Mel bins: " << opts_.num_mel_bins << std::endl;
    std::cout << "  FFT size: " << opts_.n_fft << std::endl;
}

void FbankComputer::initializeWindow() {
    // Use Hann window for NeMo compatibility (window: hann in config)
    window_.resize(frame_length_samples_);
    for (int i = 0; i < frame_length_samples_; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (frame_length_samples_ - 1)));
    }
}

void FbankComputer::initializeMelFilterbank() {
    int num_fft_bins = opts_.n_fft / 2 + 1;
    
    // Convert frequencies to mel scale
    float low_mel = melScale(opts_.low_freq);
    float high_mel = melScale(opts_.high_freq);
    
    // Create equally spaced mel points
    std::vector<float> mel_points(opts_.num_mel_bins + 2);
    for (int i = 0; i < opts_.num_mel_bins + 2; ++i) {
        mel_points[i] = low_mel + (high_mel - low_mel) * i / (opts_.num_mel_bins + 1);
    }
    
    // Convert mel points back to Hz
    std::vector<float> hz_points(opts_.num_mel_bins + 2);
    for (int i = 0; i < opts_.num_mel_bins + 2; ++i) {
        hz_points[i] = invMelScale(mel_points[i]);
    }
    
    // Convert Hz to FFT bin numbers
    std::vector<int> bin_points(opts_.num_mel_bins + 2);
    for (int i = 0; i < opts_.num_mel_bins + 2; ++i) {
        // Use proper rounding instead of floor to avoid zero-width filters
        bin_points[i] = static_cast<int>(std::round(opts_.n_fft * hz_points[i] / opts_.sample_rate));
    }
    
    // Create mel filterbank matrix
    mel_filterbank_.resize(opts_.num_mel_bins);
    for (int mel = 0; mel < opts_.num_mel_bins; ++mel) {
        mel_filterbank_[mel].resize(num_fft_bins, 0.0f);
        
        int left = bin_points[mel];
        int center = bin_points[mel + 1];
        int right = bin_points[mel + 2];
        
        // Ensure we don't have zero-width filters
        if (left == center) center = left + 1;
        if (center == right) right = center + 1;
        
        // Left slope
        for (int bin = left; bin < center; ++bin) {
            if (bin >= 0 && bin < num_fft_bins && center > left) {
                mel_filterbank_[mel][bin] = static_cast<float>(bin - left) / (center - left);
            }
        }
        
        // Right slope
        for (int bin = center; bin < right; ++bin) {
            if (bin >= 0 && bin < num_fft_bins && right > center) {
                mel_filterbank_[mel][bin] = static_cast<float>(right - bin) / (right - center);
            }
        }
    }
}

float FbankComputer::melScale(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

float FbankComputer::invMelScale(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> FbankComputer::computeFFT(const std::vector<float>& frame) {
    // Use proper FFT implementation from kaldi-native-fbank
    int fft_size = opts_.n_fft;
    std::vector<float> padded_frame(fft_size, 0.0f);
    
    // Copy and pad frame
    int copy_len = std::min(static_cast<int>(frame.size()), fft_size);
    for (int i = 0; i < copy_len; ++i) {
        padded_frame[i] = frame[i];
    }
    
    // Create RFFT computer
    knf::Rfft rfft(fft_size);
    
    // Compute FFT in-place
    rfft.Compute(padded_frame.data());
    
    // Convert to power spectrum
    // Based on RFFT output format:
    // padded_frame[0] = R[0]
    // padded_frame[1] = R[n/2]
    // for 1 < k < n/2:
    //   padded_frame[2*k] = R[k]
    //   padded_frame[2*k+1] = I[k]
    
    std::vector<float> power_spectrum(fft_size / 2 + 1);
    
    // DC component (k=0)
    power_spectrum[0] = padded_frame[0] * padded_frame[0];
    
    // Middle frequencies
    for (int k = 1; k < fft_size / 2; ++k) {
        float real = padded_frame[2 * k];
        float imag = padded_frame[2 * k + 1];
        power_spectrum[k] = real * real + imag * imag;
    }
    
    // Nyquist frequency (k=n/2)
    power_spectrum[fft_size / 2] = padded_frame[1] * padded_frame[1];
    
    return power_spectrum;
}

std::vector<float> FbankComputer::applyMelFilterbank(const std::vector<float>& power_spectrum) {
    std::vector<float> mel_energies(opts_.num_mel_bins);
    
    for (int mel = 0; mel < opts_.num_mel_bins; ++mel) {
        float energy = 0.0f;
        for (size_t bin = 0; bin < power_spectrum.size() && bin < mel_filterbank_[mel].size(); ++bin) {
            energy += power_spectrum[bin] * mel_filterbank_[mel][bin];
        }
        
        // Apply log and ensure positive values
        mel_energies[mel] = opts_.apply_log ? std::log(std::max(energy, 1e-10f)) : energy;
    }
    
    return mel_energies;
}

void FbankComputer::applyDither(std::vector<float>& audio) {
    if (opts_.dither <= 0.0f) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, opts_.dither);
    
    for (float& sample : audio) {
        sample += dist(gen);
    }
}

std::vector<std::vector<float>> FbankComputer::computeFeatures(const std::vector<float>& audio) {
    if (audio.empty()) {
        return std::vector<std::vector<float>>();
    }
    
    // Apply dither if specified
    std::vector<float> dithered_audio = audio;
    applyDither(dithered_audio);
    
    // Calculate number of frames
    int num_frames = 0;
    if (static_cast<int>(dithered_audio.size()) >= frame_length_samples_) {
        num_frames = (dithered_audio.size() - frame_length_samples_) / frame_shift_samples_ + 1;
    }
    
    if (num_frames <= 0) {
        return std::vector<std::vector<float>>();
    }
    
    std::vector<std::vector<float>> features;
    features.reserve(num_frames);
    
    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start = frame_idx * frame_shift_samples_;
        
        // Extract and window the frame
        std::vector<float> frame(frame_length_samples_);
        for (int i = 0; i < frame_length_samples_; ++i) {
            if (start + i < static_cast<int>(dithered_audio.size())) {
                frame[i] = dithered_audio[start + i] * window_[i];
            } else {
                frame[i] = 0.0f;  // Zero padding
            }
        }
        
        // Compute power spectrum via FFT
        std::vector<float> power_spectrum = computeFFT(frame);
        
        // Apply mel filterbank
        std::vector<float> mel_features = applyMelFilterbank(power_spectrum);
        
        features.push_back(mel_features);
    }
    
    // Apply CMVN if available
    if (cmvn_available_) {
        applyCMVN(features);
    } else if (opts_.normalize_per_feature && !features.empty()) {
        // Compute per-feature normalization on the fly
        // This normalizes each mel bin independently across time
        int n_frames = features.size();
        int n_features = features[0].size();
        
        // Compute mean and std for each feature
        std::vector<float> feature_means(n_features, 0.0f);
        std::vector<float> feature_stds(n_features, 0.0f);
        
        // Compute means
        for (int f = 0; f < n_features; f++) {
            float sum = 0.0f;
            for (int t = 0; t < n_frames; t++) {
                sum += features[t][f];
            }
            feature_means[f] = sum / n_frames;
        }
        
        // Compute standard deviations
        for (int f = 0; f < n_features; f++) {
            float sum_sq = 0.0f;
            for (int t = 0; t < n_frames; t++) {
                float diff = features[t][f] - feature_means[f];
                sum_sq += diff * diff;
            }
            feature_stds[f] = std::sqrt(sum_sq / n_frames + 1e-10f);
        }
        
        // Apply normalization
        for (int t = 0; t < n_frames; t++) {
            for (int f = 0; f < n_features; f++) {
                features[t][f] = (features[t][f] - feature_means[f]) / feature_stds[f];
            }
        }
    }
    
    return features;
}

void FbankComputer::setCMVNStats(const std::vector<float>& mean_stats, const std::vector<float>& var_stats, int frame_count) {
    if (mean_stats.size() != static_cast<size_t>(opts_.num_mel_bins) || 
        var_stats.size() != static_cast<size_t>(opts_.num_mel_bins)) {
        std::cerr << "CMVN stats size mismatch. Expected " << opts_.num_mel_bins 
                  << " features, got mean=" << mean_stats.size() 
                  << ", var=" << var_stats.size() << std::endl;
        return;
    }
    
    cmvn_mean_.resize(opts_.num_mel_bins);
    cmvn_var_.resize(opts_.num_mel_bins);
    
    // Convert accumulated stats to per-feature mean and variance
    for (int i = 0; i < opts_.num_mel_bins; ++i) {
        cmvn_mean_[i] = mean_stats[i] / frame_count;
        
        // Variance = (sum_x2 / N) - (mean^2)  
        float variance = (var_stats[i] / frame_count) - (cmvn_mean_[i] * cmvn_mean_[i]);
        cmvn_var_[i] = std::sqrt(std::max(variance, 1e-10f));  // Standard deviation
    }
    
    cmvn_available_ = true;
    
    std::cout << "CMVN stats loaded for " << frame_count << " frames" << std::endl;
    std::cout << "Mean range: [" << *std::min_element(cmvn_mean_.begin(), cmvn_mean_.end()) 
              << ", " << *std::max_element(cmvn_mean_.begin(), cmvn_mean_.end()) << "]" << std::endl;
    std::cout << "Std range: [" << *std::min_element(cmvn_var_.begin(), cmvn_var_.end()) 
              << ", " << *std::max_element(cmvn_var_.begin(), cmvn_var_.end()) << "]" << std::endl;
}

void FbankComputer::applyCMVN(std::vector<std::vector<float>>& features) {
    if (!cmvn_available_ || features.empty()) return;
    
    // Apply per-feature normalization: (x - mean) / std
    for (auto& frame : features) {
        for (int i = 0; i < static_cast<int>(frame.size()) && i < static_cast<int>(cmvn_mean_.size()); ++i) {
            frame[i] = (frame[i] - cmvn_mean_[i]) / cmvn_var_[i];
        }
    }
}

std::unique_ptr<FbankComputer> createNeMoCompatibleFbank(const std::string& cmvn_stats_path) {
    FbankComputer::Options opts;
    
    // Set NeMo-compatible options based on model_config.yaml
    opts.sample_rate = 16000;
    opts.num_mel_bins = 80;           // features: 80
    opts.frame_length_ms = 25;        // window_size: 0.025
    opts.frame_shift_ms = 10;         // window_stride: 0.01
    opts.n_fft = 512;                 // n_fft: 512
    opts.low_freq = 0.0f;
    opts.high_freq = 8000.0f;         // Nyquist for 16kHz
    opts.use_energy = true;
    opts.apply_log = true;
    opts.dither = 1e-5f;              // dither: 1.0e-05
    opts.normalize_per_feature = true; // normalize: per_feature
    
    auto fbank = std::make_unique<FbankComputer>(opts);
    
    // Load CMVN stats if path provided
    if (!cmvn_stats_path.empty()) {
        std::ifstream file(cmvn_stats_path);
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            
            // Parse JSON-like format from global_cmvn.stats
            // {"mean_stat": [...], "var_stat": [...], "frame_num": N}
            
            // Simple JSON parsing (production would use proper JSON library)
            size_t mean_start = line.find("\"mean_stat\": [") + 14;
            size_t mean_end = line.find("], \"var_stat\":");
            size_t var_start = line.find("\"var_stat\": [") + 13;
            size_t var_end = line.find("], \"frame_num\":");
            size_t frame_start = line.find("\"frame_num\": ") + 13;
            size_t frame_end = line.find("}", frame_start);
            
            if (mean_start != std::string::npos && var_start != std::string::npos && frame_start != std::string::npos) {
                std::string mean_str = line.substr(mean_start, mean_end - mean_start);
                std::string var_str = line.substr(var_start, var_end - var_start);
                std::string frame_str = line.substr(frame_start, frame_end - frame_start);
                
                // Parse arrays
                std::vector<float> mean_stats, var_stats;
                std::stringstream mean_ss(mean_str), var_ss(var_str);
                std::string item;
                
                while (std::getline(mean_ss, item, ',')) {
                    mean_stats.push_back(std::stof(item));
                }
                while (std::getline(var_ss, item, ',')) {
                    var_stats.push_back(std::stof(item));
                }
                
                int frame_count = std::stoi(frame_str);
                
                fbank->setCMVNStats(mean_stats, var_stats, frame_count);
            }
        } else {
            std::cerr << "Could not open CMVN stats file: " << cmvn_stats_path << std::endl;
        }
    }
    
    return fbank;
}

} // namespace improved_fbank