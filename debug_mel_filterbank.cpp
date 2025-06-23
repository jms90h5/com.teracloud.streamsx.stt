/**
 * Debug mel filterbank initialization
 */

#include <iostream>
#include <vector>
#include <cmath>

float melScale(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

float invMelScale(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

int main() {
    // Match the C++ settings
    int sample_rate = 16000;
    int num_mel_bins = 80;
    float low_freq = 0.0f;
    float high_freq = 8000.0f;
    int n_fft = 512;
    int num_fft_bins = n_fft / 2 + 1;
    
    // Convert frequencies to mel scale
    float low_mel = melScale(low_freq);
    float high_mel = melScale(high_freq);
    
    std::cout << "Mel filterbank debug:" << std::endl;
    std::cout << "Low freq: " << low_freq << " Hz -> " << low_mel << " mel" << std::endl;
    std::cout << "High freq: " << high_freq << " Hz -> " << high_mel << " mel" << std::endl;
    
    // Create equally spaced mel points
    std::vector<float> mel_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        mel_points[i] = low_mel + (high_mel - low_mel) * i / (num_mel_bins + 1);
    }
    
    // Convert mel points back to Hz
    std::vector<float> hz_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        hz_points[i] = invMelScale(mel_points[i]);
    }
    
    // Convert Hz to FFT bin numbers
    std::vector<int> bin_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((n_fft + 1) * hz_points[i] / sample_rate));
    }
    
    // Print info for first few filters
    std::cout << "\nFirst 10 mel filters:" << std::endl;
    for (int mel = 0; mel < 10; ++mel) {
        int left = bin_points[mel];
        int center = bin_points[mel + 1];
        int right = bin_points[mel + 2];
        
        std::cout << "Mel " << mel << ": bins [" << left << ", " << center << ", " << right << "]";
        std::cout << " Hz [" << hz_points[mel] << ", " << hz_points[mel+1] << ", " << hz_points[mel+2] << "]";
        
        // Check for issues
        if (left == center || center == right) {
            std::cout << " WARNING: Zero-width filter!";
        }
        if (left < 0 || right >= num_fft_bins) {
            std::cout << " WARNING: Out of bounds!";
        }
        
        // Count non-zero bins
        int nonzero = 0;
        for (int bin = left; bin < right; ++bin) {
            if (bin >= 0 && bin < num_fft_bins) {
                nonzero++;
            }
        }
        std::cout << " Non-zero bins: " << nonzero;
        
        std::cout << std::endl;
    }
    
    // Specifically check mel filter 2
    std::cout << "\nDetailed analysis of mel filter 2:" << std::endl;
    int left = bin_points[2];
    int center = bin_points[3];
    int right = bin_points[4];
    
    std::cout << "Left bin: " << left << " (Hz: " << hz_points[2] << ")" << std::endl;
    std::cout << "Center bin: " << center << " (Hz: " << hz_points[3] << ")" << std::endl;
    std::cout << "Right bin: " << right << " (Hz: " << hz_points[4] << ")" << std::endl;
    
    // Simulate filter values
    std::cout << "\nFilter weights for mel 2:" << std::endl;
    for (int bin = std::max(0, left); bin < std::min(num_fft_bins, right); ++bin) {
        float weight = 0.0f;
        if (bin >= left && bin < center) {
            weight = static_cast<float>(bin - left) / (center - left);
        } else if (bin >= center && bin < right) {
            weight = static_cast<float>(right - bin) / (right - center);
        }
        if (weight > 0) {
            std::cout << "  Bin " << bin << ": weight = " << weight << std::endl;
        }
    }
    
    return 0;
}