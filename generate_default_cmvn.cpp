/**
 * Generate default CMVN stats for the model
 * Since the model has normalize: NA, we might not need CMVN at all
 * But let's create neutral stats just in case
 */

#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // For 80-dimensional features
    const int feature_dim = 80;
    
    // Neutral CMVN stats (no normalization)
    std::vector<float> mean(feature_dim, 0.0f);  // Zero mean
    std::vector<float> var(feature_dim, 1.0f);   // Unit variance
    
    // Save to binary file
    std::ofstream cmvn_file("neutral_cmvn_stats.bin", std::ios::binary);
    
    // Write dimensions
    cmvn_file.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));
    
    // Write mean
    cmvn_file.write(reinterpret_cast<const char*>(mean.data()), 
                    feature_dim * sizeof(float));
    
    // Write variance
    cmvn_file.write(reinterpret_cast<const char*>(var.data()), 
                    feature_dim * sizeof(float));
    
    cmvn_file.close();
    
    std::cout << "Generated neutral CMVN stats:" << std::endl;
    std::cout << "  Feature dimension: " << feature_dim << std::endl;
    std::cout << "  Mean: all zeros (no mean subtraction)" << std::endl;
    std::cout << "  Variance: all ones (no variance normalization)" << std::endl;
    std::cout << "  Saved to: neutral_cmvn_stats.bin" << std::endl;
    
    return 0;
}