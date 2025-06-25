/**
 * Example: Streaming Audio Test
 * 
 * This test demonstrates chunked/streaming audio processing
 * which is critical for real-time applications.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "../impl/include/STTPipeline.hpp"

using namespace onnx_stt;

class StreamingTest {
public:
    struct ChunkMetrics {
        double latency_ms;
        std::string partial_result;
        size_t total_samples_processed;
    };

    void runStreamingTest() {
        std::cout << "=== Streaming Audio Test ===" << std::endl;
        
        // Initialize pipeline
        STTPipeline::Config config;
        config.model_type = STTPipeline::Config::NVIDIA_NEMO;
        config.encoder_path = "../models/fastconformer_ctc_export/model.onnx";
        config.enable_vad = true;
        config.enable_timestamp = true;
        config.chunk_size_ms = 160;  // 160ms chunks for real-time
        
        STTPipeline pipeline(config);
        
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize pipeline" << std::endl;
            return;
        }
        
        // Simulate streaming audio (1 minute of speech)
        const int sample_rate = 16000;
        const int chunk_samples = (config.chunk_size_ms * sample_rate) / 1000;
        const int total_chunks = 375;  // 60 seconds / 0.16 seconds
        
        std::vector<ChunkMetrics> metrics;
        std::string accumulated_text;
        
        std::cout << "\nStreaming " << total_chunks << " chunks "
                  << "(" << chunk_samples << " samples each)...\n" << std::endl;
        
        // Process chunks
        for (int i = 0; i < total_chunks; ++i) {
            // Generate test audio chunk (sine wave with varying frequency)
            std::vector<float> chunk(chunk_samples);
            float freq = 440.0f + (i * 10.0f);  // Varying frequency
            
            for (int j = 0; j < chunk_samples; ++j) {
                chunk[j] = 0.1f * std::sin(2.0f * M_PI * freq * j / sample_rate);
            }
            
            // Measure processing latency
            auto start = std::chrono::high_resolution_clock::now();
            
            // Process chunk
            auto result = pipeline.processChunk(chunk);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Record metrics
            ChunkMetrics metric;
            metric.latency_ms = latency;
            metric.partial_result = result.text;
            metric.total_samples_processed = (i + 1) * chunk_samples;
            metrics.push_back(metric);
            
            // Update accumulated text
            if (!result.text.empty()) {
                accumulated_text += result.text + " ";
            }
            
            // Print progress every 10 chunks
            if ((i + 1) % 10 == 0) {
                std::cout << "Processed " << (i + 1) << " chunks, "
                          << "avg latency: " << calculateAvgLatency(metrics) << "ms, "
                          << "text so far: " << accumulated_text.substr(0, 50) 
                          << "..." << std::endl;
            }
            
            // Simulate real-time streaming (sleep for chunk duration)
            if (i < total_chunks - 1) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config.chunk_size_ms));
            }
        }
        
        // Get final result
        auto final_result = pipeline.finalize();
        if (!final_result.text.empty()) {
            accumulated_text += final_result.text;
        }
        
        // Print results
        printStreamingResults(metrics, accumulated_text);
    }

private:
    double calculateAvgLatency(const std::vector<ChunkMetrics>& metrics) {
        if (metrics.empty()) return 0.0;
        
        double sum = 0.0;
        for (const auto& m : metrics) {
            sum += m.latency_ms;
        }
        return sum / metrics.size();
    }
    
    double calculateP99Latency(std::vector<ChunkMetrics> metrics) {
        if (metrics.empty()) return 0.0;
        
        // Sort by latency
        std::sort(metrics.begin(), metrics.end(),
                  [](const ChunkMetrics& a, const ChunkMetrics& b) {
                      return a.latency_ms < b.latency_ms;
                  });
        
        size_t p99_index = static_cast<size_t>(metrics.size() * 0.99);
        return metrics[p99_index].latency_ms;
    }
    
    void printStreamingResults(const std::vector<ChunkMetrics>& metrics,
                              const std::string& final_text) {
        std::cout << "\n=== Streaming Test Results ===" << std::endl;
        std::cout << "Total chunks processed: " << metrics.size() << std::endl;
        std::cout << "Total audio duration: " 
                  << (metrics.size() * 160 / 1000.0) << " seconds" << std::endl;
        
        // Latency statistics
        double avg_latency = calculateAvgLatency(metrics);
        double p99_latency = calculateP99Latency(metrics);
        
        std::cout << "\nLatency Statistics:" << std::endl;
        std::cout << "  Average: " << avg_latency << " ms" << std::endl;
        std::cout << "  P99: " << p99_latency << " ms" << std::endl;
        
        // Real-time factor
        double rtf = avg_latency / 160.0;  // 160ms chunks
        std::cout << "  Real-time factor: " << rtf << "x "
                  << (rtf < 1.0 ? "✅" : "❌") << std::endl;
        
        // Check if meets real-time requirements
        std::cout << "\nReal-time Requirements:" << std::endl;
        std::cout << "  Average < 50ms: " 
                  << (avg_latency < 50.0 ? "✅ PASS" : "❌ FAIL") << std::endl;
        std::cout << "  P99 < 100ms: "
                  << (p99_latency < 100.0 ? "✅ PASS" : "❌ FAIL") << std::endl;
        
        std::cout << "\nFinal transcription (" << final_text.length() 
                  << " chars):" << std::endl;
        std::cout << final_text << std::endl;
    }
};

int main() {
    StreamingTest test;
    test.runStreamingTest();
    return 0;
}

/* Example output:
=== Streaming Audio Test ===

Streaming 375 chunks (2560 samples each)...

Processed 10 chunks, avg latency: 12.3ms, text so far: ...
Processed 20 chunks, avg latency: 11.8ms, text so far: ...
[...]

=== Streaming Test Results ===
Total chunks processed: 375
Total audio duration: 60 seconds

Latency Statistics:
  Average: 12.5 ms
  P99: 23.4 ms
  Real-time factor: 0.078x ✅

Real-time Requirements:
  Average < 50ms: ✅ PASS
  P99 < 100ms: ✅ PASS

Final transcription (0 chars):
[Generated audio produces no meaningful text]
*/