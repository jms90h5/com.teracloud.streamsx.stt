#ifndef STREAMING_BUFFER_HPP
#define STREAMING_BUFFER_HPP

#include <vector>
#include <cstddef>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace onnx_stt {

/**
 * @brief Circular buffer for streaming audio processing
 * 
 * This class manages audio buffering for streaming speech recognition,
 * handling chunk extraction with configurable overlap.
 */
class StreamingBuffer {
public:
    /**
     * @brief Constructor
     * @param capacity Maximum buffer size in samples
     * @param chunk_size Size of each chunk to extract
     * @param overlap_size Number of samples to overlap between chunks
     */
    StreamingBuffer(size_t capacity, size_t chunk_size, size_t overlap_size)
        : capacity_(capacity)
        , chunk_size_(chunk_size)
        , overlap_size_(overlap_size)
        , buffer_(capacity)
        , write_pos_(0)
        , available_samples_(0) {
        
        if (overlap_size >= chunk_size) {
            throw std::invalid_argument("Overlap size must be less than chunk size");
        }
    }
    
    /**
     * @brief Append audio samples to the buffer
     * @param data Pointer to audio samples
     * @param size Number of samples to append
     * @return Number of samples actually written (may be less if buffer is full)
     */
    size_t append(const float* data, size_t size) {
        size_t samples_to_write = std::min(size, capacity_ - available_samples_);
        
        for (size_t i = 0; i < samples_to_write; ++i) {
            buffer_[write_pos_] = data[i];
            write_pos_ = (write_pos_ + 1) % capacity_;
        }
        
        available_samples_ += samples_to_write;
        return samples_to_write;
    }
    
    /**
     * @brief Get the next chunk of audio data
     * @param chunk Output vector to fill with chunk data
     * @return true if a full chunk was extracted, false if not enough data
     */
    bool getNextChunk(std::vector<float>& chunk) {
        if (available_samples_ < chunk_size_) {
            return false;
        }
        
        chunk.resize(chunk_size_);
        
        // Calculate read position
        size_t read_pos = (write_pos_ + capacity_ - available_samples_) % capacity_;
        
        // Copy chunk data
        for (size_t i = 0; i < chunk_size_; ++i) {
            chunk[i] = buffer_[(read_pos + i) % capacity_];
        }
        
        // Advance position, keeping overlap
        size_t advance = chunk_size_ - overlap_size_;
        available_samples_ -= advance;
        
        return true;
    }
    
    /**
     * @brief Get remaining audio data (less than chunk_size)
     * @param remainder Output vector for remaining samples
     * @return Number of samples in remainder
     */
    size_t getRemainder(std::vector<float>& remainder) {
        if (available_samples_ == 0) {
            remainder.clear();
            return 0;
        }
        
        remainder.resize(available_samples_);
        size_t read_pos = (write_pos_ + capacity_ - available_samples_) % capacity_;
        
        for (size_t i = 0; i < available_samples_; ++i) {
            remainder[i] = buffer_[(read_pos + i) % capacity_];
        }
        
        available_samples_ = 0;
        return remainder.size();
    }
    
    /**
     * @brief Clear the buffer
     */
    void clear() {
        available_samples_ = 0;
        write_pos_ = 0;
    }
    
    /**
     * @brief Get number of available samples
     */
    size_t available() const {
        return available_samples_;
    }
    
    /**
     * @brief Check if enough samples for a chunk
     */
    bool hasChunk() const {
        return available_samples_ >= chunk_size_;
    }
    
private:
    size_t capacity_;
    size_t chunk_size_;
    size_t overlap_size_;
    std::vector<float> buffer_;
    size_t write_pos_;
    size_t available_samples_;
};

} // namespace onnx_stt

#endif // STREAMING_BUFFER_HPP