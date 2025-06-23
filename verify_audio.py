#!/usr/bin/env python3
"""
Verify the audio file and create a short test sample
"""
import wave
import numpy as np

def check_audio_file(filename):
    """Check audio file properties and content"""
    with wave.open(filename, 'rb') as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        duration = n_frames / framerate
        
        print(f"File: {filename}")
        print(f"Channels: {n_channels}")
        print(f"Sample rate: {framerate} Hz")
        print(f"Bit depth: {sampwidth * 8} bits")
        print(f"Frames: {n_frames}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Read first few seconds
        frames_to_read = min(n_frames, framerate * 3)  # First 3 seconds
        audio_data = wav.readframes(frames_to_read)
        
        if sampwidth == 2:
            samples = np.frombuffer(audio_data, dtype=np.int16)
        else:
            samples = np.frombuffer(audio_data, dtype=np.int8)
            
        # Check if there's silence at the beginning
        first_second = samples[:framerate]
        energy = np.sum(first_second ** 2) / len(first_second)
        print(f"\nFirst second energy: {energy}")
        
        return n_frames, framerate, sampwidth, n_channels

def create_short_test_audio(input_file, output_file, duration_seconds=3.0):
    """Create a short test audio file from the first few seconds"""
    with wave.open(input_file, 'rb') as wav_in:
        params = wav_in.getparams()
        framerate = params.framerate
        frames_to_copy = int(framerate * duration_seconds)
        
        # Read audio data
        audio_data = wav_in.readframes(frames_to_copy)
        
        # Write to new file
        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(audio_data)
        
        print(f"\nCreated {output_file} with {duration_seconds} seconds of audio")

# Check the LibriSpeech file
print("=== Checking LibriSpeech test file ===")
check_audio_file("test_data/audio/librispeech-1995-1837-0001.wav")

# Create a short version for testing
create_short_test_audio(
    "test_data/audio/librispeech-1995-1837-0001.wav",
    "test_data/audio/librispeech_3sec.wav",
    3.0
)

# Check other available test files
print("\n=== Other test files ===")
import os
test_dir = "test_data/audio"
for file in sorted(os.listdir(test_dir)):
    if file.endswith('.wav') and file != 'librispeech_3sec.wav':
        print(f"\n{file}:")
        try:
            n_frames, rate, _, _ = check_audio_file(os.path.join(test_dir, file))
            duration = n_frames / rate
            if duration < 5:  # Short files
                print("  -> Good for testing (short duration)")
        except Exception as e:
            print(f"  Error: {e}")