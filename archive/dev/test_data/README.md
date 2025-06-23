# Test Data for Speech-to-Text Toolkit

This directory contains test data files that can be used with the speech-to-text toolkit samples and tests.

## Audio Files

The `audio/` directory contains various WAV files for testing:

- `0.wav` - English speech sample
- `1.wav` - English speech sample  
- `2.wav` - English speech sample
- `3.wav` - English speech sample
- `8k.wav` - 8kHz sample (for testing resampling)
- `aishell-BAC009S0724W0121.wav` - Chinese speech sample
- `librispeech-1995-1837-0001.wav` - LibriSpeech English sample

## Usage in Samples

Sample applications can reference these files using relative paths:

```spl
expression<rstring> $audioFile : "../../test_data/audio/0.wav";
```

## Model Data

The models are located in the `models/` directory:

- `models/wenet_model/` - Complete WeNet model with CMVN stats
- `models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/` - ONNX models

All test data and models are self-contained within the toolkit directory structure.