#!/bin/bash

cd /homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt

# Extract just the transcribed words from the output
echo "Extracting transcription from IBM Culture audio test..."
echo ""

# Look for all unique tokens that were decoded
grep -o "\[NeMo CTC: [0-9 ]*\]" ibm_culture_transcript.txt | \
    sed 's/\[NeMo CTC: //g' | \
    sed 's/\]//g' | \
    tr ' ' '\n' | \
    sort -n | \
    uniq -c | \
    sort -rn | \
    head -20

echo ""
echo "Token frequency (count token_id):"
echo "Token 95 appears most frequently"
echo ""

# Now let's see what words these tokens map to
echo "Mapping to vocabulary:"
python3 -c "
vocab = {}
with open('models/nemo_real/vocab.txt', 'r') as f:
    for idx, line in enumerate(f):
        vocab[idx] = line.strip()

# Show what token 95 is
print(f'Token 95 = \"{vocab.get(95, \"UNKNOWN\")}\"')
print()

# Show first 20 vocab entries
print('First 20 vocabulary entries:')
for i in range(20):
    print(f'  {i}: {vocab.get(i, \"UNKNOWN\")}')
"