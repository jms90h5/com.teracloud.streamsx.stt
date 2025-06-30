# SPL Syntax Examples for Tuple Attribute Access

Based on the analysis of TwoChannelBasicTest.spl and the type definitions, here are the proper SPL syntax patterns:

## 1. Accessing Simple Tuple Attributes

For a tuple of type `ChannelAudioStream`:
```spl
type ChannelAudioStream = tuple<
    blob audioData,
    uint64 timestamp,
    ChannelMetadata channelInfo,
    int32 sampleRate,
    int32 bitsPerSample
>;
```

**Correct syntax:**
```spl
// Direct attribute access
uint64 ts = CallerAudio.timestamp;
int32 rate = CallerAudio.sampleRate;
blob data = CallerAudio.audioData;
```

**Incorrect syntax:**
```spl
// SPL does not use getter methods
uint64 ts = CallerAudio.get_timestamp();  // WRONG!
```

## 2. Accessing Nested Tuple Attributes

For nested tuple `channelInfo` of type `ChannelMetadata`:
```spl
type ChannelMetadata = tuple<
    int32 channelNumber,
    rstring channelRole,
    rstring phoneNumber,
    rstring speakerId,
    map<rstring,rstring> additionalMetadata
>;
```

**Correct syntax:**
```spl
// Nested attribute access using dot notation
int32 chan = CallerAudio.channelInfo.channelNumber;
rstring role = CallerAudio.channelInfo.channelRole;
```

## 3. Casting to rstring for Concatenation

**Correct syntax:**
```spl
// Cast numeric types to rstring for string concatenation
printStringLn("[CALLER] Tuple " + (rstring)tupleCount);
printStringLn("  bytes: " + (rstring)dataSize);
printStringLn("  timestamp: " + (rstring)ts + "ms");
printStringLn("  channel: " + (rstring)chan);
printStringLn("  role: " + role);  // role is already rstring, no cast needed
```

## 4. Reserved Words

"timestamp" is NOT a reserved word in SPL. It can be used as an attribute name without issues. The SPL type system has a built-in `timestamp` type (different from uint64), but this doesn't prevent using it as an attribute name.

## Summary

The fix applied to TwoChannelBasicTest.spl:
- Changed line 39 from `CallerAudio.get_timestamp()` to `CallerAudio.timestamp`
- All other attribute accesses in the file were already using correct syntax
- The nested attribute accesses (e.g., `CallerAudio.channelInfo.channelNumber`) were correct
- The rstring casts for numeric types were properly done