# SPL Compiler Bug: Filenames with Spaces Break Makefile Generation

## Bug Summary

The SPL compiler (`sc`) fails to properly escape filenames containing spaces when generating makefiles, causing build failures.

## Symptoms

Build fails with error:
```
make: *** No rule to make target '/path/to/Lorem', needed by 'MyApp.sab'.  Stop.
```

When the actual filename is: `/path/to/Lorem ipsum.txt`

## Root Cause

When `sc` generates makefiles for building the .sab bundle, it creates dependency rules without properly quoting or escaping filenames that contain spaces. The makefile interprets a file named "Lorem ipsum.txt" as two separate targets: "Lorem" and "ipsum.txt".

## Impact

- Any toolkit containing files with spaces in their names cannot be compiled
- Affects integration with third-party tools that create such files
- Forces developers to sanitize entire directory trees before compilation

## Minimal Reproduction

1. Create a simple toolkit structure:
```bash
mkdir -p test.toolkit/com.test
cd test.toolkit
```

2. Create a minimal toolkit:
```xml
# info.xml
<toolkitInfoModel>
  <identity>
    <name>test</name>
    <version>1.0.0</version>
  </identity>
</toolkitInfoModel>
```

3. Add ANY file with a space in the name:
```bash
echo "test" > "test file.txt"
```

4. Create simple SPL application:
```spl
// test.spl
composite TestApp {
  graph
    () as Sink = Custom() {
      logic onProcess: { printStringLn("test"); }
    }
}
```

5. Attempt to compile:
```bash
sc -a -t . -M TestApp test.spl
```

**Result**: Build fails with "No rule to make target '/path/to/test'"

## Technical Details

### Generated Makefile Issue

The compiler generates makefile rules similar to:
```makefile
MyApp.sab: file1.txt file2.xml test file.txt other.spl
	commands...
```

Make interprets this as depending on:
- `file1.txt`
- `file2.xml` 
- `test` (ERROR: file not found)
- `file.txt` (ERROR: file not found)
- `other.spl`

### Expected Behavior

Filenames with spaces should be escaped or quoted:
```makefile
MyApp.sab: file1.txt file2.xml "test file.txt" other.spl
	commands...
```

Or using escape sequences:
```makefile
MyApp.sab: file1.txt file2.xml test\ file.txt other.spl
	commands...
```

## Severity: High

This is a fundamental issue that:
- Breaks basic makefile syntax rules
- Has no workaround within the compiler
- Forces manual intervention on toolkit directories
- Affects any toolkit with third-party components

## Proposed Fix

### Option 1: Quote All Filenames (Recommended)
```makefile
MyApp.sab: "file1.txt" "file2.xml" "test file.txt" "other.spl"
```

### Option 2: Escape Spaces
```makefile
MyApp.sab: file1.txt file2.xml test\ file.txt other.spl
```

### Option 3: Use Make Functions
```makefile
DEPS := file1.txt file2.xml test file.txt other.spl
MyApp.sab: $(foreach dep,$(DEPS),"$(dep)")
```

## Test Cases

Compiler should handle:
1. `"test file.txt"` - Simple space
2. `"multiple  spaces.txt"` - Multiple spaces
3. `"tab	file.txt"` - Tab character
4. `"special chars (test).txt"` - Parentheses
5. `"quote's.txt"` - Apostrophes
6. `"newline\nfile.txt"` - Newline (should error gracefully)

## Workaround

Until fixed, users must remove files with spaces:
```bash
# Find files with spaces
find toolkit_dir -name "* *" -type f

# Remove them
find toolkit_dir -name "* *" -type f -delete

# Or rename them
find toolkit_dir -name "* *" -type f | while IFS= read -r file; do
    mv "$file" "${file// /_}"
done
```

## Additional Notes

1. This issue is independent of file type or purpose
2. The `.toolkit-ignore` file doesn't help because the issue occurs during makefile generation, not toolkit indexing
3. This is a standard makefile escaping issue that most build tools handle correctly

## References

- GNU Make Manual: Section 4.1 "Rule Syntax" - requires proper escaping
- POSIX Make Standard: Spaces must be escaped in target names
- Common practice: Most build tools (CMake, Autotools, etc.) handle this automatically