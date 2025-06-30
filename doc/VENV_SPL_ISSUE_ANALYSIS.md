# Python Virtual Environment SPL Compilation Issue - Deep Dive

## Root Cause

The SPL compiler (sc) has an issue with files containing spaces in their names within the toolkit directory. When building an SPL application, the compiler scans the entire toolkit directory tree and generates makefiles. Files with spaces in their names break the makefile syntax.

### Problematic Files in Python venv

1. **Main culprit**: `impl/venv/lib/python3.11/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt`
2. Other files with spaces:
   - `scipy/io/tests/data/Transparent Busy.ani`
   - `setuptools/command/launcher manifest.xml`
   - `setuptools/script (dev).tmpl`

## Why This Happens

The SPL compiler generates makefiles where these files become targets or dependencies. The space in "Lorem ipsum.txt" causes the makefile to interpret it as two separate files: "Lorem" and "ipsum.txt", leading to:

```
make: *** No rule to make target '.../Lorem', needed by 'SimpleTest.sab'
```

## Alternative Solutions

### 1. Delete Problematic Files (Permanent Fix)

Remove files with spaces from the venv after installation:

```bash
# After setting up venv and installing packages
find impl/venv -name "* *" -type f -delete

# Or just remove the main culprit:
rm -f "impl/venv/lib/python3.11/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt"
```

**Pros**: Simple, permanent fix
**Cons**: May need to repeat after pip upgrades

### 2. Rename Problematic Files

Replace spaces with underscores:

```bash
# Rename all files with spaces
find impl/venv -name "* *" -type f | while IFS= read -r file; do
    newfile="${file// /_}"
    mv "$file" "$newfile"
done
```

**Pros**: Preserves files if needed
**Cons**: May break some Python packages (unlikely for these test files)

### 3. Use Symbolic Link Outside Toolkit

Create venv outside the toolkit and symlink:

```bash
# Create venv outside toolkit
python3.11 -m venv /tmp/stt_venv
ln -s /tmp/stt_venv impl/venv

# Add to .toolkit-ignore
echo "impl/venv" >> .toolkit-ignore
```

**Pros**: Venv completely outside SPL's reach
**Cons**: Less portable, symlink might not work on all systems

### 4. Use .spl-make-toolkit Exclude File

Create a file to exclude paths from toolkit indexing:

```bash
# Create exclusion file
cat > .spl-make-toolkit << 'EOF'
--exclude impl/venv
--exclude "*.pyc"
--exclude __pycache__
EOF
```

**Note**: This SHOULD work but depends on SPL version supporting these flags

### 5. Install Python Packages Globally

Use system Python or user-installed packages:

```bash
# Install for user only (no venv)
pip install --user torch onnx nemo_toolkit[asr]

# Or use system Python
sudo pip install torch onnx nemo_toolkit[asr]
```

**Pros**: No venv in toolkit
**Cons**: Less isolated, potential version conflicts

### 6. Build SPL Applications Outside Toolkit

Copy only needed files for SPL compilation:

```bash
# Create clean build directory
mkdir -p /tmp/stt_build
cp -r samples /tmp/stt_build/
cp -r com.teracloud.streamsx.stt /tmp/stt_build/
cp toolkit.xml /tmp/stt_build/

# Build there
cd /tmp/stt_build/samples
sc -a -t .. -M SimpleTest SimpleTest.spl
```

**Pros**: Clean build environment
**Cons**: More complex workflow

### 7. Use Docker/Container for SPL Builds

Create a clean environment for SPL compilation:

```dockerfile
FROM teracloud/streams:7.2.0.1
COPY --exclude=impl/venv . /toolkit
WORKDIR /toolkit/samples
RUN make
```

**Pros**: Completely isolated
**Cons**: Requires Docker setup

## Recommended Solution

For immediate use, the best approach is **Solution 1** - simply delete the problematic files:

```bash
# After installing Python packages
find impl/venv -name "* *" -type f -delete

# Then compile SPL normally
cd samples && make
```

For long-term, consider:
- Filing a bug report with Teracloud about SPL compiler handling of spaces in filenames
- Using a build script that automatically removes these files
- Keeping venv outside the toolkit directory

## Verification Script

Add this to your setup process:

```bash
#!/bin/bash
# check_venv_compatibility.sh

echo "Checking for SPL-incompatible files in venv..."
PROBLEM_FILES=$(find impl/venv -name "* *" -type f 2>/dev/null)

if [ -n "$PROBLEM_FILES" ]; then
    echo "Found files with spaces that will break SPL compilation:"
    echo "$PROBLEM_FILES"
    echo ""
    echo "Options:"
    echo "1. Delete them: find impl/venv -name '* *' -type f -delete"
    echo "2. Move venv: mv impl/venv /tmp/backup"
    echo "3. See doc/VENV_SPL_ISSUE_ANALYSIS.md for more solutions"
else
    echo "✓ No problematic files found"
fi
```

## Root Cause Fix

The real fix would be in the SPL compiler to properly escape filenames with spaces in generated makefiles. Until then, these workarounds are necessary.