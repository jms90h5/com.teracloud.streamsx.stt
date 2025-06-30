# Migration Guide: Moving to External Python Environment

Due to SPL compiler limitations with Python virtual environments, we recommend using an external venv setup.

## Why Migrate?

The SPL compiler has a bug where it tries to include ALL files in the toolkit directory when building .sab files, including Python packages. Files with spaces (like "Lorem ipsum.txt" in setuptools) cause build failures.

## Migration Steps

### 1. Backup Current Setup (Optional)

```bash
# If you have a working model already exported
cp -r opt/models /tmp/models_backup
```

### 2. Clean Current Environment

```bash
# Remove internal venv
rm -rf impl/venv
rm -rf python_env

# Remove any temporary backups
rm -rf /tmp/stt_venv_backup
```

### 3. Run New Setup

```bash
# From toolkit root
./setup_external_venv.sh
```

This creates Python environment at: `~/.stt_toolkit_env/`

### 4. Restore Models (If Backed Up)

```bash
# If you backed up models
cp -r /tmp/models_backup/* opt/models/
```

### 5. Update Your Workflow

**Old way:**
```bash
source impl/venv/bin/activate
```

**New way:**
```bash
source .envrc
# OR
source ~/.stt_toolkit_env/bin/activate
```

## Benefits

1. **No SPL compilation issues** - Python files are completely outside toolkit
2. **Shared environment** - Multiple toolkit clones can share same Python env
3. **Easier updates** - Update Python packages without touching toolkit
4. **Cleaner toolkit** - Smaller repository size, faster operations

## Using the New Setup

All commands work the same:

```bash
# Activate environment
source .envrc

# Export model
./export_model.sh

# Build C++ 
stt-build

# Run tests
stt-test

# Build SPL samples - NO WORKAROUNDS NEEDED!
cd samples && make
```

## Rollback (If Needed)

To go back to internal venv:

```bash
# Use original setup
./setup.sh

# Will need workarounds for SPL compilation
mv impl/venv /tmp/backup  # Before building SPL
```

## FAQ

**Q: Where is the Python environment now?**
A: `~/.stt_toolkit_env/` in your home directory

**Q: Can I share it between multiple toolkits?**
A: Yes! All toolkit instances can use the same environment

**Q: What if I need different Python package versions?**
A: Create multiple environments:
```bash
python3.11 -m venv ~/.stt_toolkit_env_v2
# Edit activate_python.sh to point to new location
```

**Q: Will this break existing applications?**
A: No, the C++ implementation and SPL operators are unchanged

## For CI/CD

Update your build scripts:

```bash
# Old
cd toolkit && source impl/venv/bin/activate

# New  
cd toolkit && source ~/.stt_toolkit_env/bin/activate
# OR
cd toolkit && ./setup_external_venv.sh && source .envrc
```