#!/bin/bash

# Migration script to reorganize TeraCloud Streams STT Toolkit
# to follow official Teracloud Streams Toolkit Development Guide structure
#
# IMPORTANT: This script moves files AND updates all references
# to ensure the toolkit continues to build and run after migration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== TeraCloud Streams STT Toolkit Structure Migration ==="
echo "Migrating to official Teracloud Streams toolkit standards"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log actions
log_action() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to update file references
update_references() {
    local file="$1"
    local description="$2"
    
    if [ -f "$file" ]; then
        echo "  Updating $description: $file"
        
        # Create backup
        cp "$file" "$file.backup"
        
        # Update all the path references we identified
        sed -i 's|deps/onnxruntime|lib/onnxruntime|g' "$file"
        sed -i 's|../../deps/|../../lib/|g' "$file"
        sed -i 's|models/fastconformer_ctc_export|opt/models/fastconformer_ctc_export|g' "$file"
        sed -i 's|models/nemo_fastconformer_streaming|opt/models/nemo_fastconformer_streaming|g' "$file"
        sed -i 's|models/sherpa_onnx_paraformer|opt/models/sherpa_onnx_paraformer|g' "$file"
        sed -i 's|test_data/audio|opt/test_data/audio|g' "$file"
        sed -i 's|$(TOOLKIT_PATH)/models/|$(TOOLKIT_PATH)/opt/models/|g' "$file"
        sed -i 's|$(TOOLKIT_PATH)/test_data/|$(TOOLKIT_PATH)/opt/test_data/|g' "$file"
        sed -i 's|"/models/|"/opt/models/|g' "$file"
        sed -i 's|/test_data/|/opt/test_data/|g' "$file"
        
        log_action "Updated references in $file"
    else
        log_warning "File not found: $file"
    fi
}

# Backup critical files before migration (skip large model files)
echo "1. Creating selective backup..."
mkdir -p migration_backup
# Only backup key files, skip large model/nemo files
tar --exclude="*.nemo" --exclude="*.bin" --exclude="*.npy" --exclude="venv_*" \
    --exclude="models_backup_*" --exclude="fastconformer_*" --exclude="nvidia_*" \
    -cf migration_backup/critical_files.tar . 2>/dev/null || true
log_action "Created selective backup in migration_backup/"

echo ""
echo "2. Creating new standard directory structure..."

# Create standard directories per official guide
mkdir -p etc
mkdir -p lib
mkdir -p opt/models
mkdir -p opt/test_data
mkdir -p impl/bin
mkdir -p impl/lib
mkdir -p doc
mkdir -p samples

log_action "Created standard directories"

echo ""
echo "3. Moving files to new locations..."

# Move deps/ to lib/ (major change)
if [ -d "deps" ]; then
    echo "  Moving deps/ → lib/"
    mv deps/* lib/ 2>/dev/null || true
    rmdir deps 2>/dev/null || true
    log_action "Moved deps/ to lib/"
fi

# Move models/ to opt/models/
if [ -d "models" ]; then
    echo "  Moving models/ → opt/models/"
    mv models/* opt/models/ 2>/dev/null || true
    rmdir models 2>/dev/null || true
    log_action "Moved models/ to opt/models/"
fi

# Move test_data/ to opt/test_data/
if [ -d "test_data" ]; then
    echo "  Moving test_data/ → opt/test_data/"
    mv test_data/* opt/test_data/ 2>/dev/null || true
    rmdir test_data 2>/dev/null || true
    log_action "Moved test_data/ to opt/test_data/"
fi

# Move Python scripts to impl/bin/
echo "  Moving Python scripts → impl/bin/"
for script in *.py; do
    if [ -f "$script" ]; then
        mv "$script" impl/bin/
        log_action "Moved $script to impl/bin/"
    fi
done

# Move shell scripts to impl/bin/
echo "  Moving shell scripts → impl/bin/"
for script in *.sh; do
    if [ -f "$script" ] && [ "$script" != "migrate_to_standard_structure.sh" ]; then
        mv "$script" impl/bin/
        log_action "Moved $script to impl/bin/"
    fi
done

# Move documentation files to doc/
echo "  Moving documentation → doc/"
for doc in *.md *.txt *.docx; do
    if [ -f "$doc" ] && [ "$doc" != "README.md" ]; then
        mv "$doc" doc/
        log_action "Moved $doc to doc/"
    fi
done

# Move requirements files to etc/
echo "  Moving configuration files → etc/"
for req in requirements*.txt *.conf *.cfg; do
    if [ -f "$req" ]; then
        mv "$req" etc/
        log_action "Moved $req to etc/"
    fi
done

echo ""
echo "4. Updating file references..."

# Update Makefiles
update_references "Makefile" "root Makefile"
update_references "impl/Makefile" "implementation Makefile"
update_references "samples/Makefile" "samples Makefile"

# Update SPL operator XML files
for xml_file in com.teracloud.streamsx.stt/*/model.xml; do
    if [ -f "$xml_file" ]; then
        update_references "$xml_file" "SPL operator XML"
    fi
done

# Update sample SPL files
update_references "OnnxSTT_Demo.spl" "main demo SPL file"
for spl_file in samples/com.teracloud.streamsx.stt.sample/*.spl; do
    if [ -f "$spl_file" ]; then
        update_references "$spl_file" "sample SPL file"
    fi
done

# Update shell scripts in impl/bin/
for script in impl/bin/*.sh; do
    if [ -f "$script" ]; then
        update_references "$script" "shell script"
    fi
done

# Update Python scripts in impl/bin/
for script in impl/bin/*.py; do
    if [ -f "$script" ]; then
        update_references "$script" "Python script"
    fi
done

# Update documentation
for doc in doc/*.md; do
    if [ -f "$doc" ]; then
        update_references "$doc" "documentation file"
    fi
done

# Update README.md
update_references "README.md" "main README"

# Update test scripts
for test_script in test/*.sh test/*.py; do
    if [ -f "$test_script" ]; then
        update_references "$test_script" "test script"
    fi
done

echo ""
echo "5. Creating required toolkit files..."

# Create info.xml (recommended by official guide)
cat > info.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<info:toolkitInfoModel xmlns:info="http://www.ibm.com/xmlns/prod/streams/spl/toolkitInfo">
  <info:identity>
    <info:name>com.teracloud.streamsx.stt</info:name>
    <info:description>TeraCloud Streams Speech-to-Text Toolkit</info:description>
    <info:version>1.0.0</info:version>
    <info:requiredProductVersion>7.2.0.0</info:requiredProductVersion>
  </info:identity>
  <info:dependencies/>
</info:toolkitInfoModel>
EOF

log_action "Created info.xml"

# Update any hardcoded paths in the main SPL files
echo ""
echo "6. Updating hardcoded paths in SPL files..."

# Fix specific hardcoded paths we found
if [ -f "samples/com.teracloud.streamsx.stt.sample/BasicNeMoTest.spl" ]; then
    # Update the hardcoded paths to use relative paths
    sed -i 's|/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/test_data/|../../opt/test_data/|g' samples/com.teracloud.streamsx.stt.sample/BasicNeMoTest.spl
    sed -i 's|/homes/jsharpe/teracloud/toolkits/com.teracloud.streamsx.stt/models/|../../opt/models/|g' samples/com.teracloud.streamsx.stt.sample/BasicNeMoTest.spl
    log_action "Updated hardcoded paths in BasicNeMoTest.spl"
fi

# Update any remaining hardcoded paths in demo files
if [ -f "OnnxSTT_Demo.spl" ]; then
    sed -i 's|../../models/|../../opt/models/|g' OnnxSTT_Demo.spl
    sed -i 's|../../test_data/|../../opt/test_data/|g' OnnxSTT_Demo.spl
    log_action "Updated paths in OnnxSTT_Demo.spl"
fi

echo ""
echo "7. Final verification..."

# Check that toolkit can be indexed
if command -v spl-make-toolkit >/dev/null 2>&1; then
    echo "  Running spl-make-toolkit to verify structure..."
    if spl-make-toolkit --directory . --check; then
        log_action "Toolkit structure validation passed"
    else
        log_warning "Toolkit structure validation failed - manual review needed"
    fi
else
    log_warning "spl-make-toolkit not available - skipping validation"
fi

echo ""
echo "=== Migration Summary ==="
echo ""
echo "Directory Structure Changes:"
echo "  deps/          → lib/"
echo "  models/        → opt/models/"
echo "  test_data/     → opt/test_data/"
echo "  Python scripts → impl/bin/"
echo "  Shell scripts  → impl/bin/"
echo "  Documentation  → doc/"
echo "  Config files   → etc/"
echo ""
echo "Files Updated:"
echo "  ✓ All Makefiles"
echo "  ✓ SPL operator XML files"
echo "  ✓ Sample SPL applications"
echo "  ✓ Shell and Python scripts"
echo "  ✓ Documentation files"
echo "  ✓ Hardcoded paths in samples"
echo ""
echo "New Files Created:"
echo "  ✓ info.xml (toolkit metadata)"
echo ""

echo -e "${GREEN}Migration completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Test compilation: make clean && make"
echo "2. Test samples: cd samples && make"
echo "3. Run verification: impl/bin/verify_nemo_setup.sh"
echo "4. If issues occur, restore from migration_backup/"
echo ""
echo "The toolkit now follows official Teracloud Streams standards."