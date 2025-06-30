#!/bin/bash
# Clean Python venv to be SPL-compatible
# Removes files with spaces that break SPL compilation

echo "Cleaning Python venv for SPL compatibility..."

if [ ! -d "venv" ]; then
    echo "No venv directory found in current directory"
    exit 1
fi

# Count problematic files
PROBLEM_COUNT=$(find venv -name "* *" -type f 2>/dev/null | wc -l)

if [ "$PROBLEM_COUNT" -eq 0 ]; then
    echo "✓ No problematic files found. Venv is SPL-compatible."
    exit 0
fi

echo "Found $PROBLEM_COUNT files with spaces that will break SPL compilation:"
find venv -name "* *" -type f 2>/dev/null

echo ""
read -p "Delete these files? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    find venv -name "* *" -type f -delete
    echo "✓ Deleted $PROBLEM_COUNT problematic files"
    echo "✓ Venv is now SPL-compatible"
else
    echo "Cancelled. SPL compilation will fail with these files present."
    echo "Alternative: Move venv before compiling: mv venv /tmp/backup"
    exit 1
fi