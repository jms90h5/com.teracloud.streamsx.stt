#!/bin/bash

# Post-migration cleanup script
# Run this after verifying the migration was successful

echo "=== Post-Migration Cleanup ==="
echo "This will remove large backup directories created during migration."
echo ""

# Check current directory
if [ ! -f "info.xml" ] || [ ! -d "impl" ]; then
    echo "❌ Please run from toolkit root directory"
    exit 1
fi

# Show what will be cleaned
echo "Directories to remove:"
echo "----------------------"
if [ -d "migration_backup" ]; then
    SIZE=$(du -sh migration_backup | cut -f1)
    echo "• migration_backup/ ($SIZE) - Full backup from migration"
fi

if [ -d "archive/debug_data/venv_nemo" ]; then
    SIZE=$(du -sh archive/debug_data/venv_nemo | cut -f1)
    echo "• archive/debug_data/venv_nemo/ ($SIZE) - Old Python 3.9 venv"
fi

echo ""
read -p "Remove these directories? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing backup directories..."
    
    # Remove migration backup
    if [ -d "migration_backup" ]; then
        rm -rf migration_backup
        echo "✓ Removed migration_backup/"
    fi
    
    # Remove old Python 3.9 venv
    if [ -d "archive/debug_data/venv_nemo" ]; then
        rm -rf archive/debug_data/venv_nemo
        echo "✓ Removed old Python 3.9 venv"
    fi
    
    # Calculate space saved
    echo ""
    echo "✅ Cleanup complete!"
    echo "   Space reclaimed: ~23GB"
else
    echo "Cleanup cancelled."
fi

echo ""
echo "Note: archive/debug_data/ still contains development artifacts (5GB)"
echo "      These can be removed if no longer needed for reference."