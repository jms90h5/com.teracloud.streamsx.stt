#!/usr/bin/env python3
"""
Inspect the NeMo export to understand what it expects
"""
import sys
import os
import tarfile
import zipfile
import torch

def inspect_nemo_model(nemo_path):
    """Inspect NeMo model structure"""
    print(f"=== Inspecting NeMo Model: {nemo_path} ===")
    
    # Check if it's the extracted directory
    extracted_dir = "extracted_fastconformer"
    if os.path.exists(extracted_dir):
        print(f"\nFound extracted directory: {extracted_dir}")
        
        # List contents
        for root, dirs, files in os.walk(extracted_dir):
            level = root.replace(extracted_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Limit to first 10 files
                print(f"{subindent}{file}")
        
        # Check for model config
        config_path = os.path.join(extracted_dir, "model_config.yaml")
        if os.path.exists(config_path):
            print(f"\n=== Model Config (first 50 lines) ===")
            with open(config_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    print(line.rstrip())
    
    # Check the original .nemo file if it exists
    if os.path.exists(nemo_path):
        print(f"\n=== Checking original .nemo file ===")
        
        # .nemo files are tar archives
        try:
            with tarfile.open(nemo_path, 'r') as tar:
                print("Contents of .nemo file:")
                for member in tar.getmembers()[:20]:  # First 20 files
                    print(f"  {member.name}")
                
                # Try to extract and read model_config.yaml
                try:
                    config_member = tar.getmember("model_config.yaml")
                    config_file = tar.extractfile(config_member)
                    if config_file:
                        print("\n=== Preprocessor Config ===")
                        content = config_file.read().decode('utf-8')
                        # Look for preprocessor section
                        lines = content.split('\n')
                        in_preprocessor = False
                        for line in lines:
                            if 'preprocessor:' in line:
                                in_preprocessor = True
                            if in_preprocessor and line.strip() and not line[0].isspace() and 'preprocessor:' not in line:
                                break
                            if in_preprocessor:
                                print(line)
                except:
                    pass
        except:
            print(f"Could not open {nemo_path} as tar file")
    
    # Check what the export script created
    print("\n=== Checking exported model inputs ===")
    export_dir = "fastconformer_export_full"
    if os.path.exists(export_dir):
        config_json = os.path.join(export_dir, "config.json")
        if os.path.exists(config_json):
            import json
            with open(config_json, 'r') as f:
                config = json.load(f)
            print("Exported config preprocessor section:")
            print(json.dumps(config.get('preprocessor', {}), indent=2))

if __name__ == "__main__":
    nemo_path = sys.argv[1] if len(sys.argv) > 1 else "fastconformer_full.nemo"
    inspect_nemo_model(nemo_path)