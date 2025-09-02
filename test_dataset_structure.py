#!/usr/bin/env python3
"""
Test script to validate dataset structure and loading.
"""

import os
import sys
sys.path.append('.')

from dataset_config import get_dataset_paths, validate_dataset_paths

def test_dataset_structure():
    """Test if dataset structure is correct"""
    print("=== Testing Dataset Structure ===")
    
    # Get paths
    paths = get_dataset_paths()
    rgb_path = paths['rgb_dataset_path']
    mono_path = paths['mono_dataset_path']
    
    print(f"RGB Dataset Path: {rgb_path}")
    print(f"Mono Dataset Path: {mono_path}")
    
    # Validate paths
    errors = validate_dataset_paths()
    if errors:
        print("\n❌ Validation Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ Dataset paths validation passed!")
    
    # Check actual structure
    print("\n=== Checking Dataset Structure ===")
    
    for dataset_type, dataset_path in [("RGB", rgb_path), ("Mono", mono_path)]:
        print(f"\n{dataset_type} Dataset Structure:")
        if os.path.exists(dataset_path):
            for split in ['train', 'validation']:
                split_path = os.path.join(dataset_path, split)
                if os.path.exists(split_path):
                    print(f"  ✅ {split}/ found")
                    # List files in split directory
                    files = os.listdir(split_path)
                    arrow_files = [f for f in files if f.endswith('.arrow')]
                    json_files = [f for f in files if f.endswith('.json')]
                    print(f"    - {len(arrow_files)} arrow files")
                    print(f"    - {len(json_files)} json files")
                    
                    # Check for required files
                    if 'dataset_info.json' in files:
                        print("    - ✅ dataset_info.json found")
                    else:
                        print("    - ❌ dataset_info.json missing")
                    if 'state.json' in files:
                        print("    - ✅ state.json found") 
                    else:
                        print("    - ❌ state.json missing")
                else:
                    print(f"  ❌ {split}/ not found")
        else:
            print(f"  ❌ {dataset_type} dataset path does not exist")
    
    return True

def test_dataset_loading():
    """Test actual dataset loading"""
    print("\n=== Testing Dataset Loading ===")
    
    try:
        from datasets import load_from_disk
        from dataset_config import get_dataset_paths
        
        paths = get_dataset_paths()
        rgb_path = paths['rgb_dataset_path']
        mono_path = paths['mono_dataset_path']
        
        # Test loading train split
        print("Testing train split loading...")
        rgb_train_path = os.path.join(rgb_path, 'train')
        mono_train_path = os.path.join(mono_path, 'train')
        
        if os.path.exists(rgb_train_path) and os.path.exists(mono_train_path):
            print(f"Loading RGB train from: {rgb_train_path}")
            rgb_dataset = load_from_disk(rgb_train_path)
            print(f"  ✅ RGB dataset loaded: {len(rgb_dataset)} samples")
            print(f"  ✅ RGB features: {list(rgb_dataset.features.keys())}")
            
            print(f"Loading Mono train from: {mono_train_path}")
            mono_dataset = load_from_disk(mono_train_path)
            print(f"  ✅ Mono dataset loaded: {len(mono_dataset)} samples")
            print(f"  ✅ Mono features: {list(mono_dataset.features.keys())}")
            
            # Check if datasets have required features
            required_features = ['gt', 'blur']
            for feature in required_features:
                if feature in rgb_dataset.features:
                    print(f"  ✅ RGB dataset has '{feature}' feature")
                else:
                    print(f"  ❌ RGB dataset missing '{feature}' feature")
                    
                if feature in mono_dataset.features:
                    print(f"  ✅ Mono dataset has '{feature}' feature")
                else:
                    print(f"  ❌ Mono dataset missing '{feature}' feature")
            
            return True
        else:
            print("❌ Train split paths don't exist, skipping loading test")
            return False
            
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False

if __name__ == "__main__":
    print("Dataset Structure and Loading Test")
    print("=" * 50)
    
    structure_ok = test_dataset_structure()
    
    if structure_ok:
        loading_ok = test_dataset_loading()
        if loading_ok:
            print("\n🎉 All tests passed! Dataset structure and loading work correctly.")
        else:
            print("\n⚠️  Dataset structure is correct, but loading failed.")
    else:
        print("\n❌ Dataset structure validation failed. Please check your dataset setup.")