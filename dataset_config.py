"""
Centralized dataset configuration for DiffBIR 4-channel training.
Set your dataset paths here once and they will be used across all files.
"""

import os

# =============================================================================
# TODO: SET YOUR DATASET PATHS HERE
# =============================================================================
RGB_DATASET_PATH = "datasets/color"
MONO_DATASET_PATH = "datasets/mono"

# =============================================================================
# Helper functions (do not modify)
# =============================================================================

def get_dataset_paths():
    """Get the configured dataset paths."""
    # Convert to absolute paths if they're relative
    rgb_path = RGB_DATASET_PATH if os.path.isabs(RGB_DATASET_PATH) else os.path.abspath(RGB_DATASET_PATH)
    mono_path = MONO_DATASET_PATH if os.path.isabs(MONO_DATASET_PATH) else os.path.abspath(MONO_DATASET_PATH)
    
    return {
        'rgb_dataset_path': rgb_path,
        'mono_dataset_path': mono_path
    }

def validate_dataset_paths():
    """Validate that the dataset paths exist and have correct structure."""
    errors = []
    paths = get_dataset_paths()
    
    rgb_path = paths['rgb_dataset_path']
    mono_path = paths['mono_dataset_path']
    
    # Check if base paths exist
    if not os.path.exists(rgb_path):
        errors.append(f"RGB dataset path does not exist: {rgb_path}")
    else:
        # Check for train and validation splits
        train_path = os.path.join(rgb_path, 'train')
        val_path = os.path.join(rgb_path, 'validation')
        if not os.path.exists(train_path):
            errors.append(f"RGB train split not found: {train_path}")
        if not os.path.exists(val_path):
            errors.append(f"RGB validation split not found: {val_path} (optional)")
    
    if not os.path.exists(mono_path):
        errors.append(f"Mono dataset path does not exist: {mono_path}")
    else:
        # Check for train and validation splits
        train_path = os.path.join(mono_path, 'train')
        val_path = os.path.join(mono_path, 'validation')
        if not os.path.exists(train_path):
            errors.append(f"Mono train split not found: {train_path}")
        if not os.path.exists(val_path):
            errors.append(f"Mono validation split not found: {val_path} (optional)")
    
    if RGB_DATASET_PATH == "/path/to/your/rgb_dataset":
        errors.append("Please set RGB_DATASET_PATH in dataset_config.py")
        
    if MONO_DATASET_PATH == "/path/to/your/mono_dataset":
        errors.append("Please set MONO_DATASET_PATH in dataset_config.py")
    
    return errors

def get_dataset_config_for_yaml():
    """Get dataset config in format suitable for YAML config files."""
    paths = get_dataset_paths()
    return {
        'rgb_dataset_path': paths['rgb_dataset_path'],
        'mono_dataset_path': paths['mono_dataset_path']
    }