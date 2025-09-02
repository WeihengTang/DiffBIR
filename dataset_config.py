"""
Centralized dataset configuration for DiffBIR 4-channel training.
Set your dataset paths here once and they will be used across all files.
"""

import os

# =============================================================================
# TODO: SET YOUR DATASET PATHS HERE
# =============================================================================
RGB_DATASET_PATH = "/path/to/your/rgb_dataset"
MONO_DATASET_PATH = "/path/to/your/mono_dataset"

# =============================================================================
# Helper functions (do not modify)
# =============================================================================

def get_dataset_paths():
    """Get the configured dataset paths."""
    return {
        'rgb_dataset_path': RGB_DATASET_PATH,
        'mono_dataset_path': MONO_DATASET_PATH
    }

def validate_dataset_paths():
    """Validate that the dataset paths exist."""
    errors = []
    
    if not os.path.exists(RGB_DATASET_PATH):
        errors.append(f"RGB dataset path does not exist: {RGB_DATASET_PATH}")
    
    if not os.path.exists(MONO_DATASET_PATH):
        errors.append(f"Mono dataset path does not exist: {MONO_DATASET_PATH}")
    
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