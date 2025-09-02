#!/usr/bin/env python3
"""
Setup script for 4-channel RGB+Mono DiffBIR training
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required packages are installed"""
    logger.info("Checking requirements...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "datasets",
        "huggingface_hub",
        "accelerate",
        "omegaconf",
        "einops",
        "tqdm",
        "lpips",
        "timm",
        "PIL",
        "cv2"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("All required packages are installed")
    return True


def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "scratch/gilbreth/tang843",  # HuggingFace cache directory
        "experiments/4channel_rgbmono",  # Experiment directory
        "weights",  # Model weights directory
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return True


def check_model_weights():
    """Check if required model weights exist"""
    logger.info("Checking model weights...")
    
    weights_info = {
        "weights/v2-1_512-ema-pruned.ckpt": {
            "url": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
            "description": "Stable Diffusion v2.1 checkpoint"
        }
    }
    
    missing_weights = []
    for weight_path, info in weights_info.items():
        if not os.path.exists(weight_path):
            missing_weights.append((weight_path, info))
    
    if missing_weights:
        logger.warning("Missing model weights:")
        for weight_path, info in missing_weights:
            logger.warning(f"  {weight_path} - {info['description']}")
            logger.warning(f"    Download from: {info['url']}")
        
        logger.info("\nTo download Stable Diffusion v2.1:")
        logger.info("wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O weights/v2-1_512-ema-pruned.ckpt")
        
        return False
    
    logger.info("All required model weights are present")
    return True


def create_swinir_4channel():
    """Create a 4-channel SwinIR checkpoint from 3-channel if needed"""
    logger.info("Setting up 4-channel SwinIR...")
    
    swinir_4ch_path = "weights/swinir_4channel.pth"
    
    if os.path.exists(swinir_4ch_path):
        logger.info(f"4-channel SwinIR already exists: {swinir_4ch_path}")
        return True
    
    # For now, we'll create a placeholder - you would need to adapt a real 3-channel SwinIR
    logger.info("Creating placeholder 4-channel SwinIR checkpoint...")
    
    try:
        import torch
        from omegaconf import OmegaConf
        from diffbir.utils.common import instantiate_from_config
        
        # Load config
        config_path = "configs/train/train_stage2_4channel.yaml"
        cfg = OmegaConf.load(config_path)
        
        # Create SwinIR
        swinir = instantiate_from_config(cfg.model.swinir)
        
        # Save checkpoint
        torch.save(swinir.state_dict(), swinir_4ch_path)
        logger.info(f"Created placeholder SwinIR checkpoint: {swinir_4ch_path}")
        logger.warning("This is a placeholder! You should train a proper 4-channel SwinIR or adapt a 3-channel one.")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create SwinIR checkpoint: {e}")
        return False


def test_huggingface_access():
    """Test access to HuggingFace datasets"""
    logger.info("Testing HuggingFace dataset access...")
    
    try:
        from datasets import load_from_disk
        try:
            from dataset_config import get_dataset_paths, validate_dataset_paths
        except ImportError:
            logger.error("dataset_config.py not found. Please create it with your dataset paths.")
            return False
        
        # Test loading datasets from centralized config
        dataset_paths = get_dataset_paths()
        rgb_dataset_path = dataset_paths['rgb_dataset_path']
        mono_dataset_path = dataset_paths['mono_dataset_path']
        
        # Validate paths
        validation_errors = validate_dataset_paths()
        if validation_errors:
            logger.error("Dataset path validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info(f"Testing RGB dataset at: {rgb_dataset_path}")
        logger.info("RGB dataset path exists")
        
        logger.info(f"Testing Mono dataset at: {mono_dataset_path}")
        logger.info("Mono dataset path exists")
        
        logger.info("HuggingFace dataset access test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"HuggingFace dataset access test FAILED: {e}")
        logger.error("Make sure you have internet access and the dataset names are correct")
        return False


def update_config_paths():
    """Update configuration file with absolute paths"""
    logger.info("Updating configuration paths...")
    
    try:
        config_path = "configs/train/train_stage2_4channel.yaml"
        
        # Read config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Get absolute paths
        current_dir = os.path.abspath('.')
        
        # Replace placeholder paths
        replacements = {
            'scratch/gilbreth/tang843': os.path.join(current_dir, 'scratch/gilbreth/tang843'),
            'weights/v2-1_512-ema-pruned.ckpt': os.path.join(current_dir, 'weights/v2-1_512-ema-pruned.ckpt'),
            'experiments/4channel_rgbmono': os.path.join(current_dir, 'experiments/4channel_rgbmono'),
            'weights/swinir_4channel.pth': os.path.join(current_dir, 'weights/swinir_4channel.pth'),
        }
        
        updated_content = config_content
        for placeholder, absolute_path in replacements.items():
            updated_content = updated_content.replace(placeholder, absolute_path)
        
        # Write updated config
        updated_config_path = "configs/train/train_stage2_4channel_updated.yaml"
        with open(updated_config_path, 'w') as f:
            f.write(updated_content)
        
        logger.info(f"Updated config saved to: {updated_config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


def main():
    """Run setup"""
    logger.info("Starting 4-channel RGB+Mono DiffBIR setup...")
    
    setup_steps = [
        ("Check Requirements", check_requirements),
        ("Setup Directories", setup_directories),
        ("Check Model Weights", check_model_weights),
        ("Create 4-channel SwinIR", create_swinir_4channel),
        ("Test HuggingFace Access", test_huggingface_access),
        ("Update Config Paths", update_config_paths),
    ]
    
    results = {}
    for step_name, step_func in setup_steps:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {step_name}")
        results[step_name] = step_func()
        logger.info(f"{step_name}: {'COMPLETED' if results[step_name] else 'FAILED'}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SETUP SUMMARY:")
    all_passed = True
    for step_name, passed in results.items():
        status = "COMPLETED" if passed else "FAILED"
        logger.info(f"  {step_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\nSetup COMPLETED successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Download SD v2.1 checkpoint if not already present")
        logger.info("2. Run test script: python test_4channel_setup.py")
        logger.info("3. Start training: accelerate launch train_stage2_4channel.py --config configs/train/train_stage2_4channel_updated.yaml")
    else:
        logger.info("\nSetup INCOMPLETE. Please resolve the failed steps above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)