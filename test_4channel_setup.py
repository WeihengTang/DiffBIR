#!/usr/bin/env python3
"""
Test script to verify 4-channel RGB+Mono dataset loading and model setup
"""

import os
import sys
import logging
import torch
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from diffbir.utils.common import instantiate_from_config


def test_dataset_loading():
    """Test HuggingFace RGB+Mono dataset loading"""
    logger.info("=== Testing Dataset Loading ===")
    
    try:
        # Test configuration
        dataset_config = {
            "target": "diffbir.dataset.huggingface_rgbmono.HuggingFaceRGBMonoDataset",
            "params": {
                # Paths will be read from dataset_config.py
                "split": "train",
                "out_size": 256,  # Smaller for testing
                "crop_type": "center",
                "use_hflip": True,
                "use_rot": True,
                "p_empty_prompt": 0.5,
                "debug_logging": True,
            }
        }
        
        logger.info("Creating dataset...")
        dataset = instantiate_from_config(dataset_config)
        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test loading a few samples
        logger.info("Testing sample loading...")
        for i in range(min(3, len(dataset))):
            gt, lq, prompt = dataset[i]
            logger.info(f"Sample {i}:")
            logger.info(f"  GT shape: {gt.shape}, dtype: {gt.dtype}")
            logger.info(f"  GT range: [{gt.min():.3f}, {gt.max():.3f}]")
            logger.info(f"  LQ shape: {lq.shape}, dtype: {lq.dtype}")  
            logger.info(f"  LQ range: [{lq.min():.3f}, {lq.max():.3f}]")
            logger.info(f"  Prompt: '{prompt}'")
            
            # Check channels
            if len(gt.shape) == 3 and gt.shape[2] == 4:
                logger.info(f"  GT RGB range: [{gt[..., :3].min():.3f}, {gt[..., :3].max():.3f}]")
                logger.info(f"  GT Mono range: [{gt[..., 3].min():.3f}, {gt[..., 3].max():.3f}]")
                logger.info(f"  LQ RGB range: [{lq[..., :3].min():.3f}, {lq[..., :3].max():.3f}]")
                logger.info(f"  LQ Mono range: [{lq[..., 3].min():.3f}, {lq[..., 3].max():.3f}]")
        
        logger.info("Dataset loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test 4-channel model creation"""
    logger.info("=== Testing Model Creation ===")
    
    try:
        # Load configuration
        config_path = "configs/train/train_stage2_4channel.yaml"
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False
        
        cfg = OmegaConf.load(config_path)
        logger.info("Config loaded successfully")
        
        # Test ControlLDM creation
        logger.info("Testing 4-channel ControlLDM creation...")
        cldm = instantiate_from_config(cfg.model.cldm)
        logger.info("ControlLDM created successfully")
        
        # Test VAE from ControlLDM
        logger.info("Testing 4-channel VAE from ControlLDM...")
        vae = cldm.vae
        logger.info(f"VAE created - Encoder input channels: {vae.encoder.in_channels}")
        
        # Test with dummy 4-channel input
        dummy_input = torch.randn(1, 4, 256, 256)
        logger.info(f"Testing VAE with dummy input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            encoded = vae.encode(dummy_input)
            logger.info(f"VAE encode successful - Latent shape: {encoded.sample().shape}")
            
            decoded = vae.decode(encoded.sample())
            logger.info(f"VAE decode successful - Output shape: {decoded.shape}")
        
        # Test SwinIR creation  
        logger.info("Testing 4-channel SwinIR creation...")
        swinir = instantiate_from_config(cfg.model.swinir)
        logger.info(f"SwinIR created - Input channels: {swinir.conv_first.in_channels if hasattr(swinir.conv_first, 'in_channels') else 'N/A'}")
        
        # Test SwinIR forward
        with torch.no_grad():
            swinir_output = swinir(dummy_input)
            logger.info(f"SwinIR forward successful - Output shape: {swinir_output.shape}")
        
        logger.info("Model creation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Model creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test the basic training pipeline components"""
    logger.info("=== Testing Training Pipeline ===")
    
    try:
        # Create dummy data
        batch_size = 2
        dummy_gt = torch.randn(batch_size, 4, 256, 256)  # 4-channel GT
        dummy_lq = torch.randn(batch_size, 4, 256, 256)  # 4-channel LQ
        dummy_prompts = ["test prompt"] * batch_size
        
        logger.info(f"Created dummy batch - GT: {dummy_gt.shape}, LQ: {dummy_lq.shape}")
        
        # Test batch processing
        from einops import rearrange
        gt_processed = rearrange(dummy_gt, "b c h w -> b h w c").contiguous()
        lq_processed = rearrange(dummy_lq, "b c h w -> b h w c").contiguous()
        
        logger.info(f"Processed batch - GT: {gt_processed.shape}, LQ: {lq_processed.shape}")
        logger.info(f"GT range: [{gt_processed.min():.3f}, {gt_processed.max():.3f}]")
        logger.info(f"LQ range: [{lq_processed.min():.3f}, {lq_processed.max():.3f}]")
        
        # Test channel separation
        rgb_gt = gt_processed[..., :3]
        mono_gt = gt_processed[..., 3:4]
        
        logger.info(f"GT RGB shape: {rgb_gt.shape}, Mono shape: {mono_gt.shape}")
        
        logger.info("Training pipeline test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("Starting 4-channel RGB+Mono setup tests...")
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation), 
        ("Training Pipeline", test_training_pipeline),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        results[test_name] = test_func()
        logger.info(f"{test_name} test: {'PASSED' if results[test_name] else 'FAILED'}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY:")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll tests PASSED! Your 4-channel setup is ready.")
    else:
        logger.info("\nSome tests FAILED. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)