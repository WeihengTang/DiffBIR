from typing import Dict, Union, List, Optional, Any, Mapping
import time
import random
import logging
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from datasets import load_from_disk
from huggingface_hub import snapshot_download
import io
import sys
import os

# Add root directory to path to import dataset_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dataset_config import get_dataset_paths
except ImportError:
    # Fallback if dataset_config.py doesn't exist
    def get_dataset_paths():
        return {
            'rgb_dataset_path': '/path/to/your/rgb_dataset',
            'mono_dataset_path': '/path/to/your/mono_dataset'
        }

from .utils import center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config

logger = logging.getLogger(__name__)


class HuggingFaceRGBMonoDataset(data.Dataset):
    """
    Custom dataset class for loading RGB and monochromatic images from HuggingFace,
    then fusing them into 4-channel input for DiffBIR training.
    """

    def __init__(
        self,
        rgb_dataset_path: str = None,
        mono_dataset_path: str = None,
        split: str = "train",
        out_size: int = 512,
        crop_type: str = "center",
        use_hflip: bool = True,
        use_rot: bool = True,
        p_empty_prompt: float = 0.5,
        debug_logging: bool = True,
    ) -> "HuggingFaceRGBMonoDataset":
        super(HuggingFaceRGBMonoDataset, self).__init__()
        
        # Use centralized config if paths not provided
        if rgb_dataset_path is None or mono_dataset_path is None:
            dataset_paths = get_dataset_paths()
            rgb_dataset_path = rgb_dataset_path or dataset_paths['rgb_dataset_path']
            mono_dataset_path = mono_dataset_path or dataset_paths['mono_dataset_path']
        
        self.rgb_dataset_path = rgb_dataset_path
        self.mono_dataset_path = mono_dataset_path
        self.split = split
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.p_empty_prompt = p_empty_prompt
        self.debug_logging = debug_logging
        
        assert self.crop_type in ["none", "center", "random"]
        
        if self.debug_logging:
            logger.info(f"Loading RGB dataset from: {rgb_dataset_path}")
            logger.info(f"Loading Mono dataset from: {mono_dataset_path}")
            logger.info(f"Split: {split}")
        
        # Load datasets from disk
        try:
            rgb_dataset_full = load_from_disk(rgb_dataset_path)
            mono_dataset_full = load_from_disk(mono_dataset_path)
            
            # Apply split if datasets have splits, otherwise use the full dataset
            if hasattr(rgb_dataset_full, split):
                self.rgb_dataset = rgb_dataset_full[split]
            else:
                self.rgb_dataset = rgb_dataset_full
                if self.debug_logging:
                    logger.warning(f"Split '{split}' not found in RGB dataset, using full dataset")
            
            if hasattr(mono_dataset_full, split):
                self.mono_dataset = mono_dataset_full[split]
            else:
                self.mono_dataset = mono_dataset_full
                if self.debug_logging:
                    logger.warning(f"Split '{split}' not found in mono dataset, using full dataset")
                
        except Exception as e:
            logger.error(f"Error loading datasets from disk: {e}")
            raise
        
        # Verify datasets have same length
        rgb_len = len(self.rgb_dataset)
        mono_len = len(self.mono_dataset)
        if rgb_len != mono_len:
            logger.warning(f"RGB dataset length ({rgb_len}) != Mono dataset length ({mono_len})")
            # Use the minimum length
            self.dataset_length = min(rgb_len, mono_len)
        else:
            self.dataset_length = rgb_len
        
        if self.debug_logging:
            logger.info(f"Dataset loaded successfully with {self.dataset_length} samples")
            logger.info(f"RGB dataset features: {list(self.rgb_dataset.features.keys())}")
            logger.info(f"Mono dataset features: {list(self.mono_dataset.features.keys())}")
        
    def load_and_process_image(
        self, 
        image_data: Any, 
        max_retry: int = 5,
        is_mono: bool = False
    ) -> Optional[np.ndarray]:
        """
        Load and process image data from HuggingFace dataset.
        Handles both PIL Images and image bytes.
        """
        retry_count = 0
        while retry_count < max_retry:
            try:
                # Handle PIL Image objects
                if hasattr(image_data, 'convert'):
                    if is_mono:
                        # Convert to grayscale for mono images
                        image = image_data.convert('L')
                        # Convert back to RGB for consistent processing
                        image = image.convert('RGB')
                    else:
                        image = image_data.convert('RGB')
                
                # Handle bytes data
                elif isinstance(image_data, bytes):
                    image = Image.open(io.BytesIO(image_data))
                    if is_mono:
                        image = image.convert('L').convert('RGB')
                    else:
                        image = image.convert('RGB')
                
                # Handle other formats
                else:
                    logger.error(f"Unsupported image data type: {type(image_data)}")
                    return None
                
                # Apply cropping
                if self.crop_type != "none":
                    if image.height == self.out_size and image.width == self.out_size:
                        image_array = np.array(image)
                    else:
                        if self.crop_type == "center":
                            image_array = center_crop_arr(image, self.out_size)
                        elif self.crop_type == "random":
                            image_array = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
                else:
                    assert image.height == self.out_size and image.width == self.out_size
                    image_array = np.array(image)
                
                if self.debug_logging and retry_count == 0:
                    logger.debug(f"Image processed - Shape: {image_array.shape}, "
                               f"Dtype: {image_array.dtype}, "
                               f"Range: [{image_array.min()}, {image_array.max()}], "
                               f"Is mono: {is_mono}")
                
                return image_array
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retry:
                    logger.error(f"Failed to load image after {max_retry} retries: {e}")
                    return None
                time.sleep(0.1)
                
        return None
    
    def augment_images(self, rgb_img: np.ndarray, mono_img: np.ndarray):
        """Apply the same augmentation to both RGB and mono images"""
        # Random horizontal flip
        if self.use_hflip and random.random() < 0.5:
            rgb_img = np.fliplr(rgb_img).copy()
            mono_img = np.fliplr(mono_img).copy()
        
        # Random rotation (0, 90, 180, 270 degrees)
        if self.use_rot and random.random() < 0.5:
            # Random 90-degree rotations
            k = random.randint(0, 3)
            if k > 0:
                rgb_img = np.rot90(rgb_img, k).copy()
                mono_img = np.rot90(mono_img, k).copy()
        
        return rgb_img, mono_img
    
    def fuse_rgb_mono(self, rgb_img: np.ndarray, mono_img: np.ndarray) -> np.ndarray:
        """
        Fuse RGB and monochromatic images into a 4-channel image.
        RGB: 3 channels, Mono: 1 channel (converted from the first channel of mono_img)
        """
        # Extract single channel from mono image (convert RGB mono to single channel)
        if len(mono_img.shape) == 3:
            # Take the red channel (they should be the same in grayscale converted to RGB)
            mono_channel = mono_img[..., 0:1]
        else:
            mono_channel = mono_img[..., np.newaxis]
        
        # Concatenate RGB + mono to create 4-channel image
        fused = np.concatenate([rgb_img, mono_channel], axis=-1)
        
        if self.debug_logging:
            logger.debug(f"Fused image - RGB shape: {rgb_img.shape}, "
                       f"Mono shape: {mono_channel.shape}, "
                       f"Fused shape: {fused.shape}")
        
        return fused
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # Handle index bounds
        index = index % self.dataset_length
        
        # Load images with retry mechanism
        rgb_img = None
        mono_img = None
        attempts = 0
        max_attempts = 5
        
        while (rgb_img is None or mono_img is None) and attempts < max_attempts:
            try:
                # Get RGB data
                rgb_sample = self.rgb_dataset[index]
                rgb_gt_data = rgb_sample['gt']
                rgb_blur_data = rgb_sample['blur']
                
                # Get Mono data  
                mono_sample = self.mono_dataset[index]
                mono_gt_data = mono_sample['gt']
                mono_blur_data = mono_sample['blur']
                
                # Process GT images (high quality)
                rgb_gt = self.load_and_process_image(rgb_gt_data, is_mono=False)
                mono_gt = self.load_and_process_image(mono_gt_data, is_mono=True)
                
                # Process blur images (low quality)  
                rgb_blur = self.load_and_process_image(rgb_blur_data, is_mono=False)
                mono_blur = self.load_and_process_image(mono_blur_data, is_mono=True)
                
                if rgb_gt is not None and mono_gt is not None and rgb_blur is not None and mono_blur is not None:
                    rgb_img = {'gt': rgb_gt, 'blur': rgb_blur}
                    mono_img = {'gt': mono_gt, 'blur': mono_blur}
                else:
                    raise ValueError("Failed to load one or more images")
                    
            except Exception as e:
                logger.warning(f"Failed to load sample {index}, trying next: {e}")
                index = random.randint(0, self.dataset_length - 1)
                attempts += 1
        
        if rgb_img is None or mono_img is None:
            raise RuntimeError(f"Failed to load valid images after {max_attempts} attempts")
        
        # Apply augmentations to both GT and blur images
        rgb_gt_aug, mono_gt_aug = self.augment_images(rgb_img['gt'], mono_img['gt'])
        rgb_blur_aug, mono_blur_aug = self.augment_images(rgb_img['blur'], mono_img['blur'])
        
        # Fuse RGB and mono for both GT and LQ
        fused_gt = self.fuse_rgb_mono(rgb_gt_aug, mono_gt_aug)
        fused_lq = self.fuse_rgb_mono(rgb_blur_aug, mono_blur_aug)
        
        # Convert to [0,1] float32 and apply normalization like original DiffBIR
        # GT: BGR to RGB, [-1, 1] (but now 4 channels: RGB + mono)
        gt = (fused_gt[..., ::-1] / 255.0 * 2 - 1).astype(np.float32)
        # Wait, we need to handle 4 channels properly
        # For 4-channel: RGB channels + mono channel
        gt_rgb = (fused_gt[..., :3][..., ::-1] / 255.0 * 2 - 1).astype(np.float32)  # RGB channels, BGR->RGB, [-1,1]
        gt_mono = (fused_gt[..., 3:4] / 255.0 * 2 - 1).astype(np.float32)  # Mono channel, [-1,1]
        gt = np.concatenate([gt_rgb, gt_mono], axis=-1)
        
        # LQ: BGR to RGB, [0, 1] (but now 4 channels)  
        lq_rgb = fused_lq[..., :3][..., ::-1].astype(np.float32) / 255.0  # RGB channels, BGR->RGB, [0,1]
        lq_mono = fused_lq[..., 3:4].astype(np.float32) / 255.0  # Mono channel, [0,1]
        lq = np.concatenate([lq_rgb, lq_mono], axis=-1)
        
        # Empty prompt with probability p_empty_prompt
        if random.random() < self.p_empty_prompt:
            prompt = ""
        else:
            prompt = ""  # You can add text prompts here if available in your dataset
        
        if self.debug_logging:
            logger.debug(f"Sample {index} - GT shape: {gt.shape}, GT range: [{gt.min():.3f}, {gt.max():.3f}]")
            logger.debug(f"Sample {index} - LQ shape: {lq.shape}, LQ range: [{lq.min():.3f}, {lq.max():.3f}]")
            logger.debug(f"Sample {index} - GT dtype: {gt.dtype}, LQ dtype: {lq.dtype}")
        
        return gt, lq, prompt
    
    def __len__(self) -> int:
        return self.dataset_length