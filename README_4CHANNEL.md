# 4-Channel RGB+Mono DiffBIR Setup

This document describes the modifications made to DiffBIR to support training with both RGB and monochromatic images fused into 4-channel input.

## Overview

The original DiffBIR uses 3-channel RGB images. This modified version:

1. **Loads two HuggingFace datasets**: One RGB and one monochromatic
2. **Fuses them into 4-channel input**: 3 RGB channels + 1 mono channel
3. **Uses your existing degraded images**: Instead of simulating degradation
4. **Caches datasets locally**: At your specified location `scratch/gilbreth/tang843`
5. **Includes extensive debug logging**: For monitoring image properties and model behavior

## Key Files Created/Modified

### Dataset Loading
- `diffbir/dataset/huggingface_rgbmono.py`: Custom dataset class for HuggingFace RGB+mono fusion
- Handles loading both datasets, fusing channels, and applying augmentations

### Model Architecture
- `diffbir/model/vae_4channel.py`: 4-channel VAE (encodes 4-channel input, decodes to 3-channel RGB)
- `diffbir/model/cldm_4channel.py`: 4-channel ControlLDM wrapper
- SwinIR already supports configurable input channels via `in_chans` parameter

### Training
- `configs/train/train_stage2_4channel.yaml`: Training configuration for 4-channel setup
- `train_stage2_4channel.py`: Modified training script with extensive debug logging

### Testing & Setup
- `test_4channel_setup.py`: Test script to verify dataset loading and model creation
- `setup_4channel.py`: Setup script for directories and dependencies

## Setup Instructions

### 1. Run Setup Script
```bash
python setup_4channel.py
```

This will:
- Check required packages
- Create necessary directories
- Check for model weights
- Create placeholder 4-channel SwinIR
- Test HuggingFace dataset access
- Update configuration paths

### 2. Download Required Weights
Download Stable Diffusion v2.1 checkpoint:
```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O weights/v2-1_512-ema-pruned.ckpt
```

### 3. Test the Setup
```bash
python test_4channel_setup.py
```

This will test:
- Dataset loading from HuggingFace
- 4-channel model creation
- Basic training pipeline components

### 4. Start Training
```bash
accelerate launch train_stage2_4channel.py --config configs/train/train_stage2_4channel_updated.yaml
```

## Dataset Configuration

Your HuggingFace datasets:
- **RGB Dataset**: `harshana95/quadratic_color_psfs_5db_updated_real_hybrid_Flickr2k_gt_v2_PCA_interp_file`
- **Mono Dataset**: `harshana95/quadratic_mono_psfs_5db_updated_real_hybrid_Flickr2k_gt_v2_PCA_interp_file`
- **Cache Location**: `scratch/gilbreth/tang843`
- **Expected Features**: Each dataset should have `train` and `validation` splits with `gt` and `blur` features

## Data Processing Pipeline

1. **Load RGB and Mono images** from HuggingFace datasets
2. **Process images**:
   - Convert mono to RGB format for consistent processing
   - Apply cropping (center/random/none)
   - Apply augmentations (flip, rotation) to both RGB and mono consistently
3. **Fuse channels**: Concatenate RGB (3 channels) + mono (1 channel) = 4 channels
4. **Normalize**:
   - GT: RGB channels converted BGR→RGB, normalized to [-1,1]; Mono channel normalized to [-1,1]
   - LQ: RGB channels converted BGR→RGB, normalized to [0,1]; Mono channel normalized to [0,1]

## Model Architecture Changes

### VAE (AutoencoderKL4Channel)
- **Encoder**: Accepts 4-channel input (RGB+mono)
- **Decoder**: Outputs 3-channel RGB (mono information is encoded in latent space)
- **Initialization**: Can load from 3-channel VAE by expanding first conv layer

### ControlLDM (ControlLDM4Channel)
- **Wrapper** around standard ControlLDM using 4-channel VAE
- **Condition preparation**: Handles 4-channel condition images
- **VAE encoding/decoding**: Manages 4→latent→3 channel transformation

### SwinIR
- **Native support** for configurable input channels
- **Configuration**: Set `in_chans: 4` in config
- **Mean normalization**: Handles non-3-channel inputs

## Debug Logging

The system includes extensive debug logging:

### Dataset Level
- Image shapes, dtypes, and value ranges
- Channel-wise statistics (RGB vs mono)
- Loading success/failure rates

### Training Level
- Batch shapes and ranges after each processing step
- VAE encoding/decoding shapes
- SwinIR output properties
- Loss values and training progress

### Model Level
- Layer input/output shapes
- Weight initialization details
- Gradient flow information

## Training Configuration

Key parameters in `train_stage2_4channel.yaml`:

```yaml
model:
  cldm:
    target: diffbir.model.cldm_4channel.ControlLDM4Channel
    params:
      vae_cfg:
        ddconfig:
          in_channels: 4  # Changed from 3
          out_ch: 3       # RGB output
      controlnet_cfg:
        hint_channels: 4  # 4-channel condition
  swinir:
    params:
      in_chans: 4  # 4-channel input

dataset:
  train:
    target: diffbir.dataset.huggingface_rgbmono.HuggingFaceRGBMonoDataset
    params:
      rgb_dataset_name: "your_rgb_dataset"
      mono_dataset_name: "your_mono_dataset"
      cache_dir: "scratch/gilbreth/tang843"

train:
  batch_size: 8  # Reduced for 4-channel processing
```

## Expected Output

During training, you should see debug output like:
```
Sample 0 - GT shape: (512, 512, 4), GT range: [-1.000, 1.000]
Sample 0 - LQ shape: (512, 512, 4), LQ range: [0.000, 1.000]
GT RGB channels range: [-1.000, 1.000]
GT Mono channel range: [-1.000, 1.000]
LQ RGB channels range: [0.000, 1.000]
LQ Mono channel range: [0.000, 1.000]
```

## Troubleshooting

### Common Issues

1. **HuggingFace dataset access**: Ensure internet connection and correct dataset names
2. **Memory issues**: Reduce batch size in config
3. **Channel mismatch**: Verify all models are configured for 4 channels
4. **Cache location**: Ensure `scratch/gilbreth/tang843` is accessible and has sufficient space

### Debug Steps

1. Run `python test_4channel_setup.py` to verify setup
2. Check training logs for shape mismatches
3. Monitor GPU memory usage
4. Verify dataset loading with smaller batch sizes

## Differences from Original DiffBIR

1. **Input channels**: 4 instead of 3
2. **Dataset source**: HuggingFace instead of local files  
3. **Degradation**: Uses real degraded images instead of simulation
4. **Cache location**: Configurable HuggingFace cache directory
5. **Debug logging**: Extensive shape and value monitoring
6. **Normalization**: Preserves original DiffBIR normalization for RGB, extends to mono channel

## Next Steps

1. **Train Stage 1**: You may need to train a 4-channel SwinIR first
2. **Fine-tune**: Adjust learning rates and batch sizes based on your hardware
3. **Evaluate**: Monitor training progress and sample quality
4. **Extend**: Add more sophisticated channel fusion techniques if needed

The system is designed to be as close to original DiffBIR as possible while supporting your 4-channel RGB+mono input requirement.