# Dataset Configuration Guide

The codebase has been updated to use `load_from_disk` instead of `load_dataset` for loading datasets locally with **centralized configuration**.

## ✅ Simple Setup - Edit Only One File!

**File:** `dataset_config.py` (lines 11-12)

```python
RGB_DATASET_PATH = "/path/to/your/rgb_dataset"
MONO_DATASET_PATH = "/path/to/your/mono_dataset"
```

That's it! All other files will automatically use these paths.

## Dataset Requirements
- Both RGB and mono datasets should be saved using `datasets.save_to_disk()`
- The datasets should have the same structure as the original HuggingFace datasets
- Each dataset should contain `gt` and `blur` image fields
- The paths should point to the root directory of each saved dataset

## Instructions
1. Open `dataset_config.py`
2. Replace `/path/to/your/rgb_dataset` with the actual path to your RGB dataset
3. Replace `/path/to/your/mono_dataset` with the actual path to your mono dataset  
4. Save the file - all training, testing, and setup scripts will use these paths automatically

## How It Works
- `diffbir/dataset/huggingface_rgbmono.py` automatically imports from `dataset_config.py`
- `configs/train/train_stage2_4channel.yaml` no longer needs dataset paths
- `test_4channel_setup.py` and `setup_4channel.py` read from centralized config
- The system includes validation to ensure paths exist and are properly configured