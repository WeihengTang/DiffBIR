# Dataset Configuration Guide

The codebase has been updated to use `load_from_disk` instead of `load_dataset` for loading datasets locally with **centralized configuration**.

## ✅ Simple Setup - Edit Only One File!

**File:** `dataset_config.py` (lines 11-12)

```python
RGB_DATASET_PATH = "datasets/color"      # Path to RGB dataset folder
MONO_DATASET_PATH = "datasets/mono"      # Path to mono dataset folder
```

**Expected Directory Structure:**
```
DiffBIR/
└── datasets/
    ├── color/
    │   ├── train/          # RGB training data (arrow files + dataset_info.json + state.json)
    │   └── validation/     # RGB validation data (arrow files + dataset_info.json + state.json)
    └── mono/
        ├── train/          # Mono training data (arrow files + dataset_info.json + state.json)
        └── validation/     # Mono validation data (arrow files + dataset_info.json + state.json)
```

That's it! All other files will automatically use these paths.

## Dataset Requirements
- Both RGB and mono datasets should be saved using `datasets.save_to_disk()`
- Each split folder (`train`, `validation`) should contain:
  - Arrow files with the actual data
  - `dataset_info.json` with dataset metadata  
  - `state.json` with dataset state information
- Each dataset should contain `gt` and `blur` image features
- The code will automatically load the correct split based on training configuration

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