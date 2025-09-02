<p align="center">
    <img src="assets/logo.png" width="400">
</p>

## DiffBIR: 4-Channel RGB+Mono Extension

**Extended from:** [DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior](https://arxiv.org/abs/2308.15070)

This repository extends the original DiffBIR to support **4-channel inputs (RGB + monochromatic)** with **centralized dataset configuration** using local datasets via `load_from_disk`.

<p align="center">
    <img src="assets/teaser.png">
</p>

---

<p align="center">
    <img src="assets/pipeline.png">
</p>

:star:If DiffBIR is helpful for you, please help star this repo. Thanks!:hugs:

## :book:Table Of Contents

- [Key Changes & Features](#changes)
- [Quick Setup (4-Channel)](#4channel)
- [Installation](#installation)
- [Original DiffBIR Features](#original)
- [Citation](#citation)

## <a name="changes"></a>:sparkles:Key Changes & Features

### :new: What's New in This Extension

- **4-Channel Input Support**: Fuses RGB (3 channels) + monochromatic (1 channel) images
- **Local Dataset Loading**: Uses `load_from_disk` instead of HuggingFace streaming
- **Centralized Configuration**: Single file (`dataset_config.py`) controls all dataset paths
- **Enhanced Models**: 4-channel VAE, ControlLDM, and SwinIR support
- **Simplified Setup**: One-time configuration for all training scripts

### :gear: Architecture Extensions

- **`diffbir/model/vae_4channel.py`**: 4-channel VAE (encodes 4→latent, decodes latent→3)
- **`diffbir/model/cldm_4channel.py`**: ControlLDM wrapper for 4-channel processing
- **`diffbir/dataset/huggingface_rgbmono.py`**: RGB+mono dataset fusion with augmentation
- **`dataset_config.py`**: Centralized path configuration
- **Training configs**: Optimized for 4-channel workflow

## <a name="4channel"></a>:rocket:Quick Setup (4-Channel RGB+Mono Training)

### :one: Dataset Configuration (One-Time Setup)

**Edit `dataset_config.py` (lines 11-12):**
```python
RGB_DATASET_PATH = "datasets/color"      # Path to RGB dataset folder
MONO_DATASET_PATH = "datasets/mono"      # Path to mono dataset folder
```

:sparkles: **That's it!** All training, testing, and setup scripts will automatically use these paths.

### :two: Run Setup Script

```bash
python setup_4channel.py
```

This will:
- Check required packages and create directories
- Verify dataset paths from `dataset_config.py`
- Download required model weights if missing
- Test 4-channel model compatibility

### :three: Download Required Weights

```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O weights/v2-1_512-ema-pruned.ckpt
```

### :four: Test Setup

```bash
python test_4channel_setup.py
```

### :five: Start Training

```bash
# Stage 1 (4-Channel SwinIR) - Optional, config handles 4-channel inputs automatically
# accelerate launch train_stage1.py --config configs/train/train_stage1.yaml

# Stage 2 (4-Channel ControlLDM)
accelerate launch train_stage2_4channel.py --config configs/train/train_stage2_4channel.yaml
```

### :file_folder: Dataset Requirements

Your datasets should be saved using `datasets.save_to_disk()` with structure:
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

Each dataset should have `gt` and `blur` features containing the respective image data.

### :bug: Troubleshooting

**Common Issues:**
1. **Dataset paths not found**: Edit `dataset_config.py` with correct paths
2. **Memory issues**: Reduce `batch_size` in config
3. **Channel mismatches**: Verify all models use 4-channel configuration

**Debug Commands:**
```bash
# Test dataset loading
python test_4channel_setup.py

# Verify centralized config
python -c "from dataset_config import get_dataset_paths, validate_dataset_paths; print(get_dataset_paths()); print(validate_dataset_paths())"
```

### :clipboard: Key Configuration

**Training config (`configs/train/train_stage2_4channel.yaml`):**
```yaml
model:
  cldm:
    target: diffbir.model.cldm_4channel.ControlLDM4Channel
    params:
      vae_cfg:
        ddconfig:
          in_channels: 4  # 4-channel input
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
      # Paths read automatically from dataset_config.py
      split: "train"
      out_size: 512

train:
  batch_size: 8  # Reduced for 4-channel processing
```

### :page_facing_up: Additional Documentation

- **[DATASET_SETUP.md](DATASET_SETUP.md)**: Detailed dataset configuration guide
- **Debug logging**: Extensive shape and value monitoring throughout training
- **Centralized config**: Single file (`dataset_config.py`) controls all dataset paths

## <a name="installation"></a>:gear:Installation

```shell
# clone this repo
git clone https://github.com/WeihengTang/DiffBIR.git  # Note: This is the extended 4-channel version
cd DiffBIR

# create environment
conda create -n diffbir python=3.10
conda activate diffbir
pip install -r requirements.txt
```

Our code is based on pytorch 2.2.2 for memory-efficient attention. If your GPU isn't compatible, downgrade to pytorch 1.13.1+cu116 and install xformers 0.0.16.

## <a name="original"></a>:book:Original DiffBIR Features

<details>
<summary><strong>Click to expand original DiffBIR documentation</strong></summary>

### Visual Results On Real-world Images

#### Blind Image Super-Resolution

[<img src="assets/visual_results/bsr6.png" height="223px"/>](https://imgsli.com/MTk5ODI3) [<img src="assets/visual_results/bsr7.png" height="223px"/>](https://imgsli.com/MTk5ODI4) [<img src="assets/visual_results/bsr4.png" height="223px"/>](https://imgsli.com/MTk5ODI1)

#### Blind Face Restoration

[<img src="assets/visual_results/whole_image1.png" height="370"/>](https://imgsli.com/MjA2MTU0) 
[<img src="assets/visual_results/whole_image2.png" height="370"/>](https://imgsli.com/MjA2MTQ4)

:star: Face and the background enhanced by DiffBIR.

#### Blind Image Denoising

[<img src="assets/visual_results/bid1.png" height="215px"/>](https://imgsli.com/MjUzNzkz) [<img src="assets/visual_results/bid3.png" height="215px"/>](https://imgsli.com/MjUzNzky)
[<img src="assets/visual_results/bid2.png" height="215px"/>](https://imgsli.com/MjUzNzkx)

### Quick Start (Original 3-Channel)

Run the following command to interact with the gradio website:

```shell
# For low-VRAM users, set captioner to ram or none
python run_gradio.py --captioner llava
```

### Pretrained Models

Here we list pretrained weight of stage 2 model (IRControlNet) and our trained SwinIR, which was used for degradation removal during the training of stage 2 model.

| Model Name | Description | HuggingFace | BaiduNetdisk | OpenXLab |
| :---------: | :----------: | :----------: | :----------: | :----------: |
| v2.1.pt | IRControlNet trained on filtered unsplash | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt) | N/A | N/A |
| v2.pth | IRControlNet trained on filtered laion2b-en  | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth) | [download](https://pan.baidu.com/s/1uTAFl13xgGAzrnznAApyng?pwd=xiu3)<br>(pwd: xiu3) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
| v1_general.pth | IRControlNet trained on ImageNet-1k | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth) | [download](https://pan.baidu.com/s/1PhXHAQSTOUX4Gy3MOc2t2Q?pwd=79n9)<br>(pwd: 79n9) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
| v1_face.pth | IRControlNet trained on FFHQ | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth) | [download](https://pan.baidu.com/s/1kvM_SB1VbXjbipLxdzlI3Q?pwd=n7dx)<br>(pwd: n7dx) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
| codeformer_swinir.ckpt | SwinIR trained on ImageNet-1k with CodeFormer degradation | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/codeformer_swinir.ckpt) | [download](https://pan.baidu.com/s/176fARg2ySYtDgX2vQOeRbA?pwd=vfif)<br>(pwd: vfif) | [download](https://openxlab.org.cn/models/detail/linxinqi/DiffBIR/tree/main) |
| realesrgan_s4_swinir_100k.pth | SwinIR trained on ImageNet-1k with Real-ESRGAN degradation | [download](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth) | N/A | N/A |

During inference, we use off-the-shelf models from other papers as the stage 1 model: [BSRNet](https://github.com/cszn/BSRGAN) for BSR, [SwinIR-Face](https://github.com/zsyOAOA/DifFace) used in DifFace for BFR, and [SCUNet-PSNR](https://github.com/cszn/SCUNet) for BID, while the trained IRControlNet remains **unchanged** for all tasks. Please check [code](diffbir/inference/pretrained_models.py) for more details. Thanks for their work!

## <a name="inference"></a>:crossed_swords:Inference

We provide some examples for inference, check [inference.py](inference.py) for more arguments. Pretrained weights will be **automatically downloaded**. For users with limited VRAM, please run the following scripts with [tiled sampling](#tiled-sampling).

### Blind Image Super-Resolution

```shell
# DiffBIR v2 (ECCV paper version)
python -u inference.py \
--task sr \
--upscale 4 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4 \
--input inputs/demo/bsr \
--output results/v2_demo_bsr \
--device cuda --precision fp32

# DiffBIR v2.1
python -u inference.py \
--task sr \
--upscale 4 \
--version v2.1 \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input inputs/demo/bsr \
--output results/v2.1_demo_bsr
```

### Blind Aligned-Face Restoration
<a name="inference_fr"></a>

```shell
# DiffBIR v2 (ECCV paper version)
python -u inference.py \
--task face \
--upscale 1 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4.0 \
--input inputs/demo/bfr/aligned \
--output results/v2_demo_bfr_aligned \
--device cuda --precision fp32

# DiffBIR v2.1
python -u inference.py \
--task face \
--upscale 1 \
--version v2.1 \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input inputs/demo/bfr/aligned \
--output results/v2.1_demo_bfr_aligned
```

### Blind Unaligned-Face Restoration

```shell
# DiffBIR v2 (ECCV paper version)
python -u inference.py \
--task face_background \
--upscale 2 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4.0 \
--input inputs/demo/bfr/whole_img \
--output results/v2_demo_bfr_unaligned \
--device cuda --precision fp32

# DiffBIR v2.1
python -u inference.py \
--task face_background \
--upscale 2 \
--version v2.1 \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input inputs/demo/bfr/whole_img \
--output results/v2.1_demo_bfr_unaligned
```

### Blind Image Denoising

```shell
# DiffBIR v2 (ECCV paper version)
python -u inference.py \
--task denoise \
--upscale 1 \
--version v2 \
--sampler spaced \
--steps 50 \
--captioner none \
--pos_prompt '' \
--neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
--cfg_scale 4.0 \
--input inputs/demo/bid \
--output results/v2_demo_bid \
--device cuda --precision fp32

# DiffBIR v2.1
python -u inference.py \
--task denoise \
--upscale 1 \
--version v2.1 \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input inputs/demo/bid \
--output results/v2.1_demo_bid
```

### Custom-model Inference

```shell
python -u inference.py \
--upscale 4 \
--version custom \
--train_cfg [path/to/training/config] \
--ckpt [path/to/saved/checkpoint] \
--captioner llava \
--cfg_scale 8 \
--noise_aug 0 \
--input inputs/demo/bsr \
--output results/custom_demo_bsr
```

### Other options

#### Tiled sampling
<a name="patch_based_sampling"></a>

Add the following arguments to enable tiled sampling:

```shell
[command...] \
# tiled inference for stage-1 model
--cleaner_tiled \
--cleaner_tile_size 256 \
--cleaner_tile_stride 128 \
# tiled inference for VAE encoding
--vae_encoder_tiled \
--vae_encoder_tile_size 256 \
# tiled inference for VAE decoding
--vae_decoder_tiled \
--vae_decoder_tile_size 256 \
# tiled inference for diffusion process
--cldm_tiled \
--cldm_tile_size 512 \
--cldm_tile_stride 256
```

Tiled sampling supports super-resolution with a large scale factor on low-VRAM graphics cards. Our tiled sampling is built upon [mixture-of-diffusers](https://github.com/albarji/mixture-of-diffusers) and [Tiled-VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111). Thanks for their work!

#### Condition as Start Point of Sampling

**This option only works with DiffBIR v1 and v2.** As proposed in [SeeSR](https://arxiv.org/abs/2311.16518), the LR embedding (LRE) strategy provides a more faithful
start point for sampling and consequently suppresses the artifacts in flat region:

```shell
[command...] --start_point_type cond
```

For our model, we use the diffused condition as start point. This option makes the results more stable and ensures that the outcomes from ODE samplers like DDIM and DPMS are normal. However, it may lead to a decrease in sample quality.

## <a name="train"></a>:stars:Train (Standard 3-Channel)

### Stage 1

First, we train a SwinIR, which will be used for degradation removal during the training of stage 2.

<a name="gen_file_list"></a>
1. Generate file list of training set and validation set, a file list looks like:

    ```txt
    /path/to/image_1
    /path/to/image_2
    /path/to/image_3
    ...
    ```

    You can write a simple python script or directly use shell command to produce file lists. Here is an example:
    
    ```shell
    # collect all iamge files in img_dir
    find [img_dir] -type f > files.list
    # shuffle collected files
    shuf files.list > files_shuf.list
    # pick train_size files in the front as training set
    head -n [train_size] files_shuf.list > files_shuf_train.list
    # pick remaining files as validation set
    tail -n +[train_size + 1] files_shuf.list > files_shuf_val.list
    ```

2. Fill in the [training configuration file](configs/train/train_stage1.yaml) with appropriate values.

3. Start training!

    ```shell
    accelerate launch train_stage1.py --config configs/train/train_stage1.yaml
    ```

### Stage 2

1. Download pretrained [Stable Diffusion v2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) to provide generative capabilities. :bulb:: If you have ran the [inference script](inference.py), the SD v2.1 checkpoint can be found in [weights](weights).

    ```shell
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt --no-check-certificate
    ```

2. Generate file list as mentioned [above](#gen_file_list). Currently, the training script of stage 2 doesn't support validation set, so you only need to create training file list.

3. Fill in the [training configuration file](configs/train/train_stage2.yaml) with appropriate values.

4. Start training!

    ```shell
    accelerate launch train_stage2.py --config configs/train/train_stage2.yaml
    ```

## <a name="4channel"></a>:new:4-Channel RGB+Mono Training

This repository now supports training with 4-channel inputs (3 RGB + 1 monochromatic) using local datasets via `load_from_disk`.

### :gear: Setup

#### 1. Dataset Configuration (One-Time Setup)

**Edit `dataset_config.py` (lines 11-12):**
```python
RGB_DATASET_PATH = "datasets/color"      # Path to RGB dataset folder
MONO_DATASET_PATH = "datasets/mono"      # Path to mono dataset folder
```

:sparkles: **That's it!** All training, testing, and setup scripts will automatically use these paths.

#### 2. Run Setup Script

```bash
python setup_4channel.py
```

This will:
- Check required packages and create directories
- Verify dataset paths from `dataset_config.py`
- Download required model weights if missing
- Test 4-channel model compatibility

#### 3. Download Required Weights

```bash
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -O weights/v2-1_512-ema-pruned.ckpt
```

#### 4. Test Setup

```bash
python test_4channel_setup.py
```

### :rocket: Training

#### Stage 1 (4-Channel SwinIR)
```bash
# You may need to train a 4-channel SwinIR first or modify an existing one
# The config automatically handles 4-channel inputs via in_chans: 4
```

#### Stage 2 (4-Channel ControlLDM)
```bash
accelerate launch train_stage2_4channel.py --config configs/train/train_stage2_4channel.yaml
```

### :wrench: Architecture Changes

#### Models Modified for 4-Channel Support:
- **VAE (`diffbir/model/vae_4channel.py`)**: Encodes 4-channel input, decodes to 3-channel RGB
- **ControlLDM (`diffbir/model/cldm_4channel.py`)**: Wrapper for 4-channel VAE integration
- **SwinIR**: Native support via `in_chans: 4` parameter

#### Dataset Processing:
1. **Load** RGB and mono datasets from disk using `load_from_disk`
2. **Process** both images with consistent augmentations
3. **Fuse** RGB (3 channels) + mono (1 channel) = 4 channels
4. **Normalize** following DiffBIR standards:
   - GT: RGB→[-1,1], Mono→[-1,1]
   - LQ: RGB→[0,1], Mono→[0,1]

### :file_folder: Dataset Requirements

Your datasets should be saved using `datasets.save_to_disk()` with structure:
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

Each dataset should have `gt` and `blur` features containing the respective image data.

### :clipboard: Key Configuration

**Training config (`configs/train/train_stage2_4channel.yaml`):**
```yaml
model:
  cldm:
    target: diffbir.model.cldm_4channel.ControlLDM4Channel
    params:
      vae_cfg:
        ddconfig:
          in_channels: 4  # 4-channel input
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
      # Paths read automatically from dataset_config.py
      split: "train"
      out_size: 512

train:
  batch_size: 8  # Reduced for 4-channel processing
```

### :bug: Troubleshooting

**Common Issues:**
1. **Dataset paths not found**: Edit `dataset_config.py` with correct paths
2. **Memory issues**: Reduce `batch_size` in config
3. **Channel mismatches**: Verify all models use 4-channel configuration

**Debug Commands:**
```bash
# Test dataset loading
python test_4channel_setup.py

# Verify centralized config
python -c "from dataset_config import get_dataset_paths, validate_dataset_paths; print(get_dataset_paths()); print(validate_dataset_paths())"
```

### :page_facing_up: Documentation

- **[DATASET_SETUP.md](DATASET_SETUP.md)**: Detailed dataset configuration guide
- **Debug logging**: Extensive shape and value monitoring throughout training
- **Centralized config**: Single file (`dataset_config.py`) controls all dataset paths


</details>

## <a name="citation"></a>Citation

Please cite us if our work is useful for your research.

```
@misc{lin2024diffbir,
      title={DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior}, 
      author={Xinqi Lin and Jingwen He and Ziyan Chen and Zhaoyang Lyu and Bo Dai and Fanghua Yu and Wanli Ouyang and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2308.15070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This project is based on [ControlNet](https://github.com/lllyasviel/ControlNet) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.

## Contact

If you have any questions, please feel free to contact with me at linxinqi23@mails.ucas.ac.cn.