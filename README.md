<p align="center">
    <img src="assets/logo.png" width="400">
</p>

## DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

[Paper](https://arxiv.org/abs/2308.15070) | [Project Page](https://0x3f3f3f3fun.github.io/projects/diffbir/)

![visitors](https://visitor-badge.laobi.icu/badge?page_id=XPixelGroup/DiffBIR) [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/linxinqi/DiffBIR-official) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/DiffBIR-colab/blob/main/DiffBIR_colab.ipynb) [![Try a demo on Replicate](https://replicate.com/zsxkib/diffbir/badge)](https://replicate.com/zsxkib/diffbir)

[Xinqi Lin](https://0x3f3f3f3fun.github.io/)<sup>1,\*</sup>, [Jingwen He](https://github.com/hejingwenhejingwen)<sup>2,3,\*</sup>, [Ziyan Chen](https://orcid.org/0000-0001-6277-5635)<sup>1</sup>, [Zhaoyang Lyu](https://scholar.google.com.tw/citations?user=gkXFhbwAAAAJ&hl=en)<sup>2</sup>, [Bo Dai](http://daibo.info/)<sup>2</sup>, [Fanghua Yu](https://github.com/Fanghua-Yu)<sup>1</sup>, [Wanli Ouyang](https://wlouyang.github.io/)<sup>2</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao)<sup>2</sup>, [Chao Dong](http://xpixel.group/2010/01/20/chaodong.html)<sup>1,2</sup>

<sup>1</sup>Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences<br><sup>2</sup>Shanghai AI Laboratory<br><sup>3</sup>The Chinese University of Hong Kong

<p align="center">
    <img src="assets/teaser.png">
</p>

---

<p align="center">
    <img src="assets/pipeline.png">
</p>

:star:If DiffBIR is helpful for you, please help star this repo. Thanks!:hugs:

## :book:Table Of Contents

- [Update](#update)
- [Visual Results On Real-world Images](#visual_results)
- [TODO](#todo)
- [Installation](#installation)
- [Quick Start](#quick_start)
- [Pretrained Models](#pretrained_models)
- [Inference](#inference)
- [Train (Standard 3-Channel)](#train)
- [4-Channel RGB+Mono Training](#4channel)
- [Citation](#citation)

## <a name="update"></a>:new:Update

- **2025.09.02**: ✅ Added 4-channel RGB+Mono support with centralized dataset configuration using `load_from_disk`
- **2025.07.29**: ✅ We've released our new work [HYPIR](https://github.com/XPixelGroup/HYPIR)! It **significantly outperforms DiffBIR**🔥🔥🔥, while also being **tens of times faster**🔥🔥🔥. We welcome you to try it out.
- **2024.11.27**: ✅ Release DiffBIR v2.1, including a **new model** trained on unsplash dataset with [LLaVA]()-generated captions, more samplers, better tiled-sampling support and so on. Check [release note](https://github.com/XPixelGroup/DiffBIR/releases/tag/v2.1.0) for details.
- **2024.04.08**: ✅ Release everything about our [updated manuscript](https://arxiv.org/abs/2308.15070), including (1) a **new model** trained on subset of laion2b-en and (2) a **more readable code base**, etc. DiffBIR is now a general restoration pipeline that could handle different blind image restoration tasks with a unified generation module.
- **2023.09.19**: ✅ Add support for Apple Silicon! Check [installation_xOS.md](assets/docs/installation_xOS.md) to work with **CPU/CUDA/MPS** device!
- **2023.09.14**: ✅ Integrate a patch-based sampling strategy ([mixture-of-diffusers](https://github.com/albarji/mixture-of-diffusers)). [**Try it!**](#tiled-sampling) Here is an [example](https://imgsli.com/MjA2MDA1) with a resolution of 2396 x 1596. GPU memory usage will continue to be optimized in the future and we are looking forward to your pull requests!
- **2023.09.14**: ✅ Add support for background upsampler (DiffBIR/[RealESRGAN](https://github.com/xinntao/Real-ESRGAN)) in face enhancement! :rocket: [**Try it!**](#inference_fr)
- **2023.09.13**: :rocket: Provide online demo (DiffBIR-official) in [OpenXLab](https://openxlab.org.cn/apps/detail/linxinqi/DiffBIR-official), which integrates both general model and face model. Please have a try! [camenduru](https://github.com/camenduru) also implements an online demo, thanks for his work.:hugs:
- **2023.09.12**: ✅ Upload inference code of latent image guidance and release [real47](inputs/real47) testset.
- **2023.09.08**: ✅ Add support for restoring unaligned faces.
- **2023.09.06**: :rocket: Update [colab demo](https://colab.research.google.com/github/camenduru/DiffBIR-colab/blob/main/DiffBIR_colab.ipynb). Thanks to [camenduru](https://github.com/camenduru)!:hugs:
- **2023.08.30**: This repo is released.

## <a name="visual_results"></a>:eyes:Visual Results On Real-world Images

### Blind Image Super-Resolution

[<img src="assets/visual_results/bsr6.png" height="223px"/>](https://imgsli.com/MTk5ODI3) [<img src="assets/visual_results/bsr7.png" height="223px"/>](https://imgsli.com/MTk5ODI4) [<img src="assets/visual_results/bsr4.png" height="223px"/>](https://imgsli.com/MTk5ODI1)

### Blind Face Restoration

[<img src="assets/visual_results/whole_image1.png" height="370"/>](https://imgsli.com/MjA2MTU0) 
[<img src="assets/visual_results/whole_image2.png" height="370"/>](https://imgsli.com/MjA2MTQ4)

:star: Face and the background enhanced by DiffBIR.

### Blind Image Denoising

[<img src="assets/visual_results/bid1.png" height="215px"/>](https://imgsli.com/MjUzNzkz) [<img src="assets/visual_results/bid3.png" height="215px"/>](https://imgsli.com/MjUzNzky)
[<img src="assets/visual_results/bid2.png" height="215px"/>](https://imgsli.com/MjUzNzkx)

### 8x Blind Super-Resolution With Tiled Sampling

> I often think of Bag End. I miss my books and my arm chair, and my garden. See, that's where I belong. That's home. --- Bilbo Baggins

[<img src="assets/visual_results/tiled_sampling.png" height="480px"/>](https://imgsli.com/MjUzODE4)

## <a name="todo"></a>:climbing:TODO

- [x] Release code and pretrained models :computer:.
- [x] Update links to paper and project page :link:.
- [x] Release real47 testset :minidisc:.
- [ ] Provide webui.
- [x] Reduce the vram usage of DiffBIR :fire::fire::fire:.
- [ ] Provide HuggingFace demo :notebook:.
- [x] Add a patch-based sampling schedule :mag:.
- [x] Upload inference code of latent image guidance :page_facing_up:.
- [x] Improve the performance :superhero:.
- [x] Support MPS acceleration for MacOS users.
- [ ] DiffBIR-turbo :fire::fire::fire:.
- [x] Speed up inference, such as using fp16/bf16, torch.compile :fire::fire::fire:.
- [x] Add 4-channel RGB+Mono support :new:.

## <a name="installation"></a>:gear:Installation

```shell
# clone this repo
git clone https://github.com/XPixelGroup/DiffBIR.git
cd DiffBIR

# create environment
conda create -n diffbir python=3.10
conda activate diffbir
pip install -r requirements.txt
```

Our new code is based on pytorch 2.2.2 for the built-in support of memory-efficient attention. If you are working on a GPU that is not compatible with the latest pytorch, just downgrade pytorch to 1.13.1+cu116 and install xformers 0.0.16 as an alternative.

## <a name="quick_start"></a>:flight_departure:Quick Start

Run the following command to interact with the gradio website.

```shell
# For low-VRAM users, set captioner to ram or none
python run_gradio.py --captioner llava
```

<div align="center">
    <kbd><img src="assets/gradio.png"></img></kbd>
</div>

## <a name="pretrained_models"></a>:dna:Pretrained Models

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
RGB_DATASET_PATH = "/path/to/your/rgb_dataset"
MONO_DATASET_PATH = "/path/to/your/mono_dataset"
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
dataset/
├── train/
│   ├── gt/      # High-quality images
│   └── blur/    # Low-quality/degraded images
└── validation/ (optional)
    ├── gt/
    └── blur/
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

## Citation

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