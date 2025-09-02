"""
4-Channel ControlLDM for handling RGB + Monochromatic input
"""

from typing import Tuple, Set, List, Dict
import torch
from torch import nn

from .cldm import ControlLDM, disabled_train
from .controlnet import ControlledUnetModel, ControlNet
from .vae_4channel import AutoencoderKL4Channel
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook


class ControlLDM4Channel(ControlLDM):
    """
    4-Channel ControlLDM that handles RGB + Monochromatic input
    """

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        # Initialize parent class but we'll replace the VAE
        super().__init__(unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor)
        
        # Replace the VAE with 4-channel version
        self.vae = AutoencoderKL4Channel(**vae_cfg)
        
        print("ControlLDM4Channel initialized with 4-channel VAE")

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Load pretrained weights, handling the 4-channel VAE specially
        """
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model", 
            "clip": "cond_stage_model",
        }
        
        used = set()
        missing = set()
        
        # Load UNet and CLIP normally
        for name, module in [("unet", self.unet), ("clip", self.clip)]:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        
        # Handle VAE specially for 4-channel support
        vae_prefix = module_map["vae"]
        vae_sd = {}
        for key, value in sd.items():
            if key.startswith(vae_prefix):
                vae_key = key[len(vae_prefix) + 1:]  # Remove prefix and dot
                vae_sd[vae_key] = value
                used.add(key)
        
        if vae_sd:
            missing_vae, _ = self.vae.load_from_3channel_vae(vae_sd, strict=False)
            for key in missing_vae:
                missing.add(f"{vae_prefix}.{key}")
        
        unused = set(sd.keys()) - used
        
        # Set modules to eval mode and disable gradients
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        
        return unused, missing

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        """
        Encode 4-channel image to latent space
        
        Args:
            image: 4-channel input tensor (RGB + mono)
        """
        if image.shape[1] != 4:
            raise ValueError(f"Expected 4-channel input, got {image.shape[1]} channels")
        
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare condition for 4-channel input
        
        Args:
            cond_img: 4-channel condition image (RGB + mono)
        """
        if cond_img.shape[1] != 4:
            raise ValueError(f"Expected 4-channel condition image, got {cond_img.shape[1]} channels")
        
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,  # Normalize to [-1, 1] for all 4 channels
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )

    def debug_shapes(self, x: torch.Tensor, msg: str = ""):
        """Debug utility to print tensor shapes"""
        print(f"Debug {msg}: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.3f}, {x.max():.3f}]")