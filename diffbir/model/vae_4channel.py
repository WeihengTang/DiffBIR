"""
4-Channel VAE for handling RGB + Monochromatic input
Extends the original AutoencoderKL to handle 4 input channels
"""

import torch
import torch.nn as nn
from .vae import AutoencoderKL, Encoder, Decoder, nonlinearity, Normalize
from .distributions import DiagonalGaussianDistribution


class Encoder4Channel(Encoder):
    """
    Modified encoder that handles 4-channel input (RGB + Mono)
    """
    def __init__(self, *args, **kwargs):
        # Force in_channels to be 4
        kwargs['in_channels'] = 4
        super().__init__(*args, **kwargs)
        
        # The first conv layer needs to be modified to accept 4 channels
        # We'll replace it after parent initialization
        self.conv_in = torch.nn.Conv2d(
            4,  # 4 input channels (RGB + mono)
            self.ch, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )


class Decoder4Channel(Decoder):
    """
    Modified decoder that outputs 3-channel RGB (we don't need to output the mono channel)
    """
    def __init__(self, *args, **kwargs):
        # Force out_ch to be 3 (RGB output only)
        kwargs['out_ch'] = 3
        super().__init__(*args, **kwargs)


class AutoencoderKL4Channel(nn.Module):
    """
    4-Channel AutoencoderKL for processing RGB + Monochromatic input
    Encodes 4-channel input to latent space, decodes back to 3-channel RGB
    """

    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        
        # Create 4-channel encoder and 3-channel decoder
        encoder_config = ddconfig.copy()
        encoder_config['in_channels'] = 4
        
        decoder_config = ddconfig.copy()
        decoder_config['out_ch'] = 3  # Output RGB only
        
        self.encoder = Encoder4Channel(**encoder_config)
        self.decoder = Decoder4Channel(**decoder_config)
        
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # Debug logging
        print(f"AutoencoderKL4Channel initialized:")
        print(f"  - Encoder input channels: {self.encoder.in_channels}")
        print(f"  - Decoder output channels: {decoder_config['out_ch']}")
        print(f"  - Embed dim: {embed_dim}")

    def encode(self, x):
        """
        Encode 4-channel input (RGB + mono) to latent space
        Args:
            x: tensor of shape (batch, 4, height, width)
        """
        if x.shape[1] != 4:
            raise ValueError(f"Expected 4-channel input, got {x.shape[1]} channels")
        
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """
        Decode latent to 3-channel RGB output
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        Forward pass: 4-channel input -> latent -> 3-channel output
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
    
    def load_from_3channel_vae(self, vae_3ch_state_dict, strict=False):
        """
        Initialize from a pre-trained 3-channel VAE by expanding the first conv layer
        
        Args:
            vae_3ch_state_dict: state dict from 3-channel VAE
            strict: whether to use strict loading
        """
        # Load the state dict, but we need to handle the first conv layer specially
        current_state = self.state_dict()
        
        for key, value in vae_3ch_state_dict.items():
            if key in current_state:
                if key == 'encoder.conv_in.weight':
                    # Handle the encoder's first conv layer
                    # Original: (ch_out, 3, 3, 3)
                    # New: (ch_out, 4, 3, 3)
                    original_weight = value  # shape: (ch_out, 3, 3, 3)
                    ch_out = original_weight.shape[0]
                    
                    # Initialize the 4-channel weight
                    new_weight = torch.zeros((ch_out, 4, 3, 3), dtype=original_weight.dtype, device=original_weight.device)
                    
                    # Copy RGB channels
                    new_weight[:, :3, :, :] = original_weight
                    
                    # Initialize mono channel (channel 3) with average of RGB channels  
                    new_weight[:, 3, :, :] = original_weight.mean(dim=1)
                    
                    current_state[key] = new_weight
                    print(f"Expanded encoder.conv_in.weight from {original_weight.shape} to {new_weight.shape}")
                    
                else:
                    # For all other layers, copy directly
                    if current_state[key].shape == value.shape:
                        current_state[key] = value
                    else:
                        print(f"Skipping {key} due to shape mismatch: {current_state[key].shape} vs {value.shape}")
        
        # Load the modified state dict
        missing_keys, unexpected_keys = self.load_state_dict(current_state, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading 4-channel VAE: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading 4-channel VAE: {unexpected_keys}")
        
        return missing_keys, unexpected_keys