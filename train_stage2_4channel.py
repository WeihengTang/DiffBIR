import os
from argparse import ArgumentParser
import copy
import logging

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from diffbir.model import Diffusion
from diffbir.model.cldm_4channel import ControlLDM4Channel
from diffbir.model.swinir import SwinIR
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")
        
        # Setup file logging
        file_handler = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Create model:
    cldm: ControlLDM4Channel = instantiate_from_config(cfg.model.cldm)
    
    # Load SD checkpoint
    sd_path = cfg.train.sd_path
    if not os.path.exists(sd_path):
        raise FileNotFoundError(f"SD checkpoint not found at {sd_path}. Please download it first.")
    
    logger.info(f"Loading SD checkpoint from {sd_path}")
    sd = torch.load(sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        logger.info(f"Loaded pretrained SD weight from {sd_path}")
        logger.info(f"Unused weights: {len(unused)} keys")
        logger.info(f"Missing weights: {len(missing)} keys")
        if missing:
            logger.warning(f"Missing weights: {list(missing)[:10]}...")  # Log first 10
        if unused:
            logger.info(f"Unused weights: {list(unused)[:10]}...")  # Log first 10

    # Handle resume or initialize ControlNet
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            logger.info(f"Resumed ControlNet from {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            logger.info("Initialized ControlNet from UNet")
            logger.info(f"Weights with new zeros: {len(init_with_new_zero)} keys")
            logger.info(f"Weights from scratch: {len(init_with_scratch)} keys")

    # Create SwinIR
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    swinir_path = cfg.train.swinir_path
    
    # For now, create a dummy 4-channel SwinIR checkpoint if it doesn't exist
    if not os.path.exists(swinir_path):
        logger.warning(f"SwinIR checkpoint not found at {swinir_path}")
        logger.info("Creating dummy 4-channel SwinIR checkpoint...")
        # You would need to train a 4-channel SwinIR or adapt an existing 3-channel one
        # For now, we'll use the 4-channel SwinIR with random initialization
        os.makedirs(os.path.dirname(swinir_path), exist_ok=True)
        torch.save(swinir.state_dict(), swinir_path)
        logger.info(f"Created dummy SwinIR checkpoint at {swinir_path}")
    
    sd = torch.load(swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        logger.info(f"Loaded SwinIR from {swinir_path}")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)
    logger.info(f"Optimizer setup with learning rate: {cfg.train.learning_rate}")

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset loaded with {len(dataset):,} samples")
        logger.info(f"Batch size: {cfg.train.batch_size}, Num workers: {cfg.train.num_workers}")
        
        # Test loading one sample for debugging
        try:
            logger.info("Testing sample loading...")
            sample_gt, sample_lq, sample_prompt = dataset[0]
            logger.info(f"Sample GT shape: {sample_gt.shape}, dtype: {sample_gt.dtype}")
            logger.info(f"Sample GT range: [{sample_gt.min():.3f}, {sample_gt.max():.3f}]")
            logger.info(f"Sample LQ shape: {sample_lq.shape}, dtype: {sample_lq.dtype}")
            logger.info(f"Sample LQ range: [{sample_lq.min():.3f}, {sample_lq.max():.3f}]")
            logger.info(f"Sample prompt: '{sample_prompt}'")
            
            # Test channel separation
            logger.info(f"GT RGB channels range: [{sample_gt[..., :3].min():.3f}, {sample_gt[..., :3].max():.3f}]")
            logger.info(f"GT Mono channel range: [{sample_gt[..., 3].min():.3f}, {sample_gt[..., 3].max():.3f}]")
            logger.info(f"LQ RGB channels range: [{sample_lq[..., :3].min():.3f}, {sample_lq[..., :3].max():.3f}]")
            logger.info(f"LQ Mono channel range: [{sample_lq[..., 3].min():.3f}, {sample_lq[..., 3].max():.3f}]")
            
        except Exception as e:
            logger.error(f"Error loading sample: {e}")
            raise

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM4Channel = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        logger.info(f"Training for {max_steps} steps...")
        logger.info(f"Noise augmentation timestep: {noise_aug_timestep}")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt = batch
            
            # Debug shapes
            if accelerator.is_main_process and global_step % 100 == 0:
                logger.info(f"Step {global_step} - Batch GT shape: {gt.shape}, LQ shape: {lq.shape}")
                logger.info(f"Step {global_step} - GT range: [{gt.min():.3f}, {gt.max():.3f}]")
                logger.info(f"Step {global_step} - LQ range: [{lq.min():.3f}, {lq.max():.3f}]")
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # Additional debug after rearrange
            if accelerator.is_main_process and global_step % 100 == 0:
                logger.info(f"Step {global_step} - After rearrange GT shape: {gt.shape}, LQ shape: {lq.shape}")
                logger.info(f"Step {global_step} - GT channels: RGB=[{gt[:, :3].min():.3f}, {gt[:, :3].max():.3f}], Mono=[{gt[:, 3].min():.3f}, {gt[:, 3].max():.3f}]")
                logger.info(f"Step {global_step} - LQ channels: RGB=[{lq[:, :3].min():.3f}, {lq[:, :3].max():.3f}], Mono=[{lq[:, 3].min():.3f}, {lq[:, 3].max():.3f}]")

            with torch.no_grad():
                # VAE encode
                try:
                    z_0 = pure_cldm.vae_encode(gt)
                    logger.info(f"Step {global_step} - VAE encoded GT to latent shape: {z_0.shape}")
                except Exception as e:
                    logger.error(f"Error in VAE encode: {e}")
                    raise
                
                # SwinIR clean
                try:
                    clean = swinir(lq)
                    logger.info(f"Step {global_step} - SwinIR output shape: {clean.shape}")
                    logger.info(f"Step {global_step} - SwinIR output range: [{clean.min():.3f}, {clean.max():.3f}]")
                except Exception as e:
                    logger.error(f"Error in SwinIR: {e}")
                    raise
                
                # Prepare condition
                try:
                    cond = pure_cldm.prepare_condition(clean, prompt)
                    logger.info(f"Step {global_step} - Condition prepared, c_img shape: {cond['c_img'].shape}")
                except Exception as e:
                    logger.error(f"Error in prepare_condition: {e}")
                    raise
                
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)
                    logger.info(f"Step {global_step} - Average loss: {avg_loss:.6f}")

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Checkpoint saved: {ckpt_path}")

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        # Convert to 3-channel for visualization (VAE decoder outputs 3-channel)
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt_rgb", (log_gt[:, :3] + 1) / 2),  # GT RGB channels
                            ("image/gt_mono", log_gt[:, 3:4].repeat(1, 3, 1, 1)),  # GT Mono as grayscale
                            ("image/lq_rgb", log_lq[:, :3]),  # LQ RGB channels
                            ("image/lq_mono", log_lq[:, 3:4].repeat(1, 3, 1, 1)),  # LQ Mono as grayscale
                            ("image/condition_rgb", log_clean[:, :3]),  # Clean RGB
                            ("image/condition_mono", log_clean[:, 3:4].repeat(1, 3, 1, 1)),  # Clean mono
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)
            logger.info(f"Epoch {epoch} completed - Average loss: {avg_epoch_loss:.6f}")

    if accelerator.is_main_process:
        logger.info("Training completed!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)