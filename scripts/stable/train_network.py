#!/usr/bin/env python3
# train_network.py - Basic LoRA training script for Stable Diffusion models

import os
import argparse
import torch
from typing import Optional

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from pathlib import Path
import random
import glob
from tqdm.auto import tqdm

# LoRA imports
from peft import (
    LoraConfig,
    get_peft_model,
)

# Set up argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA model on a folder of images")
    
    # Basic paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory containing the training images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    
    # LoRA specific
    parser.add_argument("--network_module", type=str, default="lora",
                        help="LoRA network module type")
    parser.add_argument("--network_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--network_dim", type=int, default=32,
                        help="LoRA rank (dimension)")
    
    # Training params
    parser.add_argument("--resolution", type=str, default="512,512",
                        help="Image resolution for training in format: width,height")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")

    # ------------------------------------------------------------------
    # after the existing parser.add_argument(...) block
    # just copy-paste these; they don’t have to be used later
    parser.add_argument("--output_name",        type=str, default="lora")
    parser.add_argument("--save_model_as",      type=str, default="safetensors")
    parser.add_argument("--save_every_n_epochs",type=int, default=1)
    parser.add_argument("--cache_latents",      action="store_true")
    parser.add_argument("--optimizer_type",     type=str, default="AdamW8bit")
    parser.add_argument("--xformers",           action="store_true")
    parser.add_argument("--bucket_no_upscale",  action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler",       type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps",    type=int, default=0)
    parser.add_argument("--caption_extension",  type=str, default=".txt")
    parser.add_argument("--clip_skip",          type=int, default=1)
    parser.add_argument("--save_every_n_steps", type=int, default=None)
    parser.add_argument("--no_half_vae",        action="store_true")
    parser.add_argument("--sdxl",               action="store_true")
    # ------------------------------------------------------------------
    
    args = parser.parse_args()
    
    # Process resolution
    if "," in args.resolution:
        args.resolution = [int(x) for x in args.resolution.split(",")]
    else:
        args.resolution = [int(args.resolution), int(args.resolution)]
    
    return args

# Simple dataset class
class ImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size, caption_extension=".txt"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.caption_extension = caption_extension
        
        # Get all image files
        self.image_files = []
        for ext in ["jpg", "jpeg", "png", "webp"]:
            self.image_files.extend(glob.glob(os.path.join(data_dir, f"*.{ext}")))
            self.image_files.extend(glob.glob(os.path.join(data_dir, f"*.{ext.upper()}")))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.size, resample=PIL.Image.LANCZOS)
        image = torch.from_numpy(
            (np.array(image) / 255.0).astype(np.float32).transpose(2, 0, 1)
        )
        
        # Get caption if available
        caption_path = os.path.splitext(img_path)[0] + self.caption_extension
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            # Use filename without extension as fallback
            caption = os.path.splitext(os.path.basename(img_path))[0]
            caption = caption.replace("_", " ").replace("-", " ")
        
        # Tokenize the caption
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": image,
            "input_ids": input_ids,
        }

# Main training function
def main():
    args = parse_args()
    
    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Load scheduler, tokenizer, and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Prepare unet for LoRA training
    lora_config = LoraConfig(
        r=args.network_dim,
        lora_alpha=args.network_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )
    
    # Create dataset and dataloader
    dataset = ImageDataset(
        data_dir=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    
    # Setup lr scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )
    
    # Prepare everything with accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # Move models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Use weight_dtype for vae and text_encoder
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    
    # Calculate steps per epoch for better logging
    num_steps_per_epoch = len(dataloader)
    
    # Import numpy here to avoid circular import issues
    import numpy as np
    
    # Training loop
    while global_step < args.max_train_steps:
        unet.train()
        for batch in dataloader:
            # Convert images to latent space
            with torch.no_grad():
                # Get text embeddings
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_output = text_encoder(input_ids)[0]
                
                # Get image latents
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict the noise
            model_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states=encoder_output
            ).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")
            
            # Backpropagate
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_description(f"Step {global_step}: loss {loss.detach().item():.4f}")
            
            global_step += 1
            
            # Check if we've reached max steps
            if global_step >= args.max_train_steps:
                break
    
    # Save the trained model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap the model
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # Save in .safetensors format
        try:
            from safetensors.torch import save_file
            
            state_dict = unwrapped_unet.unet.state_dict()
            safetensors_path = os.path.join(args.output_dir, f"lora_model.safetensors")
            save_file(state_dict, safetensors_path)
            print(f"Model saved to {safetensors_path}")
        except ImportError:
            print("safetensors not available, saving in .bin format")
            # Fallback to regular PyTorch save
            unwrapped_unet.save_pretrained(args.output_dir)
    
    accelerator.end_training()
    print("Training completed successfully!")

if __name__ == "__main__":
    try:
        import numpy as np  # Import numpy here to prevent issues
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 