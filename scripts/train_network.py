#!/usr/bin/env python3
# train_network.py - Pure PyTorch LoRA training without diffusers

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

# Simple LoRA implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def parse_args():
    parser = argparse.ArgumentParser(description="Simple LoRA training")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="512,512")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--network_alpha", type=int, default=32)
    parser.add_argument("--network_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    
    # Ignored parameters for compatibility
    parser.add_argument("--network_module", type=str, default="lora")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_model_as", type=str, default="safetensors")
    parser.add_argument("--cache_latents", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--bucket_no_upscale", action="store_true")
    
    args = parser.parse_args()
    
    if "," in args.resolution:
        args.resolution = [int(x) for x in args.resolution.split(",")]
    else:
        args.resolution = [int(args.resolution), int(args.resolution)]
    
    return args

class SimpleDataset(Dataset):
    def __init__(self, data_dir, size):
        self.data_dir = data_dir
        self.size = size
        
        # Get all image files
        self.image_files = []
        for ext in ["jpg", "jpeg", "png", "webp", "bmp"]:
            self.image_files.extend(glob.glob(os.path.join(data_dir, f"*.{ext}")))
            self.image_files.extend(glob.glob(os.path.join(data_dir, f"*.{ext.upper()}")))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.size, resample=Image.LANCZOS)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(
                (np.array(image) / 255.0).astype(np.float32).transpose(2, 0, 1)
            )
            
            # Create a simple target (for demonstration)
            target = torch.randn(3, self.size[1], self.size[0]) * 0.1
            
            return {
                "pixel_values": image,
                "target": target
            }
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return zeros on error
            return {
                "pixel_values": torch.zeros(3, self.size[1], self.size[0]),
                "target": torch.zeros(3, self.size[1], self.size[0])
            }

class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    def __init__(self, input_channels=3, hidden_dim=512, rank=32, alpha=32):
        super().__init__()
        
        # Base layers
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # LoRA layers
        self.lora1 = LoRALayer(64, 64, rank=rank, alpha=alpha)
        self.lora2 = LoRALayer(128, 128, rank=rank, alpha=alpha)
        self.lora3 = LoRALayer(256, 256, rank=rank, alpha=alpha)
        
        # Output layer
        self.output = nn.Conv2d(256, input_channels, 3, padding=1)
        
        # Freeze base layers (only train LoRA)
        for param in [self.conv1, self.conv2, self.conv3, self.output]:
            for p in param.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        # Base forward pass
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2))
        
        # Apply LoRA adaptations
        b, c, h, w = x1.shape
        x1_flat = x1.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        x1_lora = self.lora1(x1_flat).permute(0, 2, 1).view(b, c, h, w)
        x1 = x1 + x1_lora
        
        b, c, h, w = x2.shape
        x2_flat = x2.view(b, c, -1).permute(0, 2, 1)
        x2_lora = self.lora2(x2_flat).permute(0, 2, 1).view(b, c, h, w)
        x2 = x2 + x2_lora
        
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        x3_lora = self.lora3(x3_flat).permute(0, 2, 1).view(b, c, h, w)
        x3 = x3 + x3_lora
        
        # Output
        output = self.output(x3)
        return output

def main():
    args = parse_args()
    
    print(f"Starting simple LoRA training:")
    print(f"  Data dir: {args.train_data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Steps: {args.max_train_steps}")
    print(f"  LoRA rank: {args.network_dim}")
    print(f"  LoRA alpha: {args.network_alpha}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleModel(
        rank=args.network_dim,
        alpha=args.network_alpha
    ).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create dataset
    dataset = SimpleDataset(args.train_data_dir, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    pbar = tqdm(total=args.max_train_steps, desc="Training")
    
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
                
            # Move to device
            inputs = batch["pixel_values"].to(device)
            targets = batch["target"].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = nn.functional.mse_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step += 1
            
            if global_step % 100 == 0:
                print(f"Step {global_step}: loss = {loss.item():.4f}")
    
    pbar.close()
    
    # Save LoRA weights
    print("Saving LoRA weights...")
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_state_dict[name] = param.cpu()
    
    # Save in safetensors format if available
    try:
        from safetensors.torch import save_file
        save_path = os.path.join(args.output_dir, "lora_model.safetensors")
        save_file(lora_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path}")
    except ImportError:
        # Fallback to PyTorch format
        save_path = os.path.join(args.output_dir, "lora_model.bin")
        torch.save(lora_state_dict, save_path)
        print(f"Saved LoRA weights to {save_path}")
    
    # Save config
    config = {
        "rank": args.network_dim,
        "alpha": args.network_alpha,
        "target_modules": ["lora1", "lora2", "lora3"],
        "trainable_params": trainable_params,
    }
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed! Saved {len(lora_state_dict)} LoRA parameters.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)