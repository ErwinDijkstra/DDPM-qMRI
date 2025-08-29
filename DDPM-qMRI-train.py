# -*- coding: utf-8 -*-
"""
Training DDPM for Quantitative MRI Parameter Estimation
=======================================================

This script trains a Denoising Diffusion Probabilistic Model (DDPM)
using MONAI and PyTorch for quantitative MRI (PD and T1 mapping).

It is adapted from MONAI’s diffusion model framework and serves as a
simplified implementation of the method described in:

Wang, S., Ma, H., Hernandez-Tamames, J.A., Klein, S., Poot, D.H.J. (2025). 
qMRI Diffuser: Quantitative T1 Mapping of the Brain Using a Denoising Diffusion Probabilistic Model. 
In: Mukhopadhyay, A., Oksuz, I., Engelhardt, S., Mehrof, D., Yuan, Y. (eds)
Deep Generative Models. DGM4MICCAI 2024. Lecture Notes in Computer Science, vol 15224. Springer, Cham. (https://doi.org/10.1007/978-3-031-72744-3_13).

This code is based on the MONAI Generative Models tutorial(https://github.com/Project-MONAI/GenerativeModels/tree/main/tutorials), 
which provides a reference implementation of diffusion models 

Authors: Erwin Dijkstra & Shishuai Wang
Date: 2025-03-26
"""

import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from dataset_qmri_add_noise import IPIIDataset  # <-- custom dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = ""
MODEL_SAVE_PATH = ""

TOTAL_EPOCHS = 2000
BATCH_SIZE = 4
LR = 2.5e-5
NUM_TIMESTEPS = 1000

# Training Pipeline
def build_model() -> DiffusionModelUNet:
    """Construct the DDPM UNet model."""
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=9,   # conditioning channels
        out_channels=2,  # output PD and T1 maps
        num_channels=(32, 64, 64),
        attention_levels=(False, False, False),
        num_res_blocks=1,
        num_head_channels=64,
    )
    return model.to(DEVICE)


def train():
    """Main training loop for DDPM."""
    # Dataset and loader
    train_ds = IPIIDataset(DATA_PATH)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, persistent_workers=True
    )

    # Model, optimizer, scheduler, inferer
    model = build_model()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    inferer = DiffusionInferer(scheduler)
    scaler = GradScaler()

    epoch_loss_list = []

    # Training loop
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0.0

        print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS}")
        pbar = tqdm(train_loader, desc="Training", leave=False)

        for i, batch in enumerate(pbar):
            signal = batch[0].to(DEVICE)  # conditioning signal
            qmaps = batch[1].to(DEVICE)   # ground truth qMRI maps

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                noise = torch.randn_like(qmaps).to(DEVICE)

                # Sample random diffusion steps
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (qmaps.shape[0],), device=qmaps.device
                ).long()

                # Predict noise
                noise_pred = inferer(
                    inputs=qmaps,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps,
                    condition=signal,
                    mode="concat"
                )

                # Loss = MSE between predicted and true noise
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        epoch_loss_list.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            ckpt_path = MODEL_SAVE_PATH.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    # Save final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Final model saved at {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()



