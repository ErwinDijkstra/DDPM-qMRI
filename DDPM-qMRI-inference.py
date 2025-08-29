# -*- coding: utf-8 -*-
"""
Inference with DDPM for Quantitative MRI Parameter Estimation
=============================================================

This script performs inference using a trained Denoising Diffusion
Probabilistic Model (DDPM) to estimate Proton Density (PD) and T1 maps
from quantitative MRI data.

It is adapted from MONAI’s diffusion model framework and serves as a
simplified implementation of the method described in:

Wang, S., Ma, H., Hernandez-Tamames, J.A., Klein, S., Poot, D.H.J. (2025). 
qMRI Diffuser: Quantitative T1 Mapping of the Brain Using a Denoising Diffusion Probabilistic Model. 
In: Mukhopadhyay, A., Oksuz, I., Engelhardt, S., Mehrof, D., Yuan, Y. (eds)
Deep Generative Models. DGM4MICCAI 2024. Lecture Notes in Computer Science, vol 15224. Springer, Cham. [https://doi.org/10.1007/978-3-031-72744-3_13].

This code is based on the MONAI Generative Models tutorial(https://github.com/Project-MONAI/GenerativeModels/tree/main/tutorials), 
which provides a reference implementation of diffusion models 

Authors: Erwin Dijkstra & Shishuai Wang
Date: 2025-03-26
"""

import os
import pickle
import torch
import matplotlib.pyplot as plt
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["MONAI_DATA_DIRECTORY"] = "C:/Users/erwin/MONAI_data"

ROOT_DIR = os.environ.get("MONAI_DATA_DIRECTORY", "C:/Users/erwin/MonaiData")
os.makedirs(ROOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"C:\Users\erwin\Desktop\DDPM_InverseProblems\models\diffusion_model.pt"
TEST_SIGNAL_PATH = r"C:\Users\erwin\Desktop\DDPM_InverseProblems\test_invivo\invivo_s15.pkl"
SAVE_PD_PATH = r"C:\Users\erwin\Desktop\DDPM_InverseProblems\Results2\invivo_s15_DDPM_PD.pkl"
SAVE_T1_PATH = r"C:\Users\erwin\Desktop\DDPM_InverseProblems\Results2\invivo_s15_DDPM_T1.pkl"


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def build_model() -> DiffusionModelUNet:
    """Construct and return the DDPM UNet model."""
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=9,
        out_channels=2,
        num_channels=(32, 64, 64),
        attention_levels=(False, False, False),
        num_res_blocks=1,
        num_head_channels=64,
    )
    model.to(DEVICE)
    return model


def load_model(model: DiffusionModelUNet, ckpt_path: str) -> DiffusionModelUNet:
    """Load model weights from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_condition(signal_path: str) -> torch.Tensor:
    """Load conditioning signal from pickle file."""
    with open(signal_path, "rb") as f:
        signal = pickle.load(f)
    condition = torch.tensor(signal["signal"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return condition


def run_inference(model, condition, num_steps: int = 1000):
    """Perform DDPM sampling with conditioning."""
    scheduler = DDPMScheduler(num_train_timesteps=num_steps)
    inferer = DiffusionInferer(scheduler)

    noise = torch.randn((1, 2, 256, 256), device=DEVICE)
    scheduler.set_timesteps(num_inference_steps=num_steps)

    with torch.amp.autocast(device_type="cuda"):
        image, _ = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=num_steps,
            conditioning=condition,
            mode="concat",  # concatenates noise and condition
        )

    # Apply post-processing transformation
    image = 0.5 * torch.log((image + 3) / (1 - image))
    return image


def plot_maps(est_pd, est_t1, vmin=0, vmax=5):
    """Plot estimated PD and T1 maps."""
    plt.style.use("default")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axs[0].imshow(est_pd, vmin=vmin, vmax=vmax, cmap="viridis")
    axs[0].set_title("Estimated PD")
    axs[0].axis("off")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(est_t1, vmin=vmin, vmax=vmax, cmap="viridis")
    axs[1].set_title("Estimated T1")
    axs[1].axis("off")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def save_results(est_pd, est_t1, save_pd_path, save_t1_path):
    """Save estimated PD and T1 maps to pickle files."""
    with open(save_pd_path, "wb") as f:
        pickle.dump(est_pd, f)
    with open(save_t1_path, "wb") as g:
        pickle.dump(est_t1, g)


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def main():
    """Main pipeline for inference and visualization."""
    # Build and load model
    model = build_model()
    model = load_model(model, MODEL_PATH)

    # Load condition
    condition = load_condition(TEST_SIGNAL_PATH)

    # Run inference
    image = run_inference(model, condition)

    # Extract results
    est_pd = image[0, 0].cpu().numpy()
    est_t1 = image[0, 1].cpu().numpy()

    # Plot
    plot_maps(est_pd, est_t1)

    # Save
    save_results(est_pd, est_t1, SAVE_PD_PATH, SAVE_T1_PATH)


if __name__ == "__main__":
    main()

