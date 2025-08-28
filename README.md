# DDPM for Quantitative MRI Parameter Estimation

This repository contains code for training and inference of **Denoising Diffusion Probabilistic Models (DDPMs)** applied to quantitative MRI (qMRI), focusing on Proton Density (PD) and T1 parameter mapping.

The implementation is based on **MONAI’s diffusion model framework** and adapts the workflow for inverse problems in MRI.

---

## Reference & Acknowledgment

This project is a **simplified version** of the following work:

> Sun, L., et al. *Diffusion Models for Inverse Problems in Quantitative MRI.*  
> In: Medical Imaging with Deep Learning (MIDL 2024), Lecture Notes in Computer Science, vol 15213.  
> Springer, Cham. https://doi.org/10.1007/978-3-031-72744-3_13

We also acknowledge the use of [MONAI](https://monai.io) as the core framework for building and training diffusion models.

---

## Requirements

To run this project, you need the following Python packages:

- **torch**: core PyTorch library (ensure correct CUDA version for GPU)  
- **torchvision**: PyTorch vision utilities  
- **monai-weekly[tqdm]**: MONAI framework including generative modules  
- **numpy**: numerical operations  
- **scipy**: scientific computing  
- **matplotlib**: visualization of PD/T1 maps  
- **tqdm**: progress bars  
