# DDPM for Quantitative MRI Parameter Estimation

This repository provides code for training and inference of **Denoising Diffusion Probabilistic Models (DDPMs)** applied to quantitative MRI (qMRI), with a focus on **Proton Density (PD)** and **T1 parameter mapping**.

The implementation leverages the **[MONAI diffusion framework](https://monai.io/)** and adapts it for solving **inverse problems in MRI reconstruction and parameter estimation**.

---

## Reference & Acknowledgment  

This project is a **simplified implementation** inspired by the following work:  

> Wang, S., Ma, H., Hernandez-Tamames, J.A., Klein, S., Poot, D.H.J. (2025). *qMRI Diffuser: Quantitative T1 Mapping of the Brain Using a Denoising Diffusion Probabilistic Model*. In: Mukhopadhyay, A., Oksuz, I., Engelhardt, S., Mehrof, D., Yuan, Y. (eds) Deep Generative Models. DGM4MICCAI 2024. Lecture Notes in Computer Science, vol 15224. Springer, Cham. [https://doi.org/10.1007/978-3-031-72744-3_13](https://doi.org/10.1007/978-3-031-72744-3_13)  

This code is **based on the MONAI Generative Models [tutorial](https://github.com/Project-MONAI/GenerativeModels/tree/main/tutorials)**, which provides a reference implementation of diffusion models 

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
