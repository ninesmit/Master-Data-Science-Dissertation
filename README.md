# CNN and Transformer-based architecture with scattering transform
This repository contains source codes for all models implemented in my dissertation for my Master Degree.

## Model Explanation
There are 3 architectures used in this project, including a CNN-based model, vision transformer, and swin transformer.

**1. CNN-based Architecture**<br>
This includes both traditional CNN and other existing architectures. All of them are implemented with and without scattering transform. For other pre-trained architectures, we try replacing different numbers of early layers for experimenting as seen in sub-folders. 
  - CNN
  - DenseNet121
  - EfficienNetB3
  - ResNet18
**2. Vision Transformer Architecture**<br>
A vanilla vision transformer is used as a baseline model in comparison to other hybrid models, including SViT-Image, SViT-Patch, and SViT-Freq. These 3 hybrid models leverage different techniques of tokenization in combination with scattering transform.
  - ViT (Baseline)
  - SViT-Image
  - SViT-Patch
  - SViT-Freq  
-requirements to run the code
