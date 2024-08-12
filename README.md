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
<br><br>

**2. Vision Transformer Architecture**<br>
A vanilla vision transformer is used as a baseline model in comparison to other hybrid models, including SViT-Image, SViT-Patch, and SViT-Freq. These 3 hybrid models leverage different techniques of tokenization in combination with scattering transform.
  - ViT (Baseline)
  - SViT-Image
  - SViT-Patch
  - SViT-Freq
<br><br>

**3. Swin Transformer Architecture**<br>
A vanilla swin transformer is used as a baseline model, while the modified models with one and two-stage modules removed are implemented for experimenting.
  - Swin Transformer (Baseline)
  - Scattering Swin Transformer with 1 module removed
  - Scattering Swin Transformer with 2 modules removed
<br><br>

## How to run the code
With all necessary packages installed in the environment, these codes should be able to run smoothly. Each source code for each model is independent. The additional packages that you might need include ***kymatio*** and ***einops***. 
```
pip install kymatio
pip install einops
```
## Modify the models
- 
## Note
- Orientation used in this code is always 8
- version of each package
- running on different hardware device might result in different final output
