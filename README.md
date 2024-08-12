# CNN and Transformer-based architecture with scattering transform
This repository contains source codes for all models implemented in my dissertation for my Master's Degree.

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
A vanilla swin transformer is used as a baseline model, while the modified models with one- and two-stage modules removed are implemented for experimentation.
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
## Modifying the Models
Some variables might be worth understanding for modifying the models.<br>

For all architectures, the 'mode' variable defines the number of scale parameters used for scattering transform. The code provided is set to accept the number from 1 to 3. In case a higher number of the scale parameters is needed, some modifications are required. 
```
# Set the J in the function to the number you desire
# Set the K according to the J set above
mode == 1:
    scattering = Scattering2D(J=1, shape=(image_size, image_size))
    K = 17*3
mode == 2:
    scattering = Scattering2D(J=2, shape=(image_size, image_size))
    K = 81*3
....
```
For the vision transformer, there are 2 key variables, including 'depth' and 'head'. The 'depth' variable defines the number of encoder layers used in the model, while the variable 'head' determines the number of attention heads in each encoder layer.
```
depth = 5
head = 5
```
For the swin transformer, similarly to the vision transformer, the 'layers' variable controls the number of swin blocks in each stage, from stage 1 to stage 4. The 'heads' variable defines the number of attention heads in each stage, from 1 to 4.
```
layers = (2,2,6,2)
heads = (3,6,12,24)
```
## Note
- The orientation parameter of the scattering transform is set to 8 throughout this project.
- Running codes on different hardware devices might result in different final outputs.
