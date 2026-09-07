# Monet-Inspired Image Generation with CycleGAN

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-5C7C6B)

An image-to-image translation project that transforms real-world photographs into Monet-inspired artwork using a Cycle-Consistent Generative Adversarial Network (CycleGAN).

The model learns from unpaired collections of photographs and Claude Monet paintings. This allows it to transfer artistic characteristics such as color, texture, and brushstroke patterns without requiring matching image pairs.

## Project Overview

Traditional supervised image translation requires paired examples of the same scene in two different styles. These pairs are difficult to obtain for artistic style transfer.

CycleGAN solves this problem by learning mappings between two independent image domains:

- Domain A: real-world photographs
- Domain B: Claude Monet paintings

Two generators translate images in both directions, while discriminators evaluate whether the generated images resemble authentic samples from each domain.

## Objectives

- Transform real photographs into Monet-inspired images.
- Preserve the structural content of the original photograph.
- Train the model using unpaired image datasets.
- Monitor image quality using Fréchet Inception Distance (FID).
- Produce reusable generator models and generated image outputs.

## Dataset

This project uses the dataset from Kaggle’s  
[“I’m Something of a Painter Myself” competition](https://www.kaggle.com/competitions/gan-getting-started).

The dataset contains:

- 300 Claude Monet paintings
- 7,028 real-world photographs
- Images stored as JPEG and TFRecord files
- RGB images resized to `256 × 256`

## Model Architecture

The implementation uses four main neural networks:

1. Monet Generator  
   Translates real photographs into Monet-inspired images.

2. Photo Generator  
   Reconstructs Monet-style images back into photographs.

3. Monet Discriminator  
   Distinguishes real Monet paintings from generated paintings.

4. Photo Discriminator  
   Distinguishes real photographs from reconstructed photographs.

The generators use an encoder-decoder architecture with skip connections:

```text
Input image
    ↓
Downsampling encoder
    ↓
Latent representation
    ↓
Upsampling decoder
    ↓
Generated image
```

Skip connections help preserve spatial information while the network applies the target visual style.

## Training Objectives

The model combines several loss functions:

### Adversarial Loss

Encourages each generator to produce images that resemble the target domain.

### Cycle-Consistency Loss

Ensures that translating an image to another domain and back preserves its original content.

```text
Photo → Monet → Reconstructed Photo
Monet → Photo → Reconstructed Monet
```

### Identity Loss

Helps preserve important color and structural characteristics when an image already belongs to the target domain.

### Differentiable Augmentation

Translation and cutout augmentations are applied during training to improve discriminator robustness and reduce overfitting.

## Training Configuration

| Configuration | Value |
|---|---:|
| Image resolution | 256 × 256 RGB |
| Epochs | 50 |
| Steps per epoch | 100 |
| Base batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam beta₁ | 0.5 |
| Cycle-consistency weight | 10 |
| Identity weight | 0.5 |
| FID evaluation interval | Every 5 epochs |
| Framework | TensorFlow / Keras |

The training pipeline supports TPU and GPU execution. Batch sizes are automatically adjusted according to the number of available accelerator replicas.

## Evaluation

Generated images are evaluated using an InceptionV3-based Fréchet Inception Distance calculation.

FID compares feature distributions between real Monet paintings and generated Monet-style images. A lower score generally indicates that the generated distribution is closer to the real painting distribution.

During training, the project:

- Calculates FID every five epochs.
- Generates visual comparison samples.
- Saves periodic model checkpoints.
- Tracks generator, discriminator, cycle, and identity losses.

## Workflow

```text
Load TFRecord datasets
        ↓
Decode and normalize images
        ↓
Create photograph and Monet domains
        ↓
Train both generators and discriminators
        ↓
Apply adversarial, cycle, and identity losses
        ↓
Generate visual samples
        ↓
Evaluate with FID
        ↓
Save checkpoints and final generators
        ↓
Generate Monet-inspired images
```

## Project Results

The model was trained for 50 epochs with periodic qualitative and FID-based evaluation.

The completed notebook includes:

- Dataset exploration and preprocessing
- Generator and discriminator implementation
- CycleGAN training logic
- Differentiable augmentation
- Loss monitoring
- Generated image comparisons
- Periodic FID evaluation
- Model checkpointing
- Final image-generation pipeline

The project focuses on documenting the complete machine-learning workflow rather than presenting an unsupported final accuracy metric.

## Repository Contents

```text
CycleGAN-Monet-Inspired-Image/
├── Monet-Inspired Image Generation.ipynb
├── Report of Monet-Inspired Image Generation.pdf
└── README.md
```

## Project Files

- [View the Jupyter Notebook](./Monet-Inspired%20Image%20Generation.ipynb)
- [Read the Project Report](./Report%20of%20Monet-Inspired%20Image%20Generation.pdf)
- [Open the Notebook in NBViewer](https://nbviewer.org/github/rendragonnn/CycleGAN-Monet-Inspired-Image/blob/main/Monet-Inspired%20Image%20Generation.ipynb)

Main dependencies:

```text
Python
TensorFlow
Keras
TensorFlow Probability
NumPy
Matplotlib
Pillow
tqdm
Kaggle Datasets
```

Training a CycleGAN is computationally expensive. A GPU or TPU environment is strongly recommended.

## Key Learnings

This project provided practical experience with:

- Generative adversarial networks
- Unpaired image-to-image translation
- TensorFlow data pipelines
- Generator and discriminator architectures
- Distributed accelerator strategies
- GAN training stability
- Image-quality evaluation with FID
- Model checkpointing and inference workflows

## References

- [Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Original CycleGAN Project](https://junyanz.github.io/CycleGAN/)
- [Kaggle: I’m Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started)

