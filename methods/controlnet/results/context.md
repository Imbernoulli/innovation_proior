# Context

## Research Question

Large text-to-image diffusion models generate high-quality images from a prompt. Spatial composition is specified through text: layout, pose, object silhouette, edge structure, depth pattern, or segmentation mask are described in words.

An alternative interface is an additional image-valued condition: an edge map, depth map, human-pose skeleton, segmentation map, normal map, scribble, or line drawing that directly specifies the spatial structure. The general text-to-image model has been trained on billions of image-text pairs, while a dataset for one specific condition type may contain only tens or hundreds of thousands of pairs.

The question is therefore: how can a pretrained text-to-image diffusion model be given a new spatial image condition, learned from far less data than the original model saw, while preserving the original model's visual quality and text alignment?

## Background

The starting point is a latent text-to-image diffusion model. A denoiser `epsilon_theta` learns to predict the noise `epsilon` added to a clean latent `z_0`, using a loss of the form `E[||epsilon - epsilon_theta(z_t, t, c_t)||_2^2]`, where `t` is the diffusion timestep and `c_t` is a text condition. Stable Diffusion performs this denoising in a compressed latent space: a `512 x 512` image corresponds to a much smaller latent grid, commonly `64 x 64`, and a U-Net with an encoder, middle block, and skip-connected decoder predicts the noise. Text is encoded separately, typically through a CLIP text encoder, and timesteps are embedded through a time encoder.

Several established ideas shape the design space. Finetuning all weights gives the new condition full access to the model. Parameter-efficient adaptation instead restricts what can change: adapters add small modules, LoRA learns low-rank weight updates, and additive or side-tuning methods keep a base network unchanged while training an auxiliary branch.

Initialization is another consideration. New modules inserted into a trained network produce activations in internal features from the start of training. Random initialization is the standard choice for ordinary networks.

## Baselines

The natural alternatives are:

- **Direct finetuning.** Continue training the pretrained diffusion model on `(image, text, condition)` triples.
- **Image-to-image translation.** Train a conditional GAN or diffusion model from a condition image to an output image.
- **Adapters.** Insert small trainable modules into a frozen model.
- **Low-rank adaptation.** Learn low-rank updates to selected weights.
- **Side or additive branches.** Train an auxiliary branch and combine it with a base model.
- **Mask/token or attention-based spatial control.** Encode spatial inputs as tokens or modify attention layers for specific grounding formats.

## Evaluation Settings

A convincing method should be tested with a large pretrained latent diffusion model such as Stable Diffusion v1.5 or v2.1, under several condition types: Canny edges, Hough or M-LSD lines, HED edges, user scribbles, human pose, semantic segmentation, depth, normal maps, and line drawings. It should handle single conditions, multiple simultaneous conditions, and prompts ranging from empty to complete or conflicting.

The data-size regime matters because the method is motivated by condition-specific datasets that are much smaller than the text-to-image pretraining set. Evaluation should therefore include small datasets below `50k` examples and larger datasets above `1m`, along with practical training-resource measurements.

Useful metrics include FID for distributional quality, CLIP text-image score, aesthetic score, condition reconstruction metrics where available, and human preference studies.

## Code Framework

The available substrate is a pretrained latent-diffusion U-Net and its native noise-prediction objective. The open implementation problem is how to construct a trainable condition path and how to connect that path into the U-Net.

```python
import torch
import torch.nn as nn

class PretrainedDiffusionUNet(nn.Module):
    # Text-to-image latent diffusion U-Net: input blocks, middle block, output blocks.
    # It predicts the noise epsilon added to z_t from timestep t and text condition c_t.
    def __init__(self): ...
    def forward(self, z_t, t, c_t): ...

class ConditionEncoder(nn.Module):
    # Map a pixel-space condition image into features compatible with the latent U-Net.
    def __init__(self):
        super().__init__()
        self.net = None
    def forward(self, c_image):
        pass

class SpatialControlBranch(nn.Module):
    # TODO: decide which pretrained weights, if any, are reused; decide which base weights update;
    # TODO: decide where condition features enter the U-Net and how the connector is initialized.
    def __init__(self, base_unet, condition_encoder):
        super().__init__()
        self.cond_enc = condition_encoder
    def forward(self, z_t, t, c_t, c_image):
        pass

def diffusion_loss(eps, eps_pred):
    return ((eps - eps_pred) ** 2).mean()

def train_step(z0, t, c_t, c_image, model, opt, alpha_bar):
    eps = torch.randn_like(z0)
    z_t = alpha_bar[t].sqrt() * z0 + (1 - alpha_bar[t]).sqrt() * eps
    eps_pred = model(z_t, t, c_t, c_image)
    loss = diffusion_loss(eps, eps_pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss
```

