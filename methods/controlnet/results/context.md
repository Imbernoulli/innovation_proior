# Context

## Research question

Large text-to-image diffusion models produce visually stunning images from a text prompt, but text is
a weak handle on *spatial* composition: precisely specifying a layout, a human pose, an object shape,
or an edge structure through words alone is hard, and matching a specific mental image usually takes
many cycles of editing the prompt, inspecting the result, and re-editing. The natural fix is to let a
user supply an additional *image* that directly specifies the desired composition — an edge map, a
depth map, a pose skeleton, a segmentation map, surface normals, a scribble — and condition the
generation on it.

Learning such a conditional control end-to-end is the hard part, and the difficulty is a data
mismatch. The pretrained text-to-image model was trained on billions of image-text pairs; the largest
datasets for a *specific* spatial condition (depth, pose, normals) are around 100k examples — tens of
thousands of times smaller. Directly finetuning, or continuing to train, the huge pretrained model on
such a small condition-specific dataset risks **overfitting** to the small set and **catastrophic
forgetting** of the billions-of-images prior that made the model good in the first place. The precise
question: **how can a new spatial conditioning control be added to a large, frozen, pretrained
text-to-image diffusion model — learned from a small condition-specific dataset — without degrading or
forgetting the pretrained model's quality and capabilities?** A solution would need to reuse the
pretrained model as a strong backbone (so it can learn from little data), keep the original model
intact (so nothing is forgotten), and introduce the new control in a way that does not inject harmful
noise into the model's deep features at the start of training, when the new branch is still random.

## Background

The field state (early text-to-image diffusion era): large latent-diffusion text-to-image models are
the state of the art, and a large literature on *finetuning* pretrained networks without destroying
them is directly relevant. The load-bearing concepts:

- **Latent text-to-image diffusion (the model being controlled).** A diffusion model learns to reverse
  a gradual noising process: a denoiser `ε_θ` is trained to predict the noise added to a noisy image,
  minimizing `E[‖ε − ε_θ(z_t, t, c_t)‖²]` over noise levels `t`. Latent diffusion (Rombach et al.,
  2021) runs this in a compressed *latent* space — a VQ-GAN-like encoder (Esser et al., 2021) maps a
  `512×512` pixel image to a `64×64` latent — which stabilizes and cheapens training. The architecture
  is a U-Net (Ronneberger et al., 2015): an encoder, a middle block, and a skip-connected decoder, with
  text prompts encoded by a CLIP text encoder (Radford et al., 2021) and timesteps by a positional time
  encoder. Stable Diffusion is a large-scale instance.

- **Catastrophic forgetting and overfitting under finetuning.** Continuing to train a large model on a
  small new dataset tends to overfit and to overwrite the broad prior — the model "forgets" what it
  knew. The literature shows this can be mitigated by *restricting* the trainable parameters: limiting
  their number or rank, or freezing the original weights entirely.

- **Weight initialization, and zero initialization specifically.** How a layer's weights are
  initialized strongly affects training. Gaussian initialization is the usual safe default, and is
  generally considered *less risky* than initializing to zero (zero-initialized layers in a normal
  network suffer symmetry problems and can fail to learn). But scaling the initial weight of a
  convolution toward — or to — zero has been used deliberately in diffusion training (Nichol et al.,
  2021, whose "zero_module" zeroes a layer's weights as an extreme case) and in GAN training
  (ProGAN/StyleGAN, Noise2Noise) to control how a new component enters the network.

## Baselines

The prior finetuning/control strategies a new method would be measured against and reacts to:

- **Direct finetuning / continued training.** Simply keep training the pretrained model (optionally all
  weights) on the new condition. *Gap:* overfitting, mode collapse, and catastrophic forgetting on
  small condition datasets.

- **HyperNetwork (Ha et al., 2017).** Train a small network to *predict the weights* of a larger one;
  applied to restyle Stable Diffusion's outputs. *Gap:* changes global behavior (e.g. style), not
  designed to inject a precise spatial control while fully preserving the base model.

- **Adapters (Houlsby et al., 2019; ViT-Adapter; and the concurrent T2I-Adapter, Mou et al., 2023).**
  Freeze the pretrained model and insert small new module layers to customize it for a new task;
  T2I-Adapter adapts Stable Diffusion to external conditions. *Gap:* the added adapter is a small,
  shallow module — limited capacity for in-the-wild conditioning images with complex shapes and diverse
  high-level semantics.

- **Additive learning / Side-Tuning (Zhang et al., 2020).** Freeze the original model and add a small
  side branch, linearly blending the frozen model's output with the side network's output via a
  *predefined* blending-weight schedule. *Gap:* the blend schedule is hand-set rather than learned to
  start from a no-op, and the side branch is not a reuse of the pretrained backbone.

- **LoRA (Hu et al., 2021).** Learn low-rank offsets to the weights, exploiting that adaptation lives in
  a low intrinsic-dimensional subspace; prevents forgetting by limiting rank. *Gap:* a low-rank weight
  delta is a strong restriction; a complex spatial control may need a deeper, more customized branch.

- **Image-to-image translation (pix2pix, Isola et al., 2017; CycleGAN; Palette; PITI).** Conditional
  GANs/diffusion that learn a condition→image mapping. *Gap:* trained from scratch (or with a generic
  pretraining), so they do not reuse a billions-of-images text-to-image prior, and need more data.

- **Spatial control of diffusion (MakeAScene; SpaText; GLIGEN, Li et al., 2023).** Encode masks into
  tokens or learn new parameters in the attention layers for grounded/spatial generation. *Gap:*
  typically condition-type-specific and modify the base model's internals rather than wrapping a frozen
  model with a protective, reusable control branch.

## Evaluation settings

The benchmarks, conditions, metrics, and protocol that form the natural yardstick:

- **Base model.** A large pretrained latent text-to-image diffusion model (Stable Diffusion v1.5 /
  v2.1, same U-Net), held fixed.

- **Conditioning types.** A wide set of spatial conditions, each from an off-the-shelf detector:
  Canny edges, Hough / M-LSD lines, HED soft edges, user scribbles, human pose keypoints, ADE20k
  semantic segmentation, depth maps, surface normals, and cartoon line drawings — tested with a single
  condition, with multiple composed conditions, and with or without text prompts.

- **Dataset-size regimes.** Robustness is judged across small (`<50k`) to large (`>1m`) condition
  datasets, and feasibility on modest hardware (a single consumer GPU).

- **Metrics.** Fréchet Inception Distance (FID) over generated `512×512` image sets to measure
  distribution distance; CLIP text-image score and CLIP aesthetic score; human user studies; and
  ablations replacing the proposed connection (e.g. a Gaussian-initialized convolution in its place, or
  a single-convolution branch instead of a full copied backbone).

## Code framework

The available substrate is a frozen pretrained latent-diffusion U-Net (encoder, middle, skip-connected
decoder) with its native noise-prediction training loss, plus the standard finetuning toolbox. What is
missing is the *architecture* of the added control branch: how to take the pretrained encoder as a
backbone for a new conditioning input, and how to connect that branch back into the frozen model so
that training starts as a no-op and grows the control without corrupting the pretrained features.

```python
import torch
import torch.nn as nn

class PretrainedDiffusionUNet(nn.Module):
    # frozen text-to-image latent diffusion U-Net: encoder, middle, skip-connected decoder.
    # predicts the noise eps added to a noisy latent z_t given (t, text c_t).
    def __init__(self): ...
    def forward(self, z_t, t, c_t): ...

base = PretrainedDiffusionUNet()
for p in base.parameters():
    p.requires_grad = False          # locked: preserve the prior, no gradient/memory in this branch

class ConditionEncoder(nn.Module):
    # map a 512x512 conditioning image (edge/depth/pose/...) to a latent-resolution feature
    def __init__(self):
        super().__init__()
        self.net = None              # small conv stack, 512x512 -> 64x64
    def forward(self, c_image):
        pass

class ControlBranch(nn.Module):
    # TODO: reuse the pretrained ENCODER as a trainable backbone for the conditioning input,
    #       and connect its outputs back into the frozen base so that at init the whole model
    #       is unchanged (the added control starts as a no-op) and grows without harmful noise.
    def __init__(self, base_encoder, condition_encoder):
        super().__init__()
        self.cond_enc = condition_encoder
        # TODO: trainable copy of base_encoder; the connection layers back into the base
    def forward(self, z_t, t, c_t, c_image):
        pass

def diffusion_loss(eps, eps_pred):
    return ((eps - eps_pred) ** 2).mean()   # native noise-prediction objective, reused unchanged

def train_step(z0, t, c_t, c_image, opt):
    # add noise to z0 -> z_t; predict eps with base + control branch; L2 on the noise
    pass
```

This harness has a frozen diffusion model and its native loss, but the control branch — how to reuse
the encoder as a backbone and how to wire it back in so that training begins as a no-op and the new
control grows safely — is the open problem.
