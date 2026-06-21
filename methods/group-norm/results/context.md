## Research question

Modern convolutional networks rely on a normalization layer inserted after (most)
convolutions to keep the distribution of intermediate features controlled — this is what
lets very deep networks train at all, and it speeds up and stabilizes optimization. The
dominant such layer normalizes each channel using the mean and variance computed over the
mini-batch, i.e. it uses the batch as the population over which statistics are estimated.

The question: how should a per-layer feature normalization choose the population of feature
positions over which it estimates each mean and variance? Many high-value vision tasks
(detection, instance segmentation, video) run at batch sizes of 1–2 images per device
because of the memory cost of high-resolution inputs and 3D convolutions, so any normalizer
is used across a wide range of batch sizes.

## Background

**Why normalize hidden features at all.** Normalizing the *input* data makes training
faster (LeCun et al. 1998). Inside the network, controlling the distribution of each
layer's activations keeps gradients well-scaled and lets optimization proceed through many
layers. Weight-initialization schemes (Glorot & Bengio 2010; He et al. 2015) set up good
feature distributions analytically from assumptions about the feature distribution at
initialization. An explicit normalization *layer* re-centers and re-scales features on the
fly during training.

**Early normalization layers.** Local Response Normalization (LRN) was used in AlexNet
(Krizhevsky et al. 2012) and successors: it normalizes each pixel using a small
neighborhood of channels around it, i.e. a local, per-pixel statistic.

**The general form.** A large family of normalization layers can be written in one form.
Let a feature tensor be indexed by i = (i_N, i_C, i_H, i_W) in (N, C, H, W) order — batch,
channel, height, width. Each method picks a set S_i of feature positions that share one
mean and variance, then standardizes:

    x̂_i = (x_i − μ_i) / σ_i,
    μ_i = (1/m) Σ_{k∈S_i} x_k,
    σ_i = sqrt( (1/m) Σ_{k∈S_i} (x_k − μ_i)^2 + ε ),   m = |S_i|,

with ε a small constant for numerical stability. After standardizing, a learnable
per-channel affine transform y_i = γ x̂_i + β (γ, β indexed by channel) restores
representational power that pure standardization would remove. The methods below differ
**only** in how S_i is chosen; everything else is shared.

**Batch-statistics behavior across batch sizes.** When the normalization statistics are
estimated over the batch axis, the quality of those estimates depends on how many samples
the batch provides. Measured on ResNet-50 / ImageNet, the validation error of the
batch-based layer is about 23.6, 23.7, 24.8, 27.3, 34.7 % at per-device batch sizes 32, 16,
8, 4, 2. At inference there is no batch to pool over, so population statistics are
accumulated by a running average during training and frozen for testing.

**Classical vision features.** Classical hand-designed vision features are often computed
*group-wise* and normalized *within groups*. A HOG descriptor (Dalal & Triggs 2005) is
built from spatial cells, each a histogram of gradient orientations, normalized within each
block/orientation group; SIFT (Lowe 2004) similarly normalizes orientation histograms;
GIST, VLAD, and Fisher Vectors are grouped sub-vectors (e.g. one sub-vector per cluster). A
well-accepted neuroscience model normalizes responses divisively across populations of
cells with related receptive-field and frequency tunings. As for the channels of a
convolutional layer, empirically the channels are different filters (edges, colors,
textures, frequencies) whose response distributions — means and variances — differ from one
another, and a filter and its horizontally-flipped counterpart produce similar response
distributions on natural images.

## Baselines

**Batch-based normalization (Ioffe & Szegedy 2015).** S_i = { k | k_C = i_C }: for each
channel, pool over (N, H, W) — every position in that channel across the whole batch and
all spatial locations shares one μ, σ. Add the per-channel affine y = γ x̂ + β. It eases
optimization, enables very deep networks, and the noise from random batch sampling acts as
a regularizer that helps generalization. It uses a "sufficiently large" batch (e.g. 32 per
device) and accumulates frozen running statistics for inference.

**Per-sample, all-channel normalization (Ba et al. 2016).** S_i = { k | k_N = i_N }: for
each sample, pool over all of (C, H, W). Batch-independent — same at train and test, no
running statistics — and effective for recurrent models (RNN/LSTM). All channels of a
sample share one mean and variance.

**Per-sample, per-channel normalization (Ulyanov et al. 2016).** S_i = { k | k_N = i_N,
k_C = i_C }: for each sample and each channel, pool over (H, W) only. Also batch-independent
and strong for style transfer and generative models. Each channel is normalized using only
its own spatial pixels.

**Weight normalization (Salimans & Kingma 2016).** Normalizes the *filter weights* rather
than the features, reparameterizing each weight vector by its direction and a separate
scale. Batch-independent; it controls the parameters rather than the feature distribution.

**Constrained batch statistics (Ioffe 2017).** Computes correction factors between the
current batch statistics and running population statistics, then clips those factors to
limit how far the normalized activations can drift.

**Cross-device batch statistics (Peng et al. 2018).** Computes the batch statistics jointly
across multiple devices to enlarge the effective batch.

## Evaluation settings

- **Image classification:** ImageNet (1000 classes, ~1.28M train / 50k val); ResNet-50 and
  ResNet-101 backbones; report top-1 center-crop (224×224) validation error. Standard recipe:
  He-initialized convolutions; SGD with weight decay 1e-4; 100 epochs with 10× learning-rate
  drops at 30/60/90; data augmentation per the standard GoogLeNet-style pipeline; 8 devices.
  Per-device batch sizes swept across 32, 16, 8, 4, 2 with the linear learning-rate scaling
  rule. VGG-16 used to compare against no normalization (it can train without it).
- **Object detection & instance segmentation:** COCO (train2017 / val2017); Mask R-CNN with
  C4 and FPN backbones; metrics AP, AP50, AP75 for both box and mask. Fine-tuned at batch
  size 1–2 per device from ImageNet-pretrained backbones (and also trained from scratch).
- **Video classification:** Kinetics (400 classes); Inflated-3D ResNet-50; 32- and 64-frame
  clips; batch sizes 8 and 4 clips/device; report top-1/top-5 with standard 10-clip testing.
  Normalization is extended from (H, W) to the spatial-temporal (T, H, W).

## Code framework

The pieces below already exist: a `Conv → Norm → ReLU` block abstraction, an SGD optimizer,
a cross-entropy loss, a ResNet built from such blocks, and the moment-computing primitives
of the autodiff library. The one open slot is the normalization layer itself — what set of
feature positions should share each mean/variance, and how to compute it.

```python
import torch
import torch.nn as nn


def compute_moments(x, dims):
    """Mean and (biased) variance of x reduced over `dims`, keepdim."""
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return mean, var


class FeatureNorm(nn.Module):
    """A per-layer feature normalization to be inserted after a conv.

    Standardize x with a per-position mean/variance, then apply a learnable
    per-channel affine. The open question is which feature positions share one
    mean/variance, and how to estimate it.
    """

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))   # gamma
            self.bias = nn.Parameter(torch.zeros(num_channels))    # beta
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: [N, C, H, W]
        # TODO: choose the set of positions S that share one mean/variance,
        # standardize x over it, then apply the per-channel affine.
        pass


class ConvNormReLU(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm = FeatureNorm(cout)        # the slot to be filled
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


def train_step(model, images, labels, optimizer):
    logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
