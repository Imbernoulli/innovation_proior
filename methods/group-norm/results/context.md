## Research question

Modern convolutional networks rely on a normalization layer inserted after (most)
convolutions to keep the distribution of intermediate features controlled — this is what
lets very deep networks train at all, and it speeds up and stabilizes optimization. The
dominant such layer normalizes each channel using the mean and variance computed over the
mini-batch. That single design choice — using the batch as the population over which
statistics are estimated — is the source of a cluster of problems.

The precise problem: build a per-layer feature normalization whose effect does **not**
depend on the batch dimension. Concretely it should (1) estimate its statistics from
something other than the batch, so accuracy does not collapse when the per-device batch is
small (2, 4 images), (2) compute exactly the same function during training and at
inference — no statistics accumulated by a running average that must then be frozen, hence
no train/test discrepancy and no inconsistency when transferring a pre-trained model to a
small-batch task — and (3) still match the accuracy of the batch-based layer in the regime
where the batch is large and that layer already works well. Many high-value vision tasks
(detection, instance segmentation, video) are forced into batch sizes of 1–2 per device by
the memory cost of high-resolution inputs and 3D convolutions, so the batch-size
restriction directly limits which models can be built.

## Background

**Why normalize hidden features at all.** Normalizing the *input* data makes training
faster (LeCun et al. 1998). Inside the network, controlling the distribution of each
layer's activations keeps gradients well-scaled and lets optimization proceed through many
layers. Weight-initialization schemes (Glorot & Bengio 2010; He et al. 2015) try to set up
good feature distributions analytically, but they rest on assumptions about the feature
distribution that stop holding once training moves the weights, so they do not by
themselves keep activations well-behaved throughout training. An explicit normalization
*layer* that re-centers and re-scales features on the fly is more robust.

**Early normalization layers.** Local Response Normalization (LRN) was used in AlexNet
(Krizhevsky et al. 2012) and successors: it normalizes each pixel using a small
neighborhood of channels around it, i.e. a local, per-pixel statistic. It predates and is
weaker than the more global schemes that followed.

**The batch-statistics view and its general form.** A large family of normalization layers
can be written in one form. Let a feature tensor be indexed by i = (i_N, i_C, i_H, i_W) in
(N, C, H, W) order — batch, channel, height, width. Each method picks a set S_i of feature
positions that share one mean and variance, then standardizes:

    x̂_i = (x_i − μ_i) / σ_i,
    μ_i = (1/m) Σ_{k∈S_i} x_k,
    σ_i = sqrt( (1/m) Σ_{k∈S_i} (x_k − μ_i)^2 + ε ),   m = |S_i|,

with ε a small constant for numerical stability. After standardizing, a learnable
per-channel affine transform y_i = γ x̂_i + β (γ, β indexed by channel) restores
representational power that pure standardization would remove. The methods below differ
**only** in how S_i is chosen; everything else is shared.

**Diagnostic finding — batch-based normalization degrades at small batch.** When the
normalization statistics are estimated over the batch axis, the quality of those estimates
depends on how many samples the batch provides. Measured on ResNet-50 / ImageNet, the
validation error of the batch-based layer climbs sharply as the per-device batch shrinks:
about 23.6, 23.7, 24.8, 27.3, 34.7 % at batch sizes 32, 16, 8, 4, 2. Fewer samples make the
sample statistics noisier, and the usual divide-by-m variance estimate has a larger
finite-sample bias; the model error tracks this loss of statistical quality. This
degradation is a property of normalizing along the batch dimension, not of any particular
network.

**Diagnostic finding — the batch is not always a stable or meaningful population.** Because
the statistics depend on the batch, the layer behaves differently at inference, where there
is no batch to pool over: population statistics are accumulated by a running average during
training and frozen for testing, so the function computed at test time is not the one
computed during training. The frozen statistics can also be wrong when the data
distribution shifts (transfer). And the batch is not always i.i.d. — e.g. when many
regions sampled from a single image form the batch, the samples are correlated, which
further corrupts the estimate. So beyond the small-batch noise, the very reliance on a
"batch" creates inconsistency across training, transferring, and testing.

**Channels are correlated but not interchangeable.** Classical hand-designed vision
features are *group-wise* and are normalized *within groups*. A HOG descriptor (Dalal &
Triggs 2005) is built from spatial cells, each a histogram of gradient orientations, and
the histogram is normalized within each block/orientation group; SIFT (Lowe 2004) similarly
normalizes orientation histograms; GIST, VLAD, and Fisher Vectors are grouped sub-vectors
(e.g. one sub-vector per cluster). The grouping reflects that the coefficients within a
group belong together. Convolutional channels are analogous: a filter and its
horizontally-flipped counterpart should produce similar response distributions on natural
images; orientation, frequency, shape, illumination, and texture induce natural groupings
of channels whose responses are interdependent. A well-accepted neuroscience model
likewise normalizes divisively across populations of cells with related receptive-field and
frequency tunings. So channels are neither fully independent nor all interchangeable.

## Baselines

**Batch-based normalization (Ioffe & Szegedy 2015).** S_i = { k | k_C = i_C }: for each
channel, pool over (N, H, W) — every position in that channel across the whole batch and
all spatial locations shares one μ, σ. Add the per-channel affine y = γ x̂ + β. It eases
optimization, enables very deep networks, and the noise from random batch sampling acts as
a regularizer that helps generalization. **Gap:** S_i spans the batch axis, so estimate
quality depends on batch size (small-batch degradation above) and the layer needs frozen
running statistics at inference (train/test discrepancy above). It needs a "sufficiently
large" batch (e.g. 32 per device) to work well.

**Per-sample, all-channel normalization (Ba et al. 2016).** S_i = { k | k_N = i_N }: for
each sample, pool over all of (C, H, W). Batch-independent — same at train and test, no
running statistics — and effective for recurrent models (RNN/LSTM). **Gap for vision:** it
forces *all* channels of a sample to share one mean and variance, i.e. it assumes every
channel makes a similar contribution. In a convolutional layer the channels are different
filters (edges, colors, textures, frequencies) with genuinely different response
distributions, so a single shared statistic over all of them is too coarse.

**Per-sample, per-channel normalization (Ulyanov et al. 2016).** S_i = { k | k_N = i_N,
k_C = i_C }: for each sample and each channel, pool over (H, W) only. Also batch-independent
and strong for style transfer and generative models. **Gap for vision:** with only the
spatial pixels of a single channel, it has no access to any other channel — it cannot use
the dependence *between* channels at all, which limits its accuracy on recognition.

**Weight normalization (Salimans & Kingma 2016).** Normalizes the *filter weights* rather
than the features (reparameterizing each weight vector by its direction and a separate
scale). Batch-independent, but it controls the parameters, not the feature distribution,
and has not matched the batch-based layer's accuracy on visual recognition.

**Constrained batch statistics (Ioffe 2017).** Computes correction factors between the
current batch statistics and running population statistics, then clips those factors to
limit how far the normalized activations can drift. This reduces small-batch error relative
to the plain batch layer. **Gap:** it is still batch-dependent, and its accuracy still
degrades as the batch shrinks.

**Cross-device batch statistics (Peng et al. 2018).** Computes the batch statistics jointly
across multiple devices to enlarge the effective batch. **Gap:** this does not remove the
batch dependence; it pays for a larger batch with proportionally more hardware, and it
blocks asynchronous training.

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
    mean/variance, and how to estimate it without depending on the batch.
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
