# Context: normalization layers in deep networks (circa 2015-2018)

## Research question

By this point a normalization layer sits after almost every convolution in a deep network: it
recenters and rescales the activations of that layer, which controls the scale of internal
features, eases optimization, lets training use higher learning rates, and acts as a mild
regularizer. The standard recipe is to pick *one* normalization operation and use it
identically in every normalization layer of the whole network. Which operation is best depends
on the task (image classification behaves differently from artistic style transfer, which
behaves differently from a recurrent language model), on the architecture (a convolutional
backbone versus a detection head versus an LSTM cell), and — sharply — on the minibatch size
used during training. Today this choice is made by hand: a practitioner reads the literature,
decides "use batch statistics for classification, use per-image statistics for stylization,"
and bakes that one operation into every layer.

The question is whether there is a principled way to determine which statistics a normalization
layer should use, across different tasks, architectures, and batch-size regimes.

## Background

Two threads of prior work define the ground here. The first established *why* normalizing
activations helps and gave the canonical batch-based recipe; the second produced a family of
alternatives that pool statistics over different axes, each motivated by a place the canonical
recipe breaks.

**Normalization eases optimization.** Ioffe & Szegedy (2015) framed the difficulty of training
deep nets as *internal covariate shift*: as the parameters of lower layers change during
training, the distribution of inputs to each higher layer keeps moving, so every layer must
continually re-adapt to a shifting input distribution, which slows convergence and pushes
saturating nonlinearities toward their flat regions. Their fix normalizes each layer's
pre-activations to zero mean and unit variance as part of the architecture, computing the
normalizing mean and variance from the current minibatch, and then applies a learnable
per-channel affine — a scale `γ` and a shift `β` — so the layer can recover any scale and shift
it actually needs. This became the default component of deep vision networks.

**A general template, differing only in which pixels are pooled.** A unifying way to read the
whole family: take a layer input as a 4D tensor `h` of shape `(N,C,H,W)` — `N` samples, `C`
channels, `H×W` spatial positions — and normalize every pixel by

```
ĥ_{ncij} = γ · (h_{ncij} − μ) / sqrt(σ² + ε) + β,
```

where `ε` is a small constant for numerical stability. Every method in the family uses this
exact expression; they differ only in *which set of pixels* `I` they average over to get the
mean and variance:

```
μ = (1/|I|) Σ_{(n,c,i,j)∈I} h_{ncij},     σ² = (1/|I|) Σ_{(n,c,i,j)∈I} (h_{ncij} − μ)².
```

The choice of `I` is the entire design space, and the known methods are particular choices.

**Feature statistics, batch dependence, and the small-batch regime.** Two empirical facts about
this design space are relevant. First, the choice of `I` carries different *information*:
pooling over the batch keeps each example's per-channel mean/variance as a usable signal (it
subtracts one shared mean across the batch), whereas pooling within a single image subtracts
that image's own per-channel mean and variance and so erases exactly that per-image signal — a
distinction that matters because per-channel feature statistics are known to encode an image's
"style" (texture, contrast, color cast), which is a nuisance in some tasks and the label in
others. Second, when the mean and variance are estimated from the batch, their *quality* depends
on how many samples the batch has. With a large batch the batch mean and variance are good
estimates of the population statistics; as the batch shrinks the estimates become noisy, and
the normalization injects that noise into every forward pass. This is a reported, reproducible
degradation — batch-statistic normalization that is excellent with many samples per GPU loses
accuracy as the per-GPU sample count drops, and in the one-sample-per-GPU convolutional case
its training statistic collapses to the same spatial scope as instance-style pooling while
retaining the batch-normalization train/test machinery. Tasks like detection and segmentation
force the small-minibatch regime, because large input images leave room for only a couple of
samples per GPU.

## Baselines

These are the normalization layers a new design would be measured against and reacts to. Take
the layer input `h ∈ R^{N×C×H×W}`; all share the template above and differ in the pooling set
`I`. Write `μ_k, σ²_k` for the statistics of method `k`.

**Batch Normalization — BN (Ioffe & Szegedy 2015).** Pool over the batch and the spatial
dimensions, per channel: `I_bn = {(n,i,j)}`, so `μ_bn, σ²_bn ∈ R^{C}` — `2C` statistics, one
mean and one variance per channel. At test time the batch is not available or not
representative, so BN replaces the batch statistics with population estimates accumulated during
training (a running average).

**Instance Normalization — IN (Ulyanov, Vedaldi & Lempitsky 2016).** Pool over only the spatial
dimensions, separately for each sample and channel: `I_in = {(i,j)}`, so `μ_in, σ²_in ∈ R^{N×C}`
— `2NC` statistics. It subtracts each image's own per-channel mean and divides by its own
standard deviation, identically at train and test, so it has no batch dependence at all. This is
the right choice for artistic stylization, where erasing per-image style is the point.

**Layer Normalization — LN (Ba, Kiros & Hinton 2016).** Pool over all channels and spatial
positions within a single sample: `I_ln = {(c,i,j)}`, so `μ_ln, σ²_ln ∈ R^{N}` — `2N`
statistics, one mean and one variance per sample. Like IN it has no batch dependence and is
applied identically at train and test; it was introduced to ease optimization of recurrent
networks.

**Group Normalization — GN (Wu & He 2018).** Partition the channels into `g` groups and pool
over each group of channels together with the spatial positions, per sample. It is
batch-independent and was motivated precisely by BN's small-batch degradation. Its limiting
cases place it inside the same family: one group recovers layer-style pooling, one channel per
group recovers instance-style pooling.

**Reparameterizing the weights instead of the activations — WN (Salimans & Kingma 2016).**
A different lever entirely: rather than normalizing activations, reparameterize each weight
vector as `w = g · v/‖v‖`, decoupling its length `g` from its direction `v/‖v‖` to improve the
conditioning of the optimization and speed up SGD, without introducing any dependence between
examples in a batch. It is useful here less as a competitor than as an analysis tool — it
expresses normalization in *weight* space, which gives a common geometric language for comparing
the activation-space methods by how each one effectively constrains the filter norm.

**Variants that patch BN's batch dependence — BRN (Ioffe 2017), BKN (Wang et al. 2018).** Batch
Renormalization adds correction factors `r, d` so that training and inference use closer
statistics, reducing minibatch dependence; Batch Kalman Normalization estimates a full
covariance across layers.

## Evaluation settings

The natural yardsticks already in place for a normalization layer — pre-existing datasets,
backbones, and protocols into which the layer is swapped.

- **Image classification, across batch sizes.** ImageNet (1.28M training images, 1000 classes),
  ResNet-50 backbone with a normalization layer after each convolution; top-1 accuracy on the
  224×224 center crop. The key sweep is over batch configurations written as
  `(#GPUs, #samples per GPU)` — e.g. `(8,32)` down to `(8,2)`, single-GPU `(1,16)`/`(1,32)`, and
  the extremes `(1,8)` and `(8,1)` — to probe behavior as the per-GPU sample count shrinks. The
  gradient is aggregated over all GPUs while the normalization statistics are computed per GPU.
- **Optimization protocol (classification).** SGD with momentum, weight decay `1e-4` applied to
  all parameters including the affine `γ, β`; He-style parameter initialization; all `γ`
  initialized to 1 and all `β` to 0; initial learning rate 0.1, dropped by 10× at fixed epochs,
  trained for 100 epochs, with the learning rate linearly scaled with batch size; standard
  ImageNet data augmentation.
- **Dense prediction at small batch.** Object detection and instance segmentation on COCO with
  Faster R-CNN / Mask R-CNN and a Feature Pyramid Network on a ResNet-50 backbone (metric:
  bounding-box and mask average precision, AP); semantic segmentation on Cityscapes and ADE20K
  (metric: mean IoU, single- and multi-scale). These tasks use large input images and so run at
  2 samples per GPU, the regime where batch statistics are weakest.
- **Other regimes.** Video recognition on Kinetics; artistic image stylization with a
  feed-forward image-transformation network trained against a fixed VGG loss (the setting where
  per-image normalization is normally the right choice); a recurrent cell inside a neural
  architecture search.
- **Diagnostic protocol.** Repeat training under different solvers, initializations, and
  learning-rate schedules with the network, task, batch setting, and data fixed, and inspect how
  the layer's behavior varies; histogram the per-layer behavior across the network and across
  batch sizes.

## Code framework

The new layer plugs into the existing CNN training harness as a drop-in replacement for the
batch-normalization module: same constructor `(num_features=C)`, same `[N,C,H,W]` input/output,
the same learnable per-channel affine `γ` (`weight`) and `β` (`bias`), and numerically stable
behavior in both train and eval. The general normalization template above already exists as a
primitive — given a pooling set, compute its mean and variance, standardize, then apply the
affine — and the library can compute per-axis means and variances over any chosen axes via
reshaping. What is *not* settled is how the layer decides which statistics to use; that decision
rule is exactly the contribution. So the substrate below supplies the affine parameters, the
running-statistics buffers a batch-style estimate would need at test, and the standardize-and-
affine tail, and leaves one empty slot for the rule that produces the mean and variance the
layer normalizes with — plus a place for any extra learnable parameter that rule introduces.

```python
import torch
import torch.nn as nn


class CustomNorm2d(nn.Module):
    """Drop-in replacement for the per-channel normalization layer used after each
    convolution. Constructor takes the channel count C; forward maps [N, C, H, W] to
    [N, C, H, W] and must be numerically stable in both train and eval."""

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # learnable per-channel affine, applied after standardization
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))    # beta
        # buffers for the population statistics a batch-style estimate uses at test time
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        # TODO: any extra learnable parameter the decision rule we design introduces

    def forward(self, x):
        N, C, H, W = x.size()
        # The library can compute a mean/variance over any chosen axes (here via reshaping
        # x to [N, C, H*W], pooling over whichever axes the chosen statistic needs).
        # At test time the batch-pooled statistic is replaced by the running buffers above.
        #
        # TODO: the decision rule we will design — produce the mean and variance this layer
        #       normalizes with from `x` (and any parameter introduced above).
        mean = ...   # TODO
        var = ...    # TODO
        x = (x - mean) / (var + self.eps).sqrt()    # standardize with the chosen statistics
        return x * self.weight + self.bias          # learnable per-channel affine


# existing minibatch training loop the layer plugs into; unchanged
def train(model, loss_fn, data_loader, optimizer):
    model.train()
    for inputs, targets in data_loader:
        loss = loss_fn(model(inputs), targets)   # CustomNorm2d runs after each conv
        optimizer.zero_grad()
        loss.backward()                          # backprop fills grads for all parameters,
        optimizer.step()                         # including any the decision rule introduces
```

The outer loop and the standardize-and-affine tail are fixed; the contribution lives in the
slot that produces `mean` and `var`, together with any learnable parameter it needs (which then
trains by the same backprop and optimizer step as everything else).
