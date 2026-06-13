# Research question

Almost all recent convolutional-network research chases one number: accuracy on ImageNet. But for a fixed accuracy level there are usually *many* architectures that reach it, and they are not equally cheap. The unexplored axis is the *parameter count* of a network that hits a given accuracy. Why does it matter? Three concrete pain points. (1) Distributed data-parallel training is bottlenecked by communication, and that communication is proportional to the number of parameters — fewer parameters, faster training. (2) Pushing model updates over the air to deployed devices (e.g. cars receiving new vision models) means transferring the parameters; a 240MB AlexNet is a heavy update. (3) FPGAs and other embedded accelerators often have under 10MB of on-chip memory and no off-chip storage; a model that fits entirely on-chip can stream video through it in real time with no external memory traffic.

So the goal: find a convolutional architecture that matches the accuracy of a well-known reference model (AlexNet-level on ImageNet) with *drastically* fewer parameters — an order of magnitude or more — and ideally a model small enough (single-digit MB, or under 1MB after compression) to live on constrained hardware. And do it in a principled way, by understanding *which* architectural choices drive parameter count, rather than by black-box search.

# Background

**Where parameters live in a convolution.** A convolution layer made of 3×3 filters has a parameter count of `(input channels) × (number of filters) × 3 × 3`. The size is thus a product of the spatial footprint of each filter (`3×3 = 9` vs `1×1 = 1`), the number of filters, and the number of input channels feeding those filters. A 1×1 filter has 9× fewer parameters than a 3×3 filter, and is the cheapest convolution that still mixes channels.

**1×1 convolutions and Network-in-Network (Lin et al., 2013).** A 1×1 convolution performs pure cross-channel mixing at each pixel, with no spatial extent. NiN introduced using 1×1 convolutions as learned per-pixel channel transforms and replaced the parameter-heavy fully-connected classifier with global average pooling over the final feature maps — removing the single largest block of parameters in classic architectures (AlexNet/VGG fully-connected layers hold the bulk of their weights).

**Modules as a unit of design (GoogLeNet / Inception, Szegedy et al., 2014–2015).** Rather than hand-pick the dimensions of every layer, recent architectures define a small reusable *module* (e.g. the Inception module, which mixes 1×1, 3×3, and sometimes 5×5 filters) and stack many copies. This makes a large network describable by a handful of module hyperparameters. The proportion of 1×1 vs larger filters inside these modules was chosen without systematic analysis of how it trades off model size against accuracy.

**Depth and bypass connections.** VGG (Simonyan & Zisserman, 2014) showed accuracy rises with depth using only 3×3 filters; ResNet (He et al., 2015) and Highway Networks showed that *bypass* (skip) connections that add an earlier activation to a later one let much deeper networks train and improve accuracy (ResNet reported ~2 points top-5 from adding bypass to a 34-layer net). Bypass connections add little or no parameters but change the optimization and regularization of the layers they wrap.

**Delayed downsampling (He & Sun, "constrained time cost").** The spatial size of a layer's activation map is set by where in the network you place the stride>1 (downsampling) layers. If downsampling happens early, most layers operate on small maps; if it is pushed late, most layers see large maps. Empirically, delaying downsampling so that convolution layers keep large activation maps tends to raise classification accuracy at equal parameter budget.

**Model compression as the competing approach.** A parallel line of work shrinks an *already-trained* large model in a lossy way: SVD factorization of weight matrices (Denton et al., 2014); Network Pruning (Han et al., 2015), which zeroes sub-threshold weights to a sparse matrix and retrains; and Deep Compression (Han et al., 2015), which combines pruning, quantization to 5–8 bits via a codebook, and Huffman coding. These start from a big architecture; the alternative pursued here is to design a small architecture *from scratch* — and the two are complementary, since a small dense model can then also be compressed.

# Baselines

**AlexNet (Krizhevsky et al., 2012).** The reference accuracy target: ~57.2% top-1 / ~80.3% top-5 on ImageNet, with ~60M parameters (~240MB at 32-bit). Most parameters sit in its fully-connected layers. The yardstick for "AlexNet-level accuracy" and the model that compression papers start from. Its gap: enormous parameter count for that accuracy.

**Model-compression results on AlexNet.** SVD compresses AlexNet ~5× (top-1 drops to 56.0%); Network Pruning ~9× while maintaining accuracy; Deep Compression ~35× (to ~6.9MB) while maintaining accuracy. These define the parameter-reduction frontier a from-scratch small architecture is measured against. Their limitation: they require first training the large model, and the compressed model is a derivative of a heavy architecture rather than a small architecture in its own right.

**VGG / GoogLeNet.** Strong, deeper accuracy leaders, but parameter-heavy (VGG especially, due to large fully-connected layers and many 3×3 filters). They motivate the question of how to keep accuracy while removing the parameter bloat.

# Evaluation settings

- **ImageNet (ILSVRC 2012)**: 1000-class classification. Report top-1 and top-5 accuracy on the validation set. The accuracy target is "match AlexNet."
- **Model size**: number of bytes to store all trained parameters (32-bit baseline), and the size after applying compression (8-bit / 6-bit quantization with sparsity). The headline trade-off is accuracy vs. model size.
- **Training protocol available**: SGD-style training in the Caffe framework; a learning rate that decreases over training (e.g. starting around 0.04 with linear decay). ReLU activations; dropout before the final layers. No fully-connected classifier (global average pooling head, NiN-style).
- **Design-space exploration protocol**: train many architectures from scratch, each a single point, sweeping one metaparameter at a time (e.g. the ratio of 1×1 to 3×3 filters, or the degree of channel reduction before 3×3 filters) to chart how each choice moves model size and accuracy — principled A/B comparison rather than automated search.

# Code framework

The primitives below already exist: Conv2d (with a kernel size and channel counts), ReLU, MaxPool2d, Dropout, AdaptiveAvgPool2d, and channel-dimension concatenation. A network is a stack of these. What does not yet exist is the reusable *module* that achieves accuracy with few parameters; the scaffold leaves that as the empty slot.

```python
import torch
import torch.nn as nn


class FeatureModule(nn.Module):
    """TODO: the reusable building block. The whole contribution lives here —
    how to arrange convolution filters so the module is cheap in parameters
    while preserving representational power. Tunable channel dimensions only."""
    def __init__(self, in_channels, *dims):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Known good ingredients: an initial conv stem, max-pooling for
        # downsampling, a stack of FeatureModules, dropout, and a
        # global-average-pooling classifier head (no fully-connected layers).
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),  # stem dims are a choice
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        # TODO: self.features = nn.Sequential( ... FeatureModules + pools ... )
        feature_channels = None  # TODO: output width of the feature stack
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(feature_channels, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.stem(x)
        # x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
```
