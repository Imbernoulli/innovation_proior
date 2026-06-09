## Research question

Modern convolutional networks carry tens to hundreds of millions of parameters — the representational power needed for hard vision tasks, but also a standing invitation to overfit. The two regularizers that work best in practice are data augmentation (flips, crops, color jitter — cheap, ubiquitous, effective) and dropout (randomly zeroing hidden activations to break the co-adaptation of feature detectors). Dropout, however, is conspicuously *weaker inside convolutional layers* than inside fully-connected ones, and the variants invented to fix that tend to lose their edge the moment batch normalization or data augmentation is also present — at which point plain dropout quietly wins again.

The question is whether there is a regularizer that genuinely helps a *convolutional* network, that keeps helping when stacked on top of batch normalization and standard augmentation, that costs almost nothing, and that is trivial to implement.

## Background

**The field state.** CNNs trained with SGD-with-momentum, weight decay, batch normalization, and heavy data augmentation are the standard. Augmentation pipelines for natural images are mature (pad-and-crop, horizontal mirror, per-channel normalization), and dropout is the default activation-level regularizer.

**Why dropout weakens in convolutional layers.** Two reasons, both diagnostic. First, convolutional layers already have far fewer parameters than fully-connected layers, so they need less regularization to begin with. Second — and this is the load-bearing observation — *neighboring pixels in an image carry nearly the same information*. If dropout zeroes a single activation (or input pixel), the information it held is still passed forward by its still-active neighbors; the removal is undone by redundancy. The consequence is that dropout in conv layers does not produce the bagging/model-averaging effect it produces in FC layers; it merely makes feature detectors a little more robust to noise.

**Per-map dropout variants and their ceiling.** SpatialDropout (Tompson et al., 2015) drops entire feature maps rather than individual activations, sidestepping the neighbor-redundancy problem within a map. Max-drop (Park & Kwak, 2016) drops the maximal activation across maps/channels. These help in isolation, but both were found to underperform plain dropout once batch normalization is in the network. A structural commonality: each operates on feature maps *individually*, so a feature removed from one map can still be present in another, leaving an inconsistent, noisy representation.

**Reconstruct-from-context precedents.** Denoising autoencoders (Vincent et al., 2010) corrupt the input by erasing random *individual* pixels and train the model to reconstruct it — forcing useful features but only local context. Context encoders (Pathak et al., 2016) erase a large *contiguous* region and require reconstruction, which forces a *global* understanding of image content and yields higher-level features — though this had been used for self-supervised representation learning, not for supervised classification.

**Augmentation as occlusion.** Within the augmentation lineage (LeCun's affine transforms; Krizhevsky's AlexNet flips/crops/PCA-color), the closest prior is Bengio et al.'s practice of overlaying scratches and partial occlusions on characters. Object occlusion is itself a pervasive real-world condition in recognition, tracking, and pose estimation — a model that has only ever seen unoccluded objects is brittle when a key part is hidden.

## Baselines

**Dropout (Hinton et al., 2012; Srivastava et al., 2014).** Set hidden activations to zero with probability p during training; keep all at test time but rescale by p (model averaging over an exponential family of sub-networks). Core idea: discourage co-adaptation of feature detectors. Gap: in convolutional layers the neighbor-redundancy of pixels/activations defeats pointwise zeroing — it degrades to mere noise-robustness, no averaging effect.

**SpatialDropout (Tompson et al., 2015).** Drop entire feature maps at random. Core idea: remove a whole channel so neighbor redundancy within a map can't restore it. Gap: still feature-level and per-map — a feature dropped in one map survives in others; underperforms plain dropout once batch normalization is present.

**Max-drop (Park & Kwak, 2016).** Drop the maximal activation across feature maps/channels with some probability. Core idea: directly suppress the most prominent feature so the network must use others. Gap: like SpatialDropout, loses to standard dropout when combined with batch normalization.

**Denoising autoencoder corruption (Vincent et al., 2010).** Erase random individual input pixels and reconstruct. Core idea: input-space erasing forces feature learning. Gap: per-pixel erasing is local and, by neighbor-redundancy, weakly informative; it is a representation-learning objective, not a supervised regularizer.

**Context-encoder erasing (Pathak et al., 2016).** Erase a large contiguous input region and reconstruct it. Core idea: contiguous removal forces global understanding and higher-level features. Gap: framed as self-supervised reconstruction; not applied to directly regularize a supervised classifier's training, and it pairs the masking with a reconstruction decoder rather than the classification loss itself.

## Evaluation settings

The yardsticks are natural-image recognition benchmarks at 32×32: CIFAR-10 and CIFAR-100 (50k train / 10k test; 10 and 100 classes) and SVHN (73k + 531k extra train / 26k test digits), plus STL-10 (96×96, only 5k labeled train images — a low-data, higher-resolution probe). Backbones are modern residual networks — ResNet-18, Wide-ResNet (WRN-28-10 with conv-layer dropout p=0.3, WRN-16-8 for SVHN with p=0.4), and shake-shake-regularized ResNet/ResNeXt — trained with SGD, Nesterov momentum 0.9, weight decay 5e-4, batch size 128, step-decayed learning rate over 160–200 epochs. Standard augmentation ("+") is zero-pad-4-then-32×32-random-crop plus horizontal mirror at 50%; data is per-channel mean/std normalized. The metric is test error rate (averaged over several runs); a held-out 10% validation split is used to tune any new hyperparameter. The natural comparison points are the same architectures trained with existing augmentation, dropout, and batch normalization.

## Code framework

The available ingredients are a CIFAR/SVHN dataset, a torchvision transform pipeline (random crop, horizontal flip, tensor conversion, per-channel normalization), a loader yielding minibatches, a residual network, a cross-entropy criterion, and an SGD training loop. The open slot is a single callable to be filled in.

```python
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class NewTransform(object):
    """A transform applied to each image tensor during data loading.
    Returns an image of the same shape; label is untouched."""

    def __init__(self):
        pass

    def __call__(self, img):
        # img: a (C, H, W) tensor, already normalized
        # TODO: fill in.
        pass


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    # the new transform would be appended here
])
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=8)

# network, nn.CrossEntropyLoss(), SGD(Nesterov, mom=0.9, wd=5e-4),
# step-decay schedule, standard train loop — all unchanged.
```
