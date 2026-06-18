# Context

## Research question

How can a convolutional network be trained end-to-end to produce transferable visual features from a large image collection with no human labels?

The motivation is practical. Supervised pretraining on ImageNet has become the default source of reusable visual features, but ImageNet is small by modern standards, curated for object classification, and manually labeled. Scaling that recipe to billions of diverse internet images would require annotation at a scale that is not realistic. Replacing labels with raw metadata such as tags is not the same solution: metadata is biased, noisy, and distributed very unevenly. The target is therefore a label-free training procedure that can reuse standard convnet optimization machinery, can run at ImageNet/YFCC scale, and does not depend on hand-designing a semantic pretext task for every new domain.

The hard part is not merely "run clustering." Classical unsupervised objectives assume fixed descriptors. When the descriptor itself is a convnet being learned at the same time, a naive joint objective has easy degenerate optima: the representation can become uninformative, or a discriminative learner can put nearly all images into a tiny set of labels and still minimize its local loss.

## Background

**Supervised convnet training.** A convnet `f_theta` maps an image to a feature vector, and a classifier `g_W` maps that feature to class logits. With labels `y_n in {0,1}^k`, the standard objective is

`min_{theta,W} (1/N) sum_n ell(g_W(f_theta(x_n)), y_n)`,

where `ell` is the multinomial logistic loss. This pipeline already has robust engineering support: minibatch SGD, backpropagation, momentum, weight decay, dropout, batch normalization, and data augmentation.

**A weak architectural signal already exists.** Random convnets are poor feature extractors, but not chance-level ones. A classifier trained over the last convolutional layer of a random AlexNet reaches about 12% ImageNet accuracy, while chance is 0.1%. The convolutional prior supplies a weak signal that a method might be able to amplify without labels.

**Clustering and dimensionality reduction.** k-means, PCA, spectral methods, and density estimation work well as unsupervised tools over fixed descriptors; bag-of-features vision pipelines use clustering over handcrafted local descriptors. These methods are domain-agnostic, which is attractive, but they were not designed for an end-to-end convnet whose feature space changes during training.

**Discriminative clustering degeneracy.** Methods that learn both labels and a discriminative boundary are known to need balance or minimum-cluster-size constraints. Classical relaxations impose constraints over the whole dataset, which is awkward for minibatch convnet training on millions of images. Large-scale clustering libraries also contain practical machinery for keeping empty clusters from breaking a k-means run.

## Baselines

**Self-supervised pretext tasks.** Patch-position prediction, jigsaw puzzles, inpainting, colorization, video coherence, and related tasks compute labels from the raw input itself. They provide real supervision and can transfer well, but each task is hand-designed and domain-specific.

**Generative feature learning.** Autoencoders, VAEs, GANs, BiGAN/ALI-style encoders, and reconstruction losses learn image models from unlabeled data. Their representations can be useful, but the feature extractor is often a by-product rather than the direct objective, and plain GAN-discriminator features had not matched the strongest discriminative self-supervised features.

**Earlier clustering with neural networks.** Some methods use k-means for layerwise pretraining, jointly learn clusters and features on smaller datasets, or learn discriminative features with information-preserving losses. The open gap is a simple end-to-end convnet procedure that can be studied at large image scale and with standard architectures.

## Evaluation settings

- **Pretraining data.** ImageNet training images with labels withheld; an uncured Flickr/YFCC-style image distribution as a stress test for scale and dataset bias.
- **Transfer tasks.** Pascal VOC classification, detection, and semantic segmentation; ImageNet and Places linear probes over frozen convolutional features; instance-level image retrieval for information beyond class labels.
- **Diagnostics.** Normalized Mutual Information, `NMI(A;B) = I(A;B) / sqrt(H(A) H(B))`, to measure how learned partitions relate to ground-truth labels when labels are used only for analysis, and how stable successive partitions are over training.
- **Architecture controls.** AlexNet for comparability with earlier unsupervised work; deeper VGG-style convnets to test whether the method benefits from a stronger backbone.

## Code framework

The existing ingredients are a supervised convnet training loop, an unlabeled image dataset, feature extraction over the full dataset, a scalable clustering primitive, and a `Sampler` interface. The missing pieces are how to turn unlabeled features into a stable training target, how to avoid degenerate targets, and how to keep the classifier attached to changing unsupervised targets.

```python
import numpy as np
import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = ...
        self.classifier = ...
        self.top_layer = ...

    def forward(self, x):
        z = self.features(x)
        z = self.classifier(z)
        if self.top_layer is not None:
            z = self.top_layer(z)
        return z


def extract_features(model, image_loader):
    # TODO: decide which view of each image should define the unsupervised target.
    pass


def make_unsupervised_targets(features):
    # TODO: produce one training target per image without using human labels.
    pass


def make_training_sampler(targets):
    # TODO: prevent target imbalance from dominating minibatch training.
    pass


def train_with_targets(model, dataset, targets, sampler, optimizer):
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=sampler)
    for x, y in loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
