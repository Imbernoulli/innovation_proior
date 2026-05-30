# Context

## Research question

How can a convolutional network be trained *end-to-end* to produce good general-purpose visual features with *no supervision at all*, in a way that scales to internet-sized image collections?

Supervised pre-training on ImageNet has driven most of computer vision, but it is hitting limits: classifier performance on ImageNet is largely saturating (little error is left to resolve), and ImageNet is small by modern standards — a million images, curated for object classification. Pushing forward by building a far larger, more diverse dataset (potentially billions of images) would require an infeasible amount of manual annotation; replacing labels with raw metadata (hashtags and the like) injects biases into the learned representations with unpredictable effects. So a method is needed that trains a convnet from images alone. The difficulty is concrete: classical unsupervised tools were built for *linear* models on *fixed* features, and they break when the features must be learned at the same time — most starkly, naively coupling a convnet with k-means collapses to a trivial solution where the features go to zero and every cluster merges into one.

## Background

**Convnets and features.** A convnet `f_θ` maps a raw image to a fixed-dimensional feature vector; given a training set of `N` images, the goal is a parameter `θ*` such that `f_{θ*}` yields good general-purpose features. Supervised training pairs each image `x_n` with a label `y_n ∈ {0,1}^k` and learns a classifier `g_W` on top of the features by minimizing the multinomial logistic loss
`min_{θ,W} (1/N) Σ_n ℓ(g_W(f_θ(x_n)), y_n)`,
with mini-batch SGD and backpropagation. This pipeline — batch normalization (Ioffe & Szegedy, 2015), dropout (Srivastava et al., 2014), momentum, data augmentation — is mature and reliable; an unsupervised method that could reuse it would inherit all of that.

**Random convnets already carry signal.** A convnet whose parameters are sampled from a Gaussian, with no training at all, produces features far above chance: a multilayer perceptron on the last convolutional layer of a *random* AlexNet reaches about 12% accuracy on ImageNet, versus 0.1% chance. This good behavior of random convnets is tied to the convolutional structure itself, which is a strong prior on natural-image input. That weak but real signal is something a method could try to bootstrap.

**Classical unsupervised learning.** Clustering, dimensionality reduction, and density estimation are long-standing (k-means, spectral clustering, PCA). They apply to any domain or modality without labels — the bag-of-features model, for instance, clusters handcrafted local descriptors into image-level features. But they were designed for linear models on top of *fixed* descriptors; they scarcely work when the features are being learned simultaneously, and they had not been adapted to end-to-end convnet training at scale.

**Discriminative clustering and its trivial solutions.** Any method that *jointly* learns a discriminative classifier and the labels it predicts is prone to degenerate solutions — this is known even for linear discriminative clustering (Xu et al., 2005). An optimal decision boundary can assign every input to a single cluster; or, if most points fall into a few clusters, the model learns to discriminate only those, and in the extreme (all but one cluster singletons) predicts a constant output. Classical remedies constrain or penalize the minimum number of points per cluster (Bach & Harchaoui, 2008; Joulin et al., 2012), but those terms are computed over the whole dataset and do not fit mini-batch training of convnets at scale.

**Faiss k-means and empty-cluster handling.** Large-scale k-means libraries (Johnson et al., 2017) include a standard trick for empty clusters during optimization: reassign an empty cluster's centroid from a non-empty one with a small perturbation.

## Baselines

**Self-supervised learning (Doersch et al., 2015; Noroozi & Favaro, 2016; Pathak et al., 2016; colorization, video coherence, etc.).** Replace human labels with *pseudo-labels* derived from a hand-designed pretext task — predicting the relative position of patches, solving jigsaw puzzles, inpainting missing pixels, predicting color, exploiting temporal coherence. Strength: a real supervisory signal computed from raw data; strong transfer features. Gap: each is *domain-dependent* and requires expert design of a pretext task that happens to yield transferable features.

**Generative models for features (VAE, Kingma & Welling, 2013; GAN, Goodfellow et al., 2014; BiGAN/ALI, Donahue et al., 2016; Dumoulin et al., 2016).** Learn a mapping between noise and images; the GAN discriminator (or an added encoder) can produce features. Gap: features are a by-product, not the objective; plain GAN-discriminator features are relatively disappointing, and competitive features require adding an encoder.

**Clustering coupled with convnets (Coates & Ng, 2012; Yang et al., 2016; Bojanowski & Joulin, 2017).** Coates & Ng pre-train with k-means but layer-by-layer bottom-up rather than end-to-end. Yang et al. iteratively learn features and clusters in a recurrent framework — promising on small datasets but hard to scale to the image counts convnets need. Bojanowski & Joulin learn features with an information-preserving loss that discriminates between individual images (like exemplar SVM). Gap: none had been tested at the scale and on the modern architectures needed for a thorough study, and the naive coupling of convnet + k-means collapses to a trivial solution.

## Evaluation settings

- **Datasets.** ImageNet (1.3M images, 1000 classes) for training by default; YFCC100M Flickr images as an uncured alternative distribution. Transfer benchmarks: Pascal VOC classification, detection, and segmentation; image-retrieval benchmarks for instance-level information.
- **Metrics.** Downstream transfer accuracy (linear or fine-tuned classifiers on frozen/transferred features), measured with no fine-tuning on Pascal VOC for hyperparameter selection. Normalized Mutual Information, `NMI(A;B) = I(A;B) / √(H(A)·H(B))` (0 if independent, 1 if one is deterministic given the other), as a diagnostic — to gauge how cluster assignments relate to ground-truth labels and how stable the clusters are between successive rounds.
- **Protocol.** Train the convnet on images alone. Preprocess with a fixed Sobel filtering (remove color, increase local contrast). Per round: cluster the features of the central-cropped images, then train on the network with data augmentation (random flips, random-size/aspect crops). AlexNet (with batch norm, LRN removed) and VGG-16 (with batch norm); momentum 0.9, batch 256, L2 weight decay, dropout, constant step size. Features for clustering are PCA-reduced to 256 dimensions, whitened, and L2-normalized.

## Code framework

Pre-existing primitives: PyTorch convnet backbones (AlexNet, VGG), the supervised training loop with a softmax classifier and cross-entropy, mini-batch SGD with momentum, a faiss k-means with PCA/whitening, and a `Sampler` interface for the data loader. The supervised pipeline exists in full. What does not yet exist is where the *labels* come from when there are none, and the machinery to keep that label source from degenerating.

```python
import torch
from torch import nn
import numpy as np
import faiss

def preprocess_features(npdata, pca=256):
    # PCA-reduce, whiten, L2-normalize features before clustering (standard pre-clustering recipe)
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)   # whitening via eigen_power = -0.5
    mat.train(npdata); npdata = mat.apply_py(npdata)
    npdata = npdata / np.linalg.norm(npdata, axis=1)[:, None]
    return npdata

def run_kmeans(x, k):
    # faiss k-means; returns a cluster assignment per row
    # TODO: how empty clusters are prevented from collapsing the solution
    pass

class PseudoLabelSampler(torch.utils.data.Sampler):
    # TODO: how images are sampled given (arbitrary) cluster assignments,
    #       so that the network does not collapse onto a few dominant clusters
    def __init__(self, N, images_lists):
        pass
    def __iter__(self):
        pass

class Backbone(nn.Module):
    # convnet feature extractor + a classification head on top
    def __init__(self):
        super().__init__()
        self.features = None       # AlexNet/VGG conv+fc trunk (exists)
        self.classifier = None     # exists
        self.top_layer = None      # final linear layer -> one logit per (pseudo-)class
    def forward(self, x):
        pass

# one round of unsupervised training
def deepcluster_round(model, dataset, k, optimizer):
    feats = compute_features(model, dataset)          # forward pass over the whole dataset
    feats = preprocess_features(feats)
    assignments = run_kmeans(feats, k)                # pseudo-labels  (TODO: stable against trivial solutions)
    # TODO: what to do with model.top_layer given that cluster identities are arbitrary this round
    train_dataset = assign_pseudolabels(dataset, assignments)
    sampler = PseudoLabelSampler(len(train_dataset), assignments)   # TODO
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=sampler)
    crit = nn.CrossEntropyLoss()
    for x, pseudo_y in loader:
        loss = crit(model(x), pseudo_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
