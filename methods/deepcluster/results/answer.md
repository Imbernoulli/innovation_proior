# DeepCluster

## Problem

Learn good general-purpose visual features by training a convnet end-to-end with no labels, at scale. Classical clustering assumes fixed features, and naively coupling a convnet with k-means collapses to a trivial solution (zero features, one cluster).

## Key idea

Bootstrap the weak signal already present in a convnet (a random convnet scores ~12% on ImageNet vs 0.1% chance, thanks to its convolutional prior) by **alternating** two steps each epoch:

1. **Cluster** the current features `f_θ(x_n)` with k-means to produce **pseudo-labels**; keep only the assignments, discard the centroids.
2. **Train** the convnet to predict those pseudo-labels with the ordinary supervised objective (multinomial logistic loss, mini-batch SGD).

Each round, better features give a better clustering and a better clustering trains better features. k-means objective:
`min_{C ∈ ℝ^{d×k}} (1/N) Σ_n min_{y_n ∈ {0,1}^k} ‖f_θ(x_n) − C y_n‖₂²` s.t. `y_n^⊤ 1_k = 1`.

## Avoiding trivial solutions

Jointly learning a classifier and its labels is degenerate; two scalable, local fixes:
- **Empty clusters.** During the k-means solve, when a cluster empties, copy a random non-empty cluster's centroid (with a small perturbation) and split that cluster's points between the two.
- **Trivial parametrization.** Sample images **uniformly over clusters** (weight each image's loss by the inverse of its cluster size), so the network cannot collapse onto a few dominant clusters.

Because each re-clustering yields an arbitrary permutation of cluster indices, the **final classification layer is reinitialized every round** (random Normal(0, 0.01) weights, zero bias) while the convnet backbone carries forward.

## Other choices

- **Sobel filtering** of inputs (remove color, increase local contrast) so features don't shortcut on color.
- Features **PCA-reduced to 256 dims, whitened, L2-normalized** before k-means (so its Euclidean geometry is meaningful, and clustering is fast).
- **Cluster central-cropped images; train with augmentation** (random flips, random-size/aspect crops) → augmentation invariance.
- **Re-cluster every epoch** (features drift); nearly optimal on ImageNet.
- **k = 10,000** clusters (over-segmentation beyond the 1000 true classes yields richer features).
- AlexNet (5 conv: 96/256/384/384/256 filters, 3 FC; LRN removed, batch norm added) or VGG-16+BN; momentum 0.9, batch 256, L2 weight decay, dropout, constant step size; ~500 epochs.

## Code

```python
import numpy as np
import faiss
import torch
from torch import nn

def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)        # whitening
    mat.train(npdata); npdata = mat.apply_py(npdata)
    npdata = npdata / np.linalg.norm(npdata, axis=1)[:, None]  # L2 normalize
    return npdata


def run_kmeans(x, k, n_iter=20):
    n, d = x.shape
    clus = faiss.Clustering(d, k)
    clus.niter = n_iter
    clus.max_points_per_centroid = 10_000_000
    index = faiss.IndexFlatL2(d)
    clus.train(x, index)                                      # empty clusters revived during solve
    _, assignments = index.search(x, 1)
    return assignments.reshape(-1)


class UnifLabelSampler(torch.utils.data.Sampler):
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
    def __iter__(self):
        nonempty = [c for c in self.images_lists if len(c) > 0]
        per = int(self.N / len(nonempty)) + 1                 # uniform budget per cluster
        res = np.concatenate([
            np.random.choice(c, per, replace=(len(c) <= per)) for c in nonempty])
        np.random.shuffle(res)
        return iter(res[: self.N].astype(int))
    def __len__(self):
        return self.N


def deepcluster_round(model, dataset, k, optimizer, optimizer_tl, fd):
    feats = compute_features(model, dataset)                  # forward pass over all images
    feats = preprocess_features(feats)
    assignments = run_kmeans(feats, k)
    images_lists = [[] for _ in range(k)]
    for i, a in enumerate(assignments):
        images_lists[a].append(i)

    # cluster identities are arbitrary each round -> rebuild the head
    model.top_layer = nn.Linear(fd, k)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()

    train_dataset = cluster_assign(images_lists, dataset.imgs)
    sampler = UnifLabelSampler(len(train_dataset), images_lists)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, sampler=sampler)

    crit = nn.CrossEntropyLoss()
    for x, pseudo_y in loader:
        loss = crit(model(x), pseudo_y)
        optimizer.zero_grad(); optimizer_tl.zero_grad()
        loss.backward()
        optimizer.step(); optimizer_tl.step()


# outer loop: one round per epoch
# for epoch in range(500):
#     deepcluster_round(model, dataset, k=10000, optimizer, optimizer_tl, fd)
```
