# Context

## Research question

We want to learn good visual representations from images *without any labels*, in a way that (a) runs **online** — one streaming pass over data, with no separate phase that revisits the whole dataset — and therefore (b) scales to arbitrarily large, even unlimited, image collections, and (c) does so **cheaply in memory and compute**, working with either large or small batches on commodity numbers of GPUs.

The dominant recipe for label-free representation learning treats every image as its own class ("instance discrimination") and learns by *directly comparing pairs of feature vectors*: features from two augmentations of the same image are pulled together, features from different images are pushed apart. That comparison is the engine, but it is also the cost. Pushing apart requires "negatives," and the quality of the representation grows with the *number* of negatives compared against. Computing all pairwise comparisons over a dataset is quadratic and infeasible, so every practical system carries extra machinery whose only purpose is to supply enough negatives: very large batches, a memory bank of past features, or a second momentum-updated encoder with a long queue. The question is whether the same "two views of one image should agree" signal can be captured **without explicit pairwise feature comparison at all**, and crucially **without the representation collapsing** to a constant — while staying online.

## Background

**The contrastive principle.** Given an image, draw two augmentations ("views"); their features should be similar; features of *other* images should be dissimilar. Formally this is a softmax over similarities (InfoNCE / NT-Xent): for a feature `z` and its positive `z+`, with negatives `z-`, minimize `-log [ exp(z·z+/τ) / (exp(z·z+/τ) + Σ exp(z·z-/τ)) ]`. The temperature `τ` controls how sharply the softmax concentrates. The crucial empirical regularity established before any clustering approach: **representation quality increases with the number of negatives**. That single fact is *why* all the surrounding machinery exists — large batches, memory banks, momentum queues are all just ways to put more negatives in the denominator. It is also the cost: more negatives means more memory and more pairwise dot products.

**Augmentations carry the invariances.** The set of transformations (random resized crop, color jitter, grayscale, Gaussian blur, horizontal flip) defines what the features become invariant to. Random cropping is especially important: comparing two crops forces the model to relate parts of a scene. Empirically, *comparing more views* of an image (beyond the standard two) improves the learned features — but naively adding full-resolution views increases memory and compute roughly quadratically, which is why systems stick to two.

**Clustering as an alternative to instance discrimination.** Instead of treating each image as a distinct class, one can group images with similar features and use the group identity as a target. This sidesteps explicit negatives. But the existing clustering recipes are **offline**: they need the features of the *entire* dataset to form the groups, so they alternate a full-dataset "assignment" pass with a "training" pass. They also have a notorious failure mode — **collapse**: nothing stops the optimizer from putting all images in one cluster (or assigning every image the same code), which trivially minimizes the loss while learning nothing. The known fixes (periodic cluster re-assignment, balanced batch sampling, resetting the classifier head between assignments) are heuristic and disruptive.

**Entropy-regularized optimal transport.** A separate, principled way to assign points to groups while *forbidding* the collapse: pose assignment as a transport problem and add an **equipartition** constraint that forces normalized mass to spread evenly across groups. In the usual joint-distribution form, rows and columns have prescribed marginals such as `1/K` and `1/B`; multiplying the resulting matrix by `B` then turns each column into an ordinary per-sample code. Adding an entropy term `εH(Q)` to a linear assignment objective makes the optimum a simple matrix-scaling form solvable by the **Sinkhorn–Knopp** algorithm — alternately normalizing the rows and columns of `exp(M/ε)` a few times. This is fast and GPU-friendly. It was the tool that turned heuristic balanced clustering into a clean constrained optimization, but as used so far it ran on the *full dataset* and rounded the result to a hard assignment.

**Resolution bias.** Training a network on small (downsized) crops and testing on full-resolution images introduces a distribution mismatch in the features; mixing crop sizes during training mitigates it.

## Baselines

**Exemplar / instance classifier (Dosovitskiy et al. 2016).** Assign each image its own label and train an N-way classifier over augmentations. *Gap:* the classifier has as many outputs as images; intractable as the dataset grows.

**Memory bank + NCE (NPID, Wu et al. 2018).** Replace the explicit classifier with a *memory bank* storing one running feature vector per dataset image; use noise-contrastive estimation to compare the current feature against a sample of stored vectors. *Gap:* must store features for the entire dataset, and the stored vectors are stale.

**Momentum contrast (MoCo, He et al. 2019).** Maintain a slowly-updated *momentum encoder* and a *queue* of ~65k negative features from recent batches, giving a large, consistent negative set without a huge batch. *Gap:* a second encoder plus a long queue — extra memory and moving parts; still fundamentally pairwise.

**SimCLR (Chen et al. 2020).** Drop the bank and queue; take negatives from the *other elements of the same batch* with the NT-Xent loss, a 2-layer MLP **projection head** before the loss, strong augmentation, and temperature τ. *Gap:* needs *very large batches* (thousands) to have enough negatives.

**DeepCluster (Caron et al. 2018).** Each epoch, run k-means on the features of the whole dataset to produce pseudo-labels, then train the network to classify images into these pseudo-labels with a softmax head. No explicit negatives. *Gap:* **offline** (a full forward pass over the dataset to re-cluster each epoch — about a third of total training time); prone to **collapse**, requiring re-assignment and balanced-sampling tricks; and because cluster identities are arbitrary across epochs, the classification head must be **reset at every re-assignment**, disrupting training.

**SeLa (Asano et al. 2019).** Recast the pseudo-label assignment as an **optimal-transport** problem with an equal-partition constraint, solved by Sinkhorn–Knopp, giving a principled collapse-free assignment with the same loss in assignment and training phases. *Gap:* still **offline** over the whole dataset, and it **rounds** the soft transport solution to a hard label.

**Fixed-target alignment (NAT, Bojanowski et al. 2017).** Align features to a set of *fixed random* targets via a Hungarian matching. Shows targets need not be semantic class centroids. *Gap:* one target per dataset instance, hard assignment, batch-level matching.

## Evaluation settings

Pretraining is on **ImageNet** (1.28M images, 1000 classes) with labels withheld; larger-scale runs use uncurated web images. The standard yardstick for the learned features is the **linear evaluation protocol**: freeze the backbone, train a single linear classifier on top, report top-1 accuracy. The reference backbone is **ResNet-50** (24M params), with wider variants (×2/×4/×5) to test scaling. Further probes that predate this work: **semi-supervised** fine-tuning with 1% / 10% of labels; **transfer** via linear classifiers on Places205, VOC07, iNaturalist-2018, and **object detection / segmentation** fine-tuning (Faster R-CNN, Mask R-CNN, and the DETR transformer detector) on VOC07+12 and COCO; **k-nearest-neighbor** classification on frozen features. Compute/memory are measured as wall-clock per 100 epochs and peak GPU memory. Augmentation follows the contrastive standard: random resized crop, horizontal flip, color distortion, Gaussian blur.

## Code framework

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets

# ---- data: produce multiple augmented views of each image ----
def make_transforms():
    # known contrastive augmentation pipeline
    color = [get_color_distortion(), GaussianBlur()]
    base = [T.RandomHorizontalFlip(0.5), T.Compose(color), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.228,0.224,0.225])]
    # TODO: how many views, and how they are sampled
    raise NotImplementedError

class ViewsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        # TODO: return the list of augmented views for this image
        pass

# ---- backbone + the slot the method will occupy ----
class Encoder(nn.Module):
    def __init__(self, output_dim=128, hidden_mlp=2048):
        super().__init__()
        self.backbone = resnet50_trunk()                 # standard ConvNet, global-avg-pooled
        self.projection_head = nn.Sequential(            # 2-layer MLP, as in contrastive practice
            nn.Linear(2048, hidden_mlp), nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim),
        )
        # TODO: whatever extra module the learning signal requires
        self.head = None

    def forward(self, inputs):
        z = self.projection_head(self.backbone(inputs))
        z = F.normalize(z, dim=1, p=2)                   # embeddings on the unit sphere
        # TODO: produce whatever the loss needs from z
        return z

# ---- the per-view learning signal ----
def loss_across_views(views_outputs):
    # TODO: define the training objective over the views of each image
    pass

# ---- training loop: known scaffold ----
def train(loader, model, optimizer, lr_schedule):
    for it, views in enumerate(loader):
        for g in optimizer.param_groups: g["lr"] = lr_schedule[it]
        outputs = model(torch.cat(views))
        loss = loss_across_views(outputs)                # TODO
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
