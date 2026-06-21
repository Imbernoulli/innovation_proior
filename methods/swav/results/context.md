# Context

## Research question

We want to learn good visual representations from images *without any labels*, in a way that runs **online** — one streaming pass over data, with no separate phase that revisits the whole dataset — and scales to arbitrarily large, even unlimited, image collections, while working with either large or small batches on commodity numbers of GPUs. How can an online method produce strong general-purpose features from unlabeled images?

## Background

**The contrastive principle.** Given an image, draw two augmentations ("views"); their features should be similar; features of *other* images should be dissimilar. Formally this is a softmax over similarities (InfoNCE / NT-Xent): for a feature `z` and its positive `z+`, with negatives `z-`, minimize `-log [ exp(z·z+/τ) / (exp(z·z+/τ) + Σ exp(z·z-/τ)) ]`. The temperature `τ` controls how sharply the softmax concentrates. Representation quality increases with the number of negatives, which is why practical systems use large batches, memory banks, or momentum queues to supply them.

**Augmentations carry the invariances.** The set of transformations (random resized crop, color jitter, grayscale, Gaussian blur, horizontal flip) defines what the features become invariant to. Random cropping is especially important: comparing two crops forces the model to relate parts of a scene. Empirically, *comparing more views* of an image (beyond the standard two) improves the learned features.

**Clustering as an alternative to instance discrimination.** Instead of treating each image as a distinct class, one can group images with similar features and use the group identity as a target. This sidesteps explicit negatives. Existing clustering recipes alternate a full-dataset "assignment" pass with a "training" pass. Periodic cluster re-assignment, balanced batch sampling, and resetting the classifier head between assignments are techniques used to keep the clustering stable.

**Entropy-regularized optimal transport.** A principled way to assign points to groups while enforcing an **equipartition** constraint that spreads normalized mass evenly across groups. In the usual joint-distribution form, rows and columns have prescribed marginals such as `1/K` and `1/B`; multiplying the resulting matrix by `B` turns each column into an ordinary per-sample code. Adding an entropy term `εH(Q)` to a linear assignment objective makes the optimum a simple matrix-scaling form solvable by the **Sinkhorn–Knopp** algorithm — alternately normalizing the rows and columns of `exp(M/ε)` a few times. This is fast and GPU-friendly.

**Resolution bias.** Training a network on small (downsized) crops and testing on full-resolution images introduces a distribution mismatch in the features; mixing crop sizes during training mitigates it.

## Baselines

**Exemplar / instance classifier (Dosovitskiy et al. 2016).** Assign each image its own label and train an N-way classifier over augmentations.

**Memory bank + NCE (NPID, Wu et al. 2018).** Replace the explicit classifier with a *memory bank* storing one running feature vector per dataset image; use noise-contrastive estimation to compare the current feature against a sample of stored vectors.

**Momentum contrast (MoCo, He et al. 2019).** Maintain a slowly-updated *momentum encoder* and a *queue* of ~65k negative features from recent batches, giving a large, consistent negative set without a huge batch.

**SimCLR (Chen et al. 2020).** Take negatives from the *other elements of the same batch* with the NT-Xent loss, a 2-layer MLP **projection head** before the loss, strong augmentation, and temperature τ.

**DeepCluster (Caron et al. 2018).** Each epoch, run k-means on the features of the whole dataset to produce pseudo-labels, then train the network to classify images into these pseudo-labels with a softmax head.

**SeLa (Asano et al. 2019).** Recast the pseudo-label assignment as an **optimal-transport** problem with an equal-partition constraint, solved by Sinkhorn–Knopp, giving a principled balanced assignment used within the same loss and training phases as DeepCluster.

**Fixed-target alignment (NAT, Bojanowski et al. 2017).** Align features to a set of *fixed random* targets via a Hungarian matching. Shows targets need not be semantic class centroids.

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
