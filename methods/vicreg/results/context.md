# Context

## Research Question

The setting is joint-embedding self-supervised learning for images. A minibatch supplies unlabeled
images, an augmentation pipeline makes two distorted views of each image, and a network is trained
so that the two views of the same image have nearby embeddings. The hope is that the representation
kept after pretraining captures the content that survives random crops, color changes, blur,
grayscale conversion, and flips.

The bare objective is underdetermined. If every input maps to the same vector, then every pair of
views agrees and the training loss can be minimized without learning useful information. A useful
objective therefore needs a second job besides view agreement: it must make constant embeddings and
low-information embeddings unattractive.

The question is how to state this anti-collapse job as an ordinary statistic of the embedding batch,
rather than supplying it via negatives, queues, target networks, stop-gradient placement, cluster
balancing, or normalization.

## Prior Mechanisms

Contrastive methods such as SimCLR and MoCo pull together two views of one image while pushing away
other images. The repulsive term makes a constant representation bad, and it is estimated from many
negative pairs. SimCLR also shows that the augmentation composition and a projection head are
load-bearing parts of the pipeline.

Clustering methods such as DeepCluster and SwAV avoid comparing every image pair by assigning images
to prototypes or clusters. SwAV learns assignments online and keeps them balanced with an optimal
transport step.

Distillation-style methods such as BYOL and SimSiam remove negatives and regress one view toward the
other. They work by adding asymmetry: BYOL uses a target network updated by exponential moving
average, and SimSiam uses a predictor with a stop-gradient target.

Redundancy-reduction and whitening methods move closer to a direct information criterion. Barlow
Twins drives a normalized cross-correlation matrix between two branches toward the identity, making
same-index coordinates agree and different-index coordinates decorrelate. W-MSE whitens latent
features before applying an MSE.

## What Must Be Measured

There are two collapse modes. Trivial collapse is the constant-vector solution: every coordinate has
essentially zero spread across the batch. Informational collapse is less visible: the embeddings keep
nonzero spread, but many coordinates duplicate the same signal, so the effective dimension is much
smaller than the embedding dimension.

A batch statistic aimed at trivial collapse looks at each coordinate across samples and asks whether
it has enough spread. A statistic aimed at informational collapse looks between coordinates and asks
whether they move together.

The invariance objective ties the statistics to the image content by requiring two augmented views of
the same image to match.

## Evaluation Frame

The standard pretraining setting is unlabeled ImageNet-1k with a ResNet-50 encoder and a projection
or expansion head used only during pretraining. After pretraining, the head is discarded and the
frozen encoder representation is evaluated.

The usual evaluation protocols are linear classification on ImageNet, semi-supervised fine-tuning on
1 percent and 10 percent of ImageNet labels, transfer to classification datasets such as Places205,
VOC07, and iNaturalist2018, and transfer to detection or segmentation tasks using Faster R-CNN or
Mask R-CNN. k-NN evaluation on frozen ImageNet representations is another diagnostic.

## Code Frame

The codebase already has the ordinary training pieces: a two-view augmentation transform, a ResNet
encoder, a head on top of the encoder representation, distributed training, LARS, warmup, cosine
learning-rate decay, checkpointing, and downstream evaluation. The missing part is the loss computed
on the two embedding batches.

```python
import torch
import torch.nn.functional as F
from torch import nn


class JointEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone, repr_dim = build_resnet(args.arch)
        self.head = make_head(repr_dim, args.mlp)

    def forward(self, x, y):
        z = self.head(self.backbone(x))
        z_prime = self.head(self.backbone(y))
        return self.objective(z, z_prime)

    def objective(self, z, z_prime):
        # Design the self-supervised objective here.
        raise NotImplementedError
```
