# Context

## Research Question

We want to learn visual representations from images without labels. The common shape is a Siamese network: take one image, draw two random augmentations, encode both views, and make the two outputs agree. The obstacle is collapse. If every input maps to one constant vector, the two views agree perfectly while the representation contains no usable information.

The question is how little machinery is actually needed to avoid that failure. Existing successful systems add explicit repulsion, online clustering, a second slowly updated encoder, or large-batch infrastructure. Are those components intrinsically necessary, or can a direct shared-weight Siamese network learn nontrivial image features if the right minimal asymmetry is present?

## Background

A Siamese network applies weight-sharing functions to multiple inputs and compares the outputs. In self-supervised vision, the inputs are two augmentations of the same image, so the objective asks the representation to become invariant to the augmentation distribution. Collapse is the degenerate fixed point: agreement is maximized because all samples receive the same code.

Contrastive learning prevents that fixed point by using negative pairs. Two views of one image are pulled together, and views from different images are pushed apart. If every image mapped to one point, the negative terms would be violated. This makes the number and quality of negatives central.

Clustering-based self-supervision avoids pairwise negatives but introduces another anti-collapse mechanism. It assigns images to prototypes or clusters and prevents all samples from choosing the same cluster through balanced assignment constraints or repeated re-clustering.

Momentum-target methods remove explicit negatives and instead make an online network predict targets produced by a second network. The target network changes slowly, so the prediction target is more stable than the online branch. A prediction MLP on the online side is part of this recipe, but the moving-average target and the target-side training rule are coupled together.

Batch normalization, projection heads, predictor heads, and strong augmentations are also in the design space. BN can stabilize deep optimization. Projection heads improve the representation learned before the loss. Predictor heads introduce an asymmetry between the two branches. Augmentations define the invariances being learned.

Alternating optimization is a separate classical tool. When an objective has two blocks of variables, one can fix one block while optimizing the other, then swap. k-means alternates assignments and cluster centers; EM alternates latent-variable estimates and model parameters.

## Baselines

**Contrastive loss / DrLIM.** Hadsell, Chopra, and LeCun formulate a Siamese-style loss with similar pairs attracted and dissimilar pairs repelled by a margin. The dissimilar term prevents the all-constant solution.

**SimCLR.** A direct weight-sharing Siamese network uses two augmented views per image and the NT-Xent loss. Other views in the minibatch act as negatives. A nonlinear projection head before the contrastive loss improves representation quality. The recipe benefits from very large batches, commonly 4096 examples, and LARS.

**MoCo.** A query encoder is trained with a contrastive loss against a large dictionary of keys. The dictionary is maintained as a queue, and a momentum encoder keeps queued keys relatively consistent. The queue decouples the negative set size from the current minibatch size.

**DeepCluster and SwAV.** DeepCluster alternates between clustering image features and training a network to predict the cluster assignments. SwAV makes clustering online by predicting assignments between views and computing batch assignments with a Sinkhorn-Knopp balanced partition. The assignment machinery is the nontrivial anti-collapse device.

**BYOL.** An online branch predicts the projection of a target branch for another augmented view. The target branch uses a moving average of the online parameters, and the prediction MLP is applied only on the online branch. BYOL shows that explicit negatives are not required, but leaves open which part of the target-side machinery is doing the essential work.

## Evaluation Settings

The standard pretraining data is the ImageNet training set without labels. Representation quality is measured by freezing the pretrained backbone, taking features from the global average pooling layer, training a supervised linear classifier on ImageNet train, and reporting top-1 accuracy on ImageNet validation.

During pretraining, a k-nearest-neighbor classifier on frozen features can monitor progress cheaply. Collapse is diagnosed by both the loss curve and the per-channel standard deviation of the l2-normalized outputs. A collapsed constant representation has per-channel std near zero; a scattered unit-sphere representation is near the isotropic reference value \(1/\sqrt{d}\) for output dimension \(d\).

Transfer quality is checked by fine-tuning the pretrained backbone on downstream detection and segmentation tasks, including VOC with Faster R-CNN and COCO with Mask R-CNN.

The common recipe yardsticks are ResNet-50, two 224x224 augmented views, SGD or LARS depending on the method, linear learning-rate scaling by batch size, cosine learning-rate decay, and batch sizes ranging from ordinary minibatches to thousands of images.

## Code Framework

The empty slot is a shared-weight two-view model and a loss. The framework already has a ResNet backbone, MLP heads built from linear layers, batch normalization, ReLU, an l2-normalized cosine comparison, SGD with cosine learning-rate scheduling, and a two-view augmentation pipeline.

```python
import torch
import torch.nn as nn

class TwoCropsTransform:
    """Take one image, return two independently augmented views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

class SiameseModel(nn.Module):
    """A weight-sharing network over two views."""
    def __init__(self, base_encoder, dim=2048):
        super().__init__()
        self.encoder = base_encoder(num_classes=dim)
        # TODO: define the projection/prediction structure.

    def forward(self, x1, x2):
        # TODO: process both views with the shared encoder.
        pass

def loss_fn(*outputs):
    # TODO: define the training objective.
    pass

def train_one_epoch(loader, model, optimizer):
    for images, _ in loader:
        x1, x2 = images[0], images[1]
        outputs = model(x1, x2)
        loss = loss_fn(*outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
