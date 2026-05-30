# Context

## Research question

Can we learn high-quality visual representations from a collection of images that carry **no labels**, using a procedure simple enough to run unchanged on both convolutional networks and the newly emerging patch-based Transformer image encoders?

The motivation comes from a gap between modalities. In language, the decisive ingredient has been self-supervised pretraining — masked-token prediction or autoregressive language modeling — which extracts a far richer learning signal from a sentence than a single category label would. In vision, the dominant pretraining signal is still a single label per image drawn from a fixed inventory of a few thousand categories, which compresses everything an image contains into one concept. Patch-based Transformer encoders, in particular, are data-hungry and have so far been competitive only when pretrained on very large *labeled* collections or guided by a separately trained convnet; trained with ordinary supervision on a mid-sized labeled set, their features show no especially useful structure. A label-free objective that produced strong, transferable features — and did so without bolting architecture-specific machinery onto the encoder — would matter both for reducing dependence on annotation and for unlocking these encoders.

The hard part is specific. The most attractive label-free objectives ask a network to map two transformed views of the same image to *similar* outputs. Taken alone, "make matching views agree" has a trivial optimum: map every image to the same constant vector. This degenerate solution — **representation collapse** — is the central obstacle, and every method in this space is characterized mainly by the device it uses to avoid it.

## Background

**Siamese view-matching.** The shared skeleton across recent label-free methods is a Siamese setup: take an image, apply two random augmentations (cropping, color jitter, blur), encode both, and train so that the two embeddings agree. Augmentations define what "the same image" means; the encoder learns features invariant to them. The methods differ almost entirely in *how they stop collapse* and in *what target* the matching is against.

**Anti-collapse devices, and the diagnostic facts about each.** Three families had emerged:

- *Contrastive / instance discrimination.* Treat each image as its own class. Pull the two views of an image together and push views of *different* images (negatives) apart, typically with a noise-contrastive / InfoNCE objective. The negatives in the denominator make collapse impossible by construction — a constant encoder would score every pair identically and incur huge loss. The reported cost is that you need *many* negatives at once, which forces large batches or an auxiliary memory store of past embeddings; and keeping those stored embeddings consistent as the encoder drifts motivated a slowly-updated copy of the encoder.

- *Clustering with balanced assignment.* Assign features to a set of learned prototypes and require the assignments to be *balanced* across a batch (an equipartition constraint solved with an iterative optimal-transport / Sinkhorn-Knopp normalization), then predict one view's assignment from the other. Balance forbids any single prototype from absorbing all images, so no dimension can dominate. The diagnostic caveat: the balancing couples the samples within a batch and runs an iterative normalization.

- *Negative-free regression to a moving target.* Drop negatives entirely; regress one view's output onto the *other* view's output, where the target comes from a slowly-updated copy of the network and the gradient is stopped on the target side. The puzzle that this raises — why a "make views agree" objective with no negatives does not simply collapse to a constant — is itself a load-bearing fact: the empirically reported answer is that an *asymmetry* (an extra prediction head on only one branch) together with the slow-moving target keeps the system off the trivial solution. A related observation under debate at the time was a dependence on batch normalization inside the heads.

**Knowledge distillation and self-training (a different lineage).** Separately, knowledge distillation trains a *student* network to reproduce the softened output distribution of a *fixed, pretrained teacher*, using a temperature softmax and a cross-entropy match over the output classes — originally a tool for compressing a big model into a small one. Self-training generalizes this: a teacher produces *soft pseudo-labels* on unlabeled data that a student learns from. Both presuppose a teacher that already exists. A separate, well-established observation is that *averaging a model's weights over training iterations* (Polyak–Ruppert averaging) yields a model better than any single iterate, and that a "mean teacher" formed this way — an exponential moving average of the student's weights — can supply consistency targets in semi-supervised learning, with the averaged teacher tending to lead the student in quality.

**The emerging encoder.** The patch-based Transformer image encoder splits an image into a grid of fixed-size patches, linearly embeds them into tokens, prepends an extra learnable aggregation token, and processes the sequence with a stack of pre-norm self-attention + feed-forward blocks; the aggregation token's final state is the image representation. Unlike convnets it uses no batch normalization by default, and smaller patches yield longer token sequences (more compute, finer features).

## Baselines

- **Instance-discrimination with a memory bank / momentum encoder (Wu et al. 2018; He et al. 2020, "MoCo").** Each image is its own class; a noise-contrastive estimator compares a query view against a positive and a large set of negatives held in a queue/memory bank. MoCo encodes the keys with a *momentum encoder* — a copy of the query encoder updated as an exponential moving average, `θ_k ← λ θ_k + (1−λ) θ_q`, with stop-gradient — so the queued keys stay consistent as the query encoder moves. Gap: needs many negatives (large queue), and the momentum encoder is introduced only as a consistency device for the queue, not as a quality-leading target.

- **Large-batch contrastive (Chen et al. 2020, "SimCLR").** Same contrastive idea without a memory bank; gets its negatives from very large batches, and adds a projection head (a small MLP on top of the backbone, discarded at evaluation). Gap: very large batch requirement.

- **Online clustering with balanced assignment (Caron et al. 2020, "SwAV").** Maps features to `K` prototypes, computes balanced soft assignments ("codes") via the Sinkhorn–Knopp algorithm, and does a swapped prediction (predict one view's code from the other view). Introduces **multi-crop**: instead of two equal crops, use two high-resolution *global* crops plus several low-resolution *local* crops, comparing many views cheaply; this alone is a sizeable accuracy boost. Uses a projection head ending in a prototype layer. Gap: the Sinkhorn balancing couples samples within a batch and is iterative; the clustering machinery is heavier than "just match the outputs".

- **Negative-free regression to a momentum target (Grill et al. 2020, "BYOL").** Online network = backbone + projection head + an extra *predictor* head; target network = an EMA of the online network with stop-gradient. The online predictor regresses (mean-squared error on ℓ2-normalized embeddings) the target's projection of another view. No negatives. Gap / open puzzle: it avoids collapse only because of the predictor asymmetry plus the EMA target (a follow-up, Chen & He 2020, "SimSiam", shows the predictor + stop-gradient suffice even without the EMA); it is architecturally *asymmetric*; its objective is a feature-space regression with no probabilistic reading; and it was reported to rely on batch normalization in the heads (Richemond et al. 2020 replace BN with group norm + weight standardization to keep it working).

- **Mean-teacher consistency (Tarvainen & Valpola 2017).** In semi-supervised learning, a teacher whose weights are an EMA of the student supplies prediction targets the student is trained to be consistent with; weight-averaging makes the teacher a running ensemble that tends to outperform the student. **Codistillation (Anil et al. 2018)** has same-architecture peers distill from *each other* during training (bidirectional), as opposed to one network being the weight-average of the other.

## Evaluation settings

- **Pretraining data.** A large unlabeled image collection (ImageNet-1k images, labels withheld during pretraining).
- **Transfer probes on frozen features.** (1) *Linear probe*: freeze the backbone, train a supervised linear classifier on top, report top-1 accuracy. (2) *Weighted k-nearest-neighbor*: freeze the backbone, store training-set features, classify a query by weighted vote over its `k` nearest stored features (e.g. `k = 20`, cosine similarity with temperature). The k-NN probe needs no training, no augmentation, and no hyperparameter sweep — a clean, low-variance read on feature quality. Also low-shot variants (1% / 10% of labels) and transfer to other classification datasets.
- **Diagnostic measurements used during development.** Track the *entropy* of the teacher's output distribution and the *KL divergence* between teacher and student outputs over training — these detect and distinguish the collapse modes. Compare a teacher's accuracy against the student's throughout training. Vary batch size, output dimensionality, head depth, augmentation scale ranges, and the temperature/momentum hyperparameters to see which settings collapse.

## Code framework

The primitives below already exist: a multi-resolution augmentation pipeline, a generic backbone, a standard optimizer with learning-rate / weight-decay schedules, and a Siamese train loop that encodes two (or more) views and matches them. What does **not** yet exist — the projection/output head, the matching objective and its anti-collapse mechanism, and the rule for forming the second ("target") network — is left as empty slots.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewAugmentation:
    """Produce several augmented views of one image (some larger, some smaller)."""
    def __init__(self, ...):
        # known augmentations: random resized crop, flip, color jitter, blur, solarization
        ...

    def __call__(self, image):
        views = []
        # TODO: build the set of views the objective will match across
        return views


class Head(nn.Module):
    """Maps backbone features to whatever space the matching objective lives in."""
    def __init__(self, in_dim, ...):
        super().__init__()
        # TODO: the head we will design (its layers, normalization, output space)
        pass

    def forward(self, x):
        # TODO
        pass


class ViewWrapper(nn.Module):
    """Run a backbone over a list of views (grouping equal resolutions), then the head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, views):
        # one backbone forward per distinct resolution, then head on concatenated features
        ...


class MatchingObjective(nn.Module):
    """Compare outputs across views and produce the training loss."""
    def __init__(self, ...):
        super().__init__()
        # TODO: the matching loss, plus whatever device prevents collapse
        pass

    def forward(self, online_outputs, target_outputs):
        # TODO: match target's output on one view to online's output on a different view
        pass


def build_target_network(online_net):
    # TODO: how the target/second network is formed and updated
    pass


def train_one_epoch(online_net, target_net, objective, data_loader, optimizer, schedules):
    for views, _ in data_loader:
        views = [v.cuda() for v in views]
        target_outputs = target_net(...)    # TODO: which views go to the target
        online_outputs = online_net(views)
        loss = objective(online_outputs, target_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: update the target network from the online network
```
