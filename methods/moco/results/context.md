# Context

## Research question

Can we learn general-purpose visual representations *without labels* that rival or
beat supervised ImageNet pretraining as an initialization for downstream tasks
(detection, segmentation)?

The motivation is a striking asymmetry. In natural language, unsupervised
pretraining (word2vec, then GPT and BERT) has become the dominant way to learn
representations: pretrain on raw text, fine-tune on the task. In computer vision,
supervised ImageNet pretraining still dominates and unsupervised methods lag. A
plausible reason lies in the signal itself. Language lives in a **discrete** space —
words and sub-word units — so there is a natural, finite **dictionary** (a
vocabulary), and unsupervised objectives like "predict the masked/next token over
the vocabulary" are well-posed softmax problems. Vision signal is a **continuous,
high-dimensional, unstructured** array of pixels with no pre-existing dictionary.
Any unsupervised vision objective that wants a tokenized comparison set must
therefore *build that dictionary itself*, on the fly, out of the data.

A solution would have to (a) define a self-supervised objective over images that
needs no human labels, (b) produce a representation that transfers — i.e. a
*standard* backbone with no task-specific architectural surgery, so the features
drop straight into detectors and segmenters, and (c) scale to large, even
uncurated, image collections.

## Background

**Contrastive learning as dictionary look-up.** A unifying way to view a family of
recent self-supervised methods: training an encoder to perform a *dictionary
look-up*. There is an encoded query `q = f_q(x^q)` and a set of encoded keys
`{k_0, k_1, k_2, ...}` from a dynamic dictionary. Exactly one key `k_+` matches the
query (e.g. they are two views of the same image); the others are negatives. The
encoder is trained so `q` is similar to `k_+` and dissimilar to the negatives. The
keys are sampled from data and encoded by a network, so the dictionary is *dynamic*:
its entries are random and its encoder evolves during training.

**The contrastive loss lineage.** Hadsell, Chopra & LeCun (2006) introduced the
contrastive loss for dimensionality reduction: a loss on *pairs* of points that
pulls similar pairs together in embedding space and pushes dissimilar pairs apart up
to a margin `m`,
`L = (1−Y)·½·D² + Y·½·max(0, m−D)²`, with `D` the embedding distance and `Y` the
pair label. The conceptual move that matters downstream: the training target is not
a fixed external label but the *data's own representation*, computed on the fly — so
the target can vary during training.

**Noise-contrastive estimation (NCE).** Gutmann & Hyvärinen (2010) estimate an
unnormalized model by turning density estimation into **binary classification**:
data vs. samples from a known noise distribution. This sidesteps an intractable
partition function — the workhorse trick that makes softmax over an enormous number
of classes affordable.

**The InfoNCE objective and its mutual-information bound.** A softmax form of the
contrastive loss over one positive and N−1 negatives:
`L_N = −E[ log( f(x_+, c) / Σ_{x_j∈X} f(x_j, c) ) ]`,
where X is the set of one positive (drawn from `p(x|c)`) and N−1 negatives (from the
marginal `p(x)`). The optimal solution is a **density ratio**: minimizing `L_N`
drives `f(x,c) ∝ p(x|c)/p(x)`. The optimal-classifier algebra
`p(d=i|X,c) = (p(x_i|c)/p(x_i)) / Σ_j (p(x_j|c)/p(x_j))` makes this exact. From it
one derives a lower bound on mutual information,
`I(x, c) ≥ log N − L_N`,
so a *larger* set of negatives N gives a *tighter* bound — a precise argument that
more negatives should help.

**Two random views as the positive pair.** A simple, transfer-friendly way to form
positives: take two random data augmentations (crops, color jitter, flips,
grayscale) of the same image; they are a positive pair, every other image is a
negative.

**Large-batch optimization.** Goyal et al. (2017) showed large-batch SGD can train
ImageNet fast using a linear learning-rate scaling rule plus warmup. But large-batch
optimization remains delicate: without the scaling rule accuracy drops (~2% at batch
1024), and it is unclear the trend extrapolates to much larger batches even with
enough memory. So any approach that obtains many negatives *by enlarging the batch*
inherits all of large-batch optimization's open difficulties.

**Diagnostic observations about existing systems.** Two empirical facts about the
prior art set up the problem. First, methods using contrastive losses consistently
improve as the number of negatives grows — observed under the memory-bank mechanism
and consistent with the InfoNCE MI bound. Second, the negatives' *encoder
consistency* matters: when the comparison set is built from feature vectors that
were computed at many different, stale encoder states (as in a per-sample feature
table updated once per epoch), the representation quality is measurably worse than
when the keys share an encoder.

## Baselines

**Instance discrimination with a memory bank** (Wu, Xiong, Yu & Lin, 2018). Pretext
task: treat *every image as its own class*. A non-parametric softmax over instances,
`P(i|v) = exp(v_i^T v / τ) / Σ_{j=1}^{n} exp(v_j^T v / τ)`,
with temperature τ. The sum over all n images (≈1.28M) is intractable, so it is
approximated with NCE: uniform noise `P_n = 1/n`, posterior
`h(i,v) = P(i|v) / (P(i|v) + m·P_n(i))`, objective
`J = −E_{Pd}[log h] − m·E_{Pn}[log(1 − h)]`.
The negatives come from a **memory bank**: an N×128 table holding one feature vector
per dataset image. Each step samples negative rows from the bank — no extra forward
pass — so the effective comparison set is the whole dataset (very large). After a
feature is computed it is written back with a per-sample update
`v_i ← (1−λ) v_i + λ f_θ(x_i)`, plus a proximal term `λ‖v_i^(t) − v_i^(t−1)‖²` to
damp drift. **Gap:** a bank entry was last written the previous time *that specific
image* was sampled — up to a whole epoch ago, by a very different encoder state. The
keys in any one step therefore come from encoders scattered across the entire past
epoch and are mutually *inconsistent*. The bank's own momentum acts on the stored
*features of a sample*, not on the *encoder*, so it does not address cross-key
consistency. The table also stores all N samples, which does not scale to
billion-image data.

**End-to-end contrastive learning.** Both the query and key encoders are updated by
back-propagation, and the keys are the *other samples in the current minibatch*. The
keys are then perfectly consistent — all encoded by the current encoder. **Gap:**
the dictionary size equals the minibatch size, so it is capped by GPU memory.
Enlarging it means large-batch training, which is itself an open problem (needs the
linear scaling rule; accuracy degrades without it; extrapolation is uncertain). Some
variants enlarge the comparison set via many spatial positions, but those require
patchifying the input or customizing receptive fields, which complicates
transferring the backbone to downstream tasks.

**Earlier pretext-task methods** (relative patch position, jigsaw, colorization,
rotation prediction, clustering, exemplar). These define a hand-crafted surrogate
task and predict a fixed target with cross-entropy or margin losses. They establish
the protocol but generally trail contrastive methods on the linear-classification
yardstick and often rely on task-specific architectures.

## Evaluation settings

- **Unsupervised pretraining data.** ImageNet-1M (the ImageNet-1K training images,
  ≈1.28M images, labels unused), well-balanced and iconic. For scale/uncuratedness,
  a ≈1-billion-image Instagram set (long-tailed, real-world).
- **Linear classification protocol.** Freeze the unsupervised features; train a
  single supervised linear layer on the global-average-pooled features; report
  1-crop top-1 accuracy on the ImageNet validation set.
- **Transfer protocol.** Fine-tune the pretrained backbone on downstream tasks and
  compare against the supervised-ImageNet-pretrained initialization (and against
  random init) under the *same* schedule and hyper-parameters. Tasks/datasets:
  object detection on PASCAL VOC (Faster R-CNN, R50-C4 / R50-dilated-C5 backbones;
  AP50, COCO-style AP, AP75) and COCO; instance segmentation on COCO and LVIS;
  keypoint and dense-pose on COCO; semantic segmentation on Cityscapes and VOC;
  fine-grained classification on iNaturalist. Feature normalization during
  fine-tuning (synchronized BN tuned rather than frozen) is used to reconcile
  feature-magnitude differences between unsupervised and supervised pretraining.
- **kNN monitor.** A k-nearest-neighbor classifier on frozen features as a quick
  validation signal during pretraining.

## Code framework

The pre-existing primitives: a deep CNN backbone (ResNet) that ends in a global
average pool plus a final linear layer producing a fixed-dimensional vector; L2
normalization; a data pipeline producing two augmented views of each image;
softmax cross-entropy; and an SGD training loop. The contrastive objective itself
is a softmax over one positive key and many negative keys. What is not yet
decided is the single empty slot below: how to obtain the keys used in that
comparison.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.07):
        super().__init__()
        self.T = T
        self.encoder = base_encoder(num_classes=dim)
        # TODO: choose the mechanism that supplies:
        #       - one positive key for each query, shape [N, C]
        #       - many negative keys, shape [C, K]
        pass

    def get_keys(self, view_for_key):
        # TODO: fill the key-supply mechanism.
        raise NotImplementedError

    def forward(self, im_q, im_k):
        q = F.normalize(self.encoder(im_q), dim=1)
        k_pos, k_neg = self.get_keys(im_k)
        l_pos = torch.einsum("nc,nc->n", [q, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, k_neg])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, target


def train_loop(model, loader):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03,
                                momentum=0.9, weight_decay=1e-4)
    for images in loader:          # images = [view_q, view_k]
        logits, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
