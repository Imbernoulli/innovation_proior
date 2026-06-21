# Research question

Image recognition is owned by convolutional networks. From LeNet through AlexNet to deep residual
networks, every state-of-the-art classifier is a stack of convolutions. Convolution hard-wires three
image-specific assumptions into every layer — **locality** (a unit only sees a small spatial
neighborhood), **two-dimensional neighborhood structure** (the input is a grid and nearby pixels belong
together), and **translation equivariance** (the same filter slides everywhere, so shifting the input
shifts the response). These priors make CNNs sample-efficient: much of the right answer is built in
before any data is seen.

In a neighboring field — natural-language processing — the dominant architecture is a domain-agnostic,
almost prior-free sequence model that scales well: pre-train on an enormous corpus, fine-tune on the small
downstream task, and keep adding parameters and data, with the accuracy curve still rising. This raises a
question for vision. The built-in convolutional priors help when labeled data is scarce; what happens to
the accuracy ceiling if the image-specific inductive bias is removed almost entirely and a standard,
hardware-friendly sequence model is run directly on images, across the small-data and large-data regimes?

The setting, then, is how to apply a standard sequence model to images and study its behavior as a function
of pre-training data scale, judged against convolutional systems at comparable or lower compute.

# Background

**The Transformer and self-attention.** The self-attention layer (Vaswani et al., 2017) maps a sequence
`z ∈ ℝ^{N×D}` to a new sequence. Each element is projected to a query, key, and value via a shared linear map
`[q, k, v] = z·U_qkv`, with `U_qkv ∈ ℝ^{D×3D_h}`. Attention weights are pairwise similarities,
`A = softmax(q·kᵀ / √D_h) ∈ ℝ^{N×N}`, and the output is the weighted sum of values, `SA(z) = A·v`. The factor
`1/√D_h` is load-bearing: if query and key entries are roughly independent with unit variance, the dot product
`q·k` has variance `D_h`, so for large `D_h` the logits grow large in magnitude, the softmax saturates toward a
one-hot distribution, and the gradient through the softmax collapses. Dividing by `√D_h` rescales the logits to
unit variance and keeps the softmax in its high-gradient regime. **Multi-head self-attention (MSA)** runs `k`
such operations in parallel in `D_h = D/k`-dimensional subspaces, concatenates their outputs, and projects back:
`MSA(z) = [SA_1(z); …; SA_k(z)]·U_msa`, `U_msa ∈ ℝ^{kD_h×D}`. Multiple heads let different subspaces attend to
different relations at once (some broad, some narrow), and fixing `D_h = D/k` keeps compute and parameter count
constant as `k` varies. A full Transformer encoder block pairs MSA with a position-wise feed-forward MLP — two
linear layers with a nonlinearity and a hidden width about 4× the model width — each sublayer wrapped in a
residual connection and layer normalization. Attention mixes information *across positions*; the MLP mixes
*across channels* within each position.

Self-attention is **permutation-equivariant**: it sees a set, not an ordered sequence, so order is supplied
explicitly. The original Transformer added fixed sinusoidal position signals; encoder-style pre-trained models
(Devlin et al., 2019) instead used a table of **learned position embeddings**, one per position, added to the
token embeddings. The cost of self-attention is `O(N²·D)` — quadratic in sequence length.

**The pretrain/fine-tune recipe and the [class] token.** Large encoder models (Devlin et al., 2019) established
the template the field now uses: pre-train on huge data, then fine-tune. They also introduced a way to get a
single vector out of a sequence model for classification — prepend one extra learnable token to the input; its
state at the output is taken as the pooled representation and fed to a classifier head. Their Base and Large
configurations (12 layers / width 768 / 12 heads; 24 / 1024 / 16) became standard sizes, and their position
embeddings were initialized from a small normal distribution.

**Where to put the normalization.** In the original post-normalization Transformer (layer norm applied *after*
the residual add), gradients near the output are large at initialization, which makes large learning rates
unstable and requires careful warmup (Xiong et al., 2020). Applying layer norm *inside* the residual branch — to
the input of each sublayer (pre-normalization, Wang et al., 2019; Baevski & Auli, 2019) — leaves an unobstructed
residual path and well-behaved gradients at initialization, so the model trains stably at larger learning rates.

**Attention in vision so far.** Several lines of work bring attention to images. One restricts attention to local
neighborhoods so it can stand in for convolution (Parmar et al., 2018; Ramachandran et al., 2019; Hu et al., 2019;
Zhao et al., 2020). Another approximates global attention with sparse patterns (Child et al., 2019) or by attending
along single axes at a time (Ho et al., 2019; Wang et al., 2020); these use specialized attention patterns. A third
line keeps the CNN and adds attention as an augmentation (Wang et al., 2018; Bello et al., 2019; Carion et al.,
2020). One model (Chen et al., 2020, "image GPT") runs a Transformer on raw pixels after shrinking resolution and
color, trained generatively, reaching about 72% on ImageNet by linear probe.

**Self-attention can imitate convolution.** A theoretical result (Cordonnier et al., 2020) shows that a multi-head
self-attention layer with enough heads and a positional encoding can express *any* convolution of a given kernel
size; in that work, `2×2` patches are extracted from the image and full self-attention is applied on top.

**Empirical findings on data scale.** Two facts about *existing* systems frame the setting. First, convolutional
performance grows steadily — roughly logarithmically — with pre-training set size all the way up to
hundred-million-image scale (Sun et al., 2017), with no observed plateau. Second, large *supervised* pre-training on
`ImageNet-21k` (14M images) and `JFT-300M` (303M images) followed by simple transfer already beats more elaborate
recipes (Kolesnikov et al., 2020). So the substrate a scale-hungry model would use — very large labeled datasets and
a clean transfer protocol — already exists. The inductive-bias trade-off is itself an observable phenomenon: priors
that help in the small-data regime are progressively outweighed as data grows and the model can learn those
regularities directly.

# Baselines

**Big Transfer / large ResNets (Kolesnikov et al., 2020).** The strongest convolutional transfer recipe of the
time: pre-train large residual networks (with Group Normalization replacing Batch Normalization, and weight
standardization on the convolutions, which improve transfer) on `ImageNet-21k` or `JFT-300M`, then fine-tune on
each downstream task with a simple, fixed protocol, often at higher resolution than pre-training. Its core idea is
that scale plus a disciplined transfer recipe beats architectural cleverness.

**Large EfficientNet trained with extra data (Xie et al., 2020).** A scaled convolutional network trained
semi-supervisedly using a large unlabeled corpus, holding the state of the art on ImageNet.

**The hybrid option — CNN features feeding a Transformer.** An intermediate baseline: run a convolutional stem
first and feed its feature map (rather than the raw image) into a Transformer. This keeps convolutional locality
at the front while letting attention mix globally afterward, and is the comparison point for asking how much the
convolutional front-end contributes.

**Earlier attention-in-vision models** (local/stand-alone self-attention, sparse and axial attention, attention-
augmented CNNs; see Background) form the prior art a pure-Transformer approach is compared against.

# Evaluation settings

Pre-training datasets of increasing size, to probe the data-scale axis: `ImageNet-1k` (ILSVRC-2012, 1.3M images,
1k classes), its superset `ImageNet-21k` (14M images, 21k classes), and `JFT-300M` (303M images, 18k classes).
Pre-training sets are de-duplicated against downstream test sets. To isolate intrinsic model behavior from
regularization, one can also pre-train on random `JFT` subsets (9M / 30M / 90M / full) with shared hyperparameters
and early stopping.

Downstream transfer benchmarks: `ImageNet` validation (and the cleaned-up ReaL labels), `CIFAR-10` and
`CIFAR-100`, `Oxford-IIIT Pets`, `Oxford Flowers-102`, and the 19-task `VTAB` suite (low-data transfer with 1,000
training examples per task, grouped into Natural / Specialized / Structured). Metrics: top-1 fine-tuning accuracy,
and — for fast on-the-fly evaluation — few-shot linear accuracy obtained by closed-form regularized least-squares
regression from frozen representations to one-hot targets. The yardsticks are the convolutional baselines above,
compared on accuracy *and* on pre-training compute (TPU-core-days / exaFLOPs), since compute-matched comparison is
the fair way to judge a scale-driven claim. Fine-tuning is commonly done at higher resolution than pre-training.
Models are optimized with Adam (`β₁=0.9, β₂=0.999`), batch size 4096, high weight decay, linear learning-rate
warmup and decay during pre-training, and SGD with momentum during fine-tuning.

# Code framework

The starting point is a standard sequence-model toolkit: linear layers, layer norm, GELU, dropout, dense
self-attention, residual encoder blocks, image tensors from the input pipeline, and optimizers with warmup. The
image classifier needs a generic slot that converts an image batch into a sequence, feeds a sequence encoder, and
reduces the encoded sequence to class logits.

```python
import torch
from torch import nn

def build_sequence_encoder(dim, depth, heads, mlp_dim, dropout=0.):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        ),
        num_layers=depth,
    )

class ImageSequenceModel(nn.Module):
    def __init__(self, image_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.image_size = image_size
        self.encoder = build_sequence_encoder(dim, depth, heads, mlp_dim, dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def image_to_sequence(self, images):
        # TODO: map a batch of images to (batch, sequence_length, dim).
        pass

    def sequence_to_vector(self, encoded):
        # TODO: reduce the encoded sequence to one vector per image.
        pass

    def forward(self, images):
        tokens = self.image_to_sequence(images)
        encoded = self.encoder(tokens)
        vector = self.sequence_to_vector(encoded)
        return self.classifier(vector)

def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=0.1)

loss = nn.CrossEntropyLoss()
```
