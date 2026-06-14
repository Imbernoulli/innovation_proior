## Research question

The expressive power of a neural network grows roughly exponentially with its depth, but its
trainability does not. Stacking more width-preserving layers `x_{i+1} = F[W_i](x_i)` should, in
principle, only ever help — a deeper model can always represent everything a shallower one can by
making the extra layers do nothing — yet in practice past a few dozen layers the optimizer simply
fails: the loss plateaus, diverges, or needs exotic schedules to move at all. The core mechanism is
signal propagation. If a small perturbation of the input is rescaled by a factor `r` as it passes
through one layer, then after `L` layers it has been rescaled by `r^L`; the same multiplicative
factor governs the backward pass, so for `r > 1` gradients explode like `r^L` and for `r < 1` they
vanish like `r^L`. Only `r ≈ 1`, sustained across all `L` layers, keeps both signals and gradients
alive at large depth.

The precise goal is a mechanism that keeps signal and gradient magnitudes stable at large depth,
lets optimization begin from a well-conditioned regime rather than an unstable early phase, is cheap
enough to drop into fully-connected nets, convolutional ResNets, or Transformers without
per-architecture retuning, and reduces the scaffolding (careful per-layer initialization recipes,
normalization layers, learning-rate warm-up) that the field had accumulated to make deep training
work at all. Each prior approach below buys a subset of these and pays for it somewhere. Closing
that gap is the problem.

## Background

By this time the dominant way to make depth trainable falls into three families — careful
initialization, normalization, and residual connections — often stacked together. The relevant
theory and the relevant pain points:

**The vanishing/exploding gradient problem and the `r^L` picture.** A deep network propagates a
signal `x_0` of width `w` through `L` width-preserving functions `F[W_i] : R^w -> R^w`. Both the
forward signal and the backward gradient are scaled multiplicatively at each layer, so a per-layer
factor `r` compounds to `r^L`. This is why naive deep stacks are untrainable and why every fix below
is ultimately trying to hold the per-layer factor near one.

**Signal propagation / mean field theory.** A line of work analyzing randomly initialized deep and
wide networks (Poole et al. 2016; Schoenholz et al. 2016; Pennington et al. 2017, 2018; Xiao et al.
2018) showed that the cosine distance between two distinct input signals,
`x_i · x'_i / (||x_i|| ||x'_i||)`, flows to a fixed point as it propagates with depth. If that fixed
point is 1 the network is in a *stable/ordered* phase: every input is mapped to essentially the same
output, so the output carries no information about which input it came from and weight-update
gradients vanish. If the fixed point is 0 the network is in a *chaotic* phase: arbitrarily similar
inputs are driven to very different outputs, and gradients explode. Trainability lives on the
boundary between them — the *edge of chaos*.

**The input-output Jacobian and dynamical isometry.** To diagnose which phase a network is in, one
studies the input-output Jacobian `J_io ≡ ∂x_L / ∂x_0`. The mean squared singular value `χ` of
`J_io` measures the average growth/decay of an input perturbation as it crosses the network; the
edge of chaos is `χ ≈ 1`, where on average a signal is neither amplified nor attenuated. This is the
condition that motivated the classic Glorot (2010) and He (2015) weight rescalings. Pennington et
al. (2017, 2018) then showed that `χ ≈ 1` *on average* is not enough: if the singular vectors of
`J_io` belonging to the very large and very small singular values happen to align with the data's
perturbations, training is still inefficient. They proposed the stronger condition of *dynamical
isometry* (Saxe et al. 2013): **all** singular values of `J_io` should be close to 1, so that every
perturbation of the input propagates through the network equally well. A consequential negative
result from the same work: a network whose layers use the ReLU activation cannot satisfy dynamical
isometry, because ReLU maps some perturbations to zero — and for some activations elaborate
orthogonal initialization schemes can recover it, but ReLU and similar nonlinearities are out of
reach by initialization alone.

**Why Transformers are especially hard.** Two components of a Transformer layer are individually
hostile to dynamical isometry. LayerNorm, `LayerNorm(x) = (x - E[x]) / sqrt(Var(x)) · γ + β`, is by
construction invariant to perturbations that purely shift the mean or rescale the variance of its
input; each such invariance is a zero singular value of the layer Jacobian, so applied across `n`
sequence elements LayerNorm contributes `2n` vanishing singular values per Transformer layer.
Self-attention, `softmax(Q K^T / sqrt(d)) · V`, can collapse too: in the small-score regime the
softmax returns a nearly uniform `n × n` matrix with all entries ≈ `1/n`, which
averages all `n` embedding vectors onto a single direction, leaving only `d` of the `n × d` Jacobian
singular values nonzero and discarding most of the input signal. A residual connection restores some
of it but still amplifies some directions and attenuates others. Empirically, the input-output
Jacobian of a deep Transformer encoder at initialization has a large fraction of its singular values
decayed to machine precision, consistent with the well-known difficulty of training deep
Transformers.

**Why the default Transformer needs warm-up.** Mean-field analysis of the original (Post-LN)
Transformer (Xiong et al. 2020) shows that at initialization the expected gradients of the
parameters near the output layer are large, so applying a large learning rate immediately makes
training unstable — which is exactly why the original Transformer recipe prescribes a learning-rate
warm-up. Moving the normalization inside the residual branch (Pre-LN) makes the initial gradients
well-behaved and lets warm-up be dropped, with its own tradeoffs. Either way, the
warm-up/normalization-placement machinery exists because the init-time residual stream is not yet a
well-conditioned object on its own.

## Baselines

The prior methods a new residual-stream design would be measured against and would react to.

**Plain deep network, `x_{i+1} = F(x_i)`.** No shortcut at all; the `r^L` factor is unmitigated.
The diagnostic finding that frames the whole area (He et al. 2016): a deeper *plain* network has
*higher training error* than a shallower one — not a generalization gap but an optimization failure,
since the deeper model could in principle copy the shallower one and set the extra layers to the
identity, yet the optimizer cannot find that solution.

**ResNet, `x_{i+1} = σ(x_i + F(x_i))` (He et al. 2016).** Add an identity shortcut around each
block and let `F` learn a residual. The rationale: if the optimal map for a block is close to the
identity, it is easier for the optimizer to push the residual `F` toward zero than to fit the
identity out of a stack of nonlinear layers. This made hundreds of convolutional layers trainable.
Pre-activation ResNet (He et al. 2016b) moves the activation before the addition so the shortcut
carries a cleaner, unmodulated signal. **Gap:** the residual branch contributes at full strength at
initialization — the block is *not* the identity at step 0 — and with standard initialization a
residual block's output variance grows by roughly a constant factor per layer, so the output
variance still compounds with depth; the shortcut tames the problem for moderate depth but does not
make the block start as a true identity.

**Highway Networks, `x_{i+1} = C(x)·x_i + T(x)·F(x_i)` (Srivastava et al. 2015).** The first
feed-forward nets with hundreds of layers; learned, *data-dependent* transform and carry gates
`T = σ(W_T^T x + b_T)`, `C = 1 - T`, with `b_T` initialized negative to bias toward carrying the
signal. Gated ResNet (Savarese et al. 2016) simplifies this to a single scalar gate by setting
`W_T = 0`, `b_T = α`, and `C = 1 - T`. **Gap:** Highway's gates add input-dependent parameters and
compute, while the scalar gated-residual variant removes that dependence but still keeps a sigmoid
transform/carry split; a finite negative bias can lean toward carrying the signal, but it does not
make the residual transformation exactly zero and the carry path exactly one at initialization.

**Zero-`γ` (Goyal et al. 2017; Hardt & Ma 2016; He et al. 2019).** When the last operation inside a
residual branch is a normalization layer with a learnable scale `γ`, initialize that `γ` to zero, so
the branch outputs zero at the start and the block begins as the identity. **Gap:** it applies only
when such a normalization layer exists, and it zero-initializes many parameters per block (one per
channel), so the mechanism is tied to normalization rather than to the residual stream itself.

**FixUp initialization (Zhang et al. 2019).** A normalization-free recipe derived top-down from the
requirement that one SGD step change the network *function* by an amount that is `Θ(η)` and
independent of depth. Its analysis shows the gradient norm of certain layers in a standard-init
residual net is lower-bounded by a quantity that grows with depth, so the network's logits blow up
at initialization. The fix is several coordinated pieces: zero-initialize the last layer of each
residual branch and the classifier; scale the weight layers inside a residual branch of `m` layers
by `L^{-1/(2m-2)}`; and add a per-branch scalar multiplier (initialized at 1) plus scalar biases
(initialized at 0). **Gap:** it is an elaborate, architecture-aware initialization — the branch
rescaling exponent depends on how many layers each branch has — and its scalar multiplier starts at
1, so the multiplier is not the part that suppresses the residual branch at initialization.

**The Transformer residual placements.** The original Post-LN block
`x_{i+1} = LayerNorm(x_i + sublayer(x_i))` (Vaswani et al. 2017) requires learning-rate warm-up and
becomes hard to train much beyond a dozen layers without large compute or auxiliary intermediate
losses (Al-Rfou et al. 2019). Pre-LN, `x_{i+1} = x_i + sublayer(LayerNorm(x_i))` (Xiong et al.
2020), removes the warm-up requirement by moving the normalization before the sublayer. The GPT-2
placement `x_{i+1} = x_i + Norm(F(x_i))` (Radford et al. 2019) is another point in this space.
**Gap:** all of these carry the cost and Jacobian effects of a normalization layer in every
sublayer, and the residual branch still contributes a full learned transformation at initialization.

Together these baselines leave a tension: the general tools still require normalization, warm-up, or
careful initialization to keep very deep signal propagation under control, while the more
identity-biased tools are tied to particular architectures, normalization placements, or multi-part
recipes.

## Evaluation settings

The natural pre-existing yardsticks for "does this make deep nets trainable and fast":

- **Deep fully-connected classification on CIFAR-10.** Stacks of fully-connected ReLU layers (e.g.
  32 layers, width 256), trained only to fit the training data; the metric is how many iterations to
  fit, comparing a plain net, a residual net, a normalized net, and the candidate. Initial weights
  drawn `~N(0, 2/w)` (the He variance). A stress test pushing depth to thousands of layers to ask
  whether training is possible at all.
- **Convolutional ResNets on CIFAR-10** (ResNet-56, ResNet-110, PreAct-ResNet-18/50). Metrics:
  validation error, epochs to reach a target accuracy (e.g. 80%), and final training loss. Standard
  step-down schedule (SGD, momentum 0.9, weight decay `5e-4`, batch 128, LR 0.1 decayed ×10 at
  epochs 100 and 150 over 200 epochs); also a one-cycle "superconvergence" schedule. The yardstick
  for whether a method speeds up training without sacrificing the regularization that normalization
  provides.
- **Character-level language modeling on enwiki8** (the Hutter Prize text). A 12-layer Transformer
  decoder; metric is the number of training iterations to reach a target bits-per-byte (e.g. 1.2
  BPB) on the validation set, and final BPB on the test set. The natural setting to compare
  residual/normalization placements head to head (Post-LN with and without warm-up, Pre-LN, the
  GPT-2 placement). Word/sub-word language modeling on WikiText-2/WikiText-103 is the companion
  benchmark.
- **Depth scaling for Transformers.** Take a Transformer that trains at 12 layers and scale it to
  substantially greater depth, asking whether it still converges at all, at fixed or memory-reduced
  hidden size and adjusted batch size. Optimizers in use include SGD/Adam for vision and large-batch
  optimizers (LAMB) for the Transformers; some setups deliberately omit learning-rate schedules to
  isolate the architectural effect.
- **Diagnostic, not a benchmark:** the histogram of input-output Jacobian singular values at
  initialization and during training, used to read off directly whether a network sits near
  dynamical isometry.

## Code framework

The substrate is a standard Transformer encoder layer. The pieces that already exist are
multi-head self-attention, the point-wise feed-forward network, dropout, and a residual stream that
must keep the same tensor shape after each sublayer. What is not settled is how each sublayer output
should re-enter the residual stream. The scaffold leaves those residual-stream updates as the empty
slots.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


class CandidateEncoderLayer(nn.Module):
    """One Transformer encoder layer. The attention and feed-forward sublayers
    already exist; the residual-stream update rule is the slot to design."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        # TODO: any residual-stream state required by the design.

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # TODO: combine the attention output with the residual stream.
        src = src

        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # TODO: combine the feed-forward output with the residual stream.
        src = src
        return src
```

The residual-stream state and the two combine steps are the slots the design will fill.
