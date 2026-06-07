# Synthesis — Group Normalization (GN)

## Pain point being solved
Batch Normalization normalizes each channel using the mean/variance computed over the
batch axis (and spatial axes): per channel, statistics are taken over (N, H, W). Its
effectiveness therefore depends on the batch size N estimating those statistics well.

Diagnostic finding (about the *existing* system, pre-method): on ResNet-50 / ImageNet,
BN's validation error rises sharply as the per-GPU batch size shrinks — roughly
23.6 / 23.7 / 24.8 / 27.3 / 34.7 % at batch 32 / 16 / 8 / 4 / 2. The statistics are
estimated over fewer samples, so they become noisy/inaccurate. This is intrinsic to
normalizing along the batch dimension.

A second, structural problem with the batch axis: "batch" is not always a meaningful or
stable concept. At inference there is no batch to normalize over, so BN must instead use
population statistics accumulated by a running average during training and freeze them —
introducing a train/inference discrepancy (the function computed at test time differs
from the one during training). When transferring features (detection, segmentation,
video), the batch is forced tiny by memory (high resolution, 3D conv), and BN is often
"frozen" into an affine layer using pre-trained statistics, again inconsistent across
train/transfer/test. Non-i.i.d. batches (e.g. 512 RoIs sampled from one image) further
corrupt the batch-statistics estimate.

What a solution must achieve: a per-layer feature normalization whose computation does
not depend on the batch dimension at all — identical at train and inference, no running
stats, no tiny-batch degradation — while keeping BN's accuracy when batches are large.

## Load-bearing ancestors and where each falls short
- **Batch Norm (Ioffe & Szegedy 2015).** Normalize each channel over (N, H, W); add a
  per-channel affine y = γ x̂ + β to restore representational power. Eases optimization,
  enables very deep nets, and the batch-sampling noise acts as a regularizer. Gap: tied to
  the batch axis → small-batch degradation, train/inference running-stat discrepancy.
- **Layer Norm (Ba et al. 2016).** Normalize over *all* channels of a sample: per sample,
  statistics over (C, H, W). Batch-independent; great for RNN/LSTM. Gap for vision: it
  forces *all* channels to share one mean/variance, i.e. assumes all channels make
  "similar contributions" — questionable for convnets where different filters (edges,
  colors, textures, frequencies) have genuinely different response distributions.
- **Instance Norm (Ulyanov et al. 2016).** Normalize each channel of each sample over
  (H, W) only. Batch-independent; strong for style transfer / generative models. Gap:
  with only spatial pixels per (sample, channel) it throws away all cross-channel
  information — it cannot exploit dependence *between* channels, which limits recognition.
- **Weight Norm (Salimans & Kingma 2016).** Normalize the *filter weights*, not features.
  Batch-independent, but reparameterizes weights rather than controlling the feature
  distribution, and has not matched BN's accuracy on visual recognition.
- **Batch Renorm (Ioffe 2017).** Constrains BN's estimated mean/var to reduce small-batch
  drift; helps but is still batch-dependent and still degrades as batch shrinks.
- **Synchronized BN (Peng et al. 2018).** Computes BN statistics across GPUs; doesn't
  remove the dependence, it just buys a larger effective batch with more hardware, and
  blocks asynchronous solvers.

## Motivation that channels are not independent (grouping prior)
Classical vision features are *group-wise* and use *group-wise normalization*: SIFT (Lowe
2004) and HOG (Dalal & Triggs 2005) build per-cell orientation histograms that are
normalized within each block/orientation group; GIST, VLAD, Fisher Vectors are likewise
grouped sub-vectors. Channels of a convnet are analogous: a filter and its horizontal flip
should have similar response distributions (so could be normalized together); orientation,
frequency, shape, illumination, texture induce natural groupings. Neuroscience divisive
normalization also normalizes across populations of cells with related tuning. So the right
granularity is between "all channels together" (LN, too coarse — channels aren't
interchangeable) and "each channel alone" (IN, too fine — ignores correlation): normalize
within *groups* of channels.

## The unified view (the key derivation)
A whole family of feature normalizers shares one form. With i = (i_N, i_C, i_H, i_W)
indexing (N, C, H, W):

    x̂_i = (x_i − μ_i) / σ_i,
    μ_i = (1/m) Σ_{k∈S_i} x_k,
    σ_i = sqrt( (1/m) Σ_{k∈S_i} (x_k − μ_i)^2 + ε ),

and a per-channel affine y_i = γ x̂_i + β. They differ ONLY in the set S_i of pixels
sharing one mean/variance:
- **BN:** S_i = { k | k_C = i_C }  → reduce over (N, H, W) per channel.
- **LN:** S_i = { k | k_N = i_N }  → reduce over (C, H, W) per sample.
- **IN:** S_i = { k | k_N = i_N, k_C = i_C }  → reduce over (H, W) per sample-channel.
- **GN:** S_i = { k | k_N = i_N, ⌊k_C/(C/G)⌋ = ⌊i_C/(C/G)⌋ }  → reduce over (H, W) and
  over a contiguous group of C/G channels, per sample. G = number of groups (default 32).

Crucially every set with k_N = i_N is batch-independent: BN is the only one whose S_i spans
the batch axis. GN keeps k_N = i_N, so its computation is identical in training and
inference — no running statistics, no train/test discrepancy, no batch-size dependence.

GN interpolates exactly between LN and IN:
- G = 1  → one group = all channels → LN.
- G = C  → one channel per group → IN.
Intermediate G lets each group learn its own distribution (more flexible than LN's single
shared distribution) while still pooling correlated channels (unlike IN, which can't). The
per-channel γ, β are kept so representational power isn't lost.

## Design-decision → why (with rejected alternatives)
| Decision | Why | Rejected alternative & its failure |
|---|---|---|
| Don't normalize over the batch axis | removes the root cause of small-batch noise and the train/inference running-stat discrepancy | keep BN → degrades at small batch, inconsistent at test/transfer |
| Normalize over a *group* of channels per sample | channels are correlated but not interchangeable; grouping matches classical group-wise normalization (HOG/SIFT orientations) | LN (G=1): all channels share one μ,σ → too coarse for convnets; IN (G=C): one channel alone → ignores channel correlation |
| Keep spatial axes (H,W) in S_i always | a single channel-group on one image still has H·W·(C/G) pixels → enough samples for a stable estimate without any batch | reducing only over channels would give too few samples and lose spatial pooling |
| Per-channel affine γ, β (γ init 1, β init 0; γ init 0 at block end) | restores representational power lost to normalization; zero-γ at residual-block end makes initial block ≈ identity | drop affine → constrains features |
| Default G = 32 groups (not C/G fixed) | empirically flat over a wide G range; G fixed per layer is simplest; both "fix G" and "fix C/G" work, fix-G chosen as default | extremes G=1 (LN) and G=C (IN) are worst in the sweep |
| Implementation = reshape (N,C,H,W)→(N,G,C/G,H,W), reduce over (2,3,4), reshape back | exposes the group axis so the same `moments` primitive computes group statistics; a few lines, autodiff handles backward | hand-deriving group statistics is unnecessary given reshape |

## Canonical code (Phase 1.4)
- Original TensorFlow snippet: reshape to [N,G,C//G,H,W]; `tf.nn.moments(x,[2,3,4])`;
  normalize; reshape back; `* gamma + beta`. (BN is the same with `moments(x,[0,2,3])`.)
- `torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)`: separates channels
  into groups, computes mean/std per group, biased variance, **same statistics in train and
  eval** (no running buffers), per-channel affine weight/bias of size num_channels.
- Verified numerically (numpy): G=1 == LN over (C,H,W); G=C == IN over (H,W); the floor
  grouping ⌊c/(C/G)⌋ gives contiguous channel blocks identical to the reshape; reductions
  never touch the batch axis.

## Out of scope (proposed-method results — excluded from context/reasoning)
GN-vs-BN ImageNet/COCO/Kinetics win numbers, the BR comparison numbers, the group-division
ablation outcomes. (The BN small-batch *degradation* curve IS a diagnostic finding about
the existing system and is in scope as context.)
