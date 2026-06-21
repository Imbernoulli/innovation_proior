# Context: multivariate time-series forecasting on networked sensors (circa 2019-2020)

## Research question

We are given a collection of `N` time-series that are not independent — traffic-speed sensors on
adjacent highway segments, electricity meters across a grid, daily case counts across countries.
Stack them into a matrix `X ∈ R^{N×T}`: row `i` is one series, column `t` is the snapshot of all
`N` series at timestamp `t`. From a window of the last `K` timestamps `X_{t-K},…,X_{t-1}` we must
forecast the next `H` timestamps `X_t,…,X_{t+H-1}` for every series at once.

Accuracy here depends on modeling two things together. There is **intra-series** structure: each
series has its own temporal dynamics — trend, periodicity (a daily rush-hour cycle), autocorrelation.
And there is **inter-series** structure: the series interact — congestion on one road segment
propagates to the next a few minutes later, so series `i`'s future depends on series `j`'s recent
past. The question is how to build a single model that captures both the temporal patterns of each
individual series and the correlations across the full set.

## Background

**Temporal modeling of a single series.** Recurrent nets — LSTM (Hochreiter & Schmidhuber 1997),
GRU (Cho et al. 2014) — carry a hidden state across time and are the default for sequence modeling.
Temporal convolutional networks (Bai et al. 2018) replace recurrence with stacked dilated causal
convolutions, getting a large receptive field with parallel training. Gated linear units (Dauphin
et al. 2017) are a convolutional building block in which the output is an elementwise product of a
linear projection and a sigmoid gate of another, `GLU(x) = (W₁x) ⊙ σ(W₂x)` — the gate learns to let
through only the components relevant to the sequential pattern, and the multiplicative form gives
cleaner gradient flow than a stacked nonlinearity. Separately, **frequency-domain** views of a
series have a long history: the discrete Fourier transform `y_u = (1/N) Σ_{t} x_t e^{-i2πut/N}`
decomposes a series onto a trigonometric basis, exposing periodic and autocorrelation structure
that is awkward to read off in the time domain. State Frequency Memory (Zhang et al. 2017) folds DFT
into LSTM cell states for stock prediction; Spectral Residual (Ren et al. 2019) uses DFT for anomaly
detection. So DFT is a known, useful lens for *temporal* patterns.

**Modeling correlations across series via graphs.** The natural object for inter-series structure is
a graph: nodes are series, edge weights `w_{ij}` are correlation strengths, collected in an adjacency
matrix `W ∈ R^{N×N}`. Graph signal processing gives a Fourier theory on such a graph. For a symmetric
nonnegative adjacency, the normalized graph Laplacian is `L = I_N − D^{-1/2} W D^{-1/2}`, where `D`
is the diagonal degree matrix `D_{ii}=Σ_j w_{ij}`. `L` is real symmetric positive-semidefinite, so it
has an orthonormal eigendecomposition `L = U Λ U^T` with eigenvalues in `[0,2]`. The eigenvectors `U`
are the "graph Fourier modes" — smooth-to-oscillatory patterns over the nodes — and the **graph
Fourier transform** (GFT) of a graph signal `x ∈ R^N` is `GF(x) = U^T x`, with inverse
`GF^{-1}(x̂) = U x̂`. A spectral graph convolution filters in this basis: `y = U g_θ(Λ) U^T x`, where
`g_θ(Λ)` is a learnable function of the eigenvalues. This is the foundation of the spectral GCN line
(Bruna et al. 2013).

A useful fact from graph signal processing is that projecting correlated node signals onto the
Laplacian eigenbasis rotates the mixed node coordinates into **orthogonal graph modes**. Low-frequency
modes capture slowly varying components over strongly connected nodes; higher-frequency modes capture
sharper contrasts. When the raw node traces mix several such components, this rotation can expose
network-level or community-level temporal components that are less entangled than the originals.

## Baselines

**Spectral GCN with Chebyshev approximation — ChebNet (Defferrard et al. 2016).** Computing
`U g_θ(Λ) U^T x` directly needs the eigendecomposition of `L`, which is `O(N^3)`. ChebNet
sidesteps the eigendecomposition by writing the filter as a truncated Chebyshev polynomial of the
(rescaled) Laplacian, `g_θ(L) = Σ_{k=0}^{K} θ_k T_k(L̃)`, with `L̃ = 2L/λ_max − I` so the argument
lands in `[-1,1]`, the polynomials computed by the recurrence `T_0 = I`, `T_1 = L̃`,
`T_k = 2 L̃ T_{k-1} − T_{k-2}`. Because `T_k(L̃)` is a degree-`k` polynomial in `L`, applying it only
mixes each node with its `k`-hop neighborhood — the filter is *localized* and costs `O(K·|E|)`
sparse matrix-vector products, no eigendecomposition. GCN (Kipf & Welling 2016) is a first-order
special case.

**Stacked graph-conv + recurrent spatio-temporal models — DCRNN (Li et al. 2017), STGCN
(Yu et al. 2017), GraphWaveNet (Wu et al. 2019).** These wire the two lines together by alternating
a graph-convolution module (for the spatial/cross-series mixing) with a temporal module (a GRU in
DCRNN, gated temporal convolutions in STGCN, dilated causal convolutions in GraphWaveNet). They are
the state of the art for traffic forecasting. DCRNN and STGCN use a pre-defined adjacency (the road
network) as a prior; GraphWaveNet adds a learnable adjacency.

**Pure-temporal deep forecasters — N-BEATS (Oreshkin et al. 2019), TCN (Bai et al. 2018), DeepGLO
(Sen et al. 2019), DeepState (Rangapuram et al. 2018).** N-BEATS is a deep stack of fully-connected
blocks with two design ideas worth carrying forward. First, **doubly residual stacking**: each block
emits two outputs, a *forecast* of the future and a *backcast* (a reconstruction of its own input
window); the backcast is subtracted from the input before it passes to the next block, so each block
only has to explain the residual its predecessors could not, and the block forecasts are summed.
Second, **basis expansion**: a block produces its output as `Y = V θ`, where `V` is a set of learnable
basis vectors and `θ` are expansion coefficients from a fully-connected layer. DeepGLO captures
global structure through matrix factorization.

## Evaluation settings

The natural yardsticks for networked-sensor forecasting at this time:

- **METR-LA** — 207 loop-detector sensors on Los Angeles highways, traffic *speed*, 5-minute
  intervals (Li et al. 2017, introduced with DCRNN).
- **PEMS-BAY** — 325 sensors in the San Francisco Bay Area, traffic *speed*, 5-minute intervals
  (Li et al. 2017).
- **PEMS04** — 307 sensors, traffic *flow*, California Caltrans District 4, 5-minute intervals
  (used with ASTGCN, Guo et al. 2019).
- Beyond traffic, multivariate forecasting benchmarks of the era span solar/electricity/ECG and
  daily case counts across regions.
- Protocol: a sliding window of `K=12` timestamps (one hour at 5-min sampling) predicts the next
  `H=12` (the following hour); per-dataset Z-score normalization with metrics computed after the
  inverse transform; missing values (encoded as `0.0`) masked out of the loss and the metrics.
- Metrics: MAE, RMSE, MAPE, all lower-is-better, in original scale after inverse transform.

## Code framework

A forecasting model plugs into a fixed training/evaluation harness: an Adam optimizer, a masked
regression loss, Z-score (de)normalization, and a training loop that already exist. What the harness
calls is a single module whose `forward` maps a window to a horizon. Nothing about how that module
should mix the `N` series, model their temporal structure, or relate them is settled — that is
exactly what is to be designed. The starting substrate is therefore a bare module that consumes
`[B, K, N]` and must return `[B, H, N]`, with one empty slot where the architecture will go, plus
the generic primitives (gated linear unit, a recurrent encoder, basic linear layers, the FFT) that
already exist and may or may not be used.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class Config:
    input_len: int = 12       # K: history window
    output_len: int = 12      # H: forecast horizon
    num_features: int = 207   # N: number of series / nodes
    hidden_size: int = 5
    num_blocks: int = 2
    dropout: float = 0.5


class GLU(nn.Module):
    """Gated linear unit (Dauphin et al. 2017): a generic gated projection that
    already exists. out = (W1 x) * sigmoid(W2 x)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.left = nn.Linear(in_dim, out_dim)
        self.right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.left(x) * torch.sigmoid(self.right(x))


class Model(nn.Module):
    """Maps a window of N series to a horizon for all N series.

    forward(inputs): inputs is [B, K, N]; returns [B, H, N].

    The data pipeline, Adam optimizer, masked loss, and Z-score (de)normalization
    already exist around this module. Available primitives: GLU above, nn.GRU,
    nn.Linear, torch.fft. How to model intra-series temporal structure, how to
    relate the N series to each other, and how to fuse the two is undecided."""

    def __init__(self, config: Config):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        # TODO: the architecture we will design goes here.

    def forward(self, inputs):
        # inputs: [B, K, N]
        # TODO: produce the [B, H, N] forecast.
        raise NotImplementedError
```
