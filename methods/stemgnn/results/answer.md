# StemGNN, distilled

StemGNN (Spectral Temporal Graph Neural Network) forecasts a set of correlated time-series by
modeling intra-series temporal patterns and inter-series correlations *jointly in the spectral
domain*. It learns the dependency graph from data via self-attention, then in each block applies a
Graph Fourier Transform (across series) and a Discrete Fourier Transform (across time) so both axes
are processed in one spectral representation, before transforming back.

## Problem it solves

Multivariate time-series forecasting on networked sensors: from a window of `K` past timestamps of
`N` coupled series `X ∈ R^{N×K}`, predict the next `H` timestamps for all series — capturing both
each series' temporal dynamics and how the series influence one another, without being given the
correlation graph as a prior.

## Key ideas

1. **Latent correlation layer (learn the graph).** Turn each node's history window into a node
   representation, then score all ordered pairs by self-attention. The high-level formula is
   query-key attention `W = softmax(QK^T/√d)`, `Q = R W^Q`, `K = R W^K`; the MLS-Bench edit follows
   the BasicTS code shape and uses additive scalar key/query scores `softmax(LeakyReLU(k_i + q_j))`.
   `W ∈ R^{N×N}` is the adjacency — data-driven, no prior topology, interpretable. Symmetrize
   `W ← ½(W + W^T)` and form the normalized Laplacian `L = D^{-1/2}(D − W)D^{-1/2}` with
   `D_{ii} = Σ_j W_{ij}`. The edit computes this degree from the symmetrized matrix. BasicTS computes
   the row-sum degree immediately before symmetrization and then uses that degree with the symmetrized
   adjacency, so the two are not algebraically identical when the pre-symmetry attention is asymmetric.

2. **Joint spectral modeling.** The Graph Fourier Transform `GF(X) = U^T X` (eigenbasis of `L`)
   rotates the coupled node series into orthogonal, often smoother graph modes. The temporal
   structure of each mode is then modeled with the DFT rather than an RNN, so both transforms act on
   the same representation.

3. **Spectral Sequential (Spe-Seq) Cell.** DFT each (graph-spectral) series; process the real and
   imaginary parts with separate gated linear units `GLU(z) = (W₁z) ⊙ σ(W₂z)` (the gate keeps the
   frequency components carrying the sequential pattern); recombine and inverse-DFT. Frequency-domain
   modeling genuinely forecasts: predicting the next time value is equivalent to predicting a single
   new DFT coefficient (derivation below), with no periodicity assumption.

4. **StemGNN block.** Embed the Spe-Seq Cell `S` inside a spectral graph convolution:
   `Z_j = GF^{-1}( Σ_i g_{Θ_ij}(Λ_i) · S(GF(X_i)) )`, where `g_Θ(Λ)` is a learnable eigenvalue filter.

5. **Doubly residual stacking + basis expansion (from N-BEATS).** Each block emits a forecast and a
   backcast (input reconstruction) via basis expansion `Y = Vθ`. The backcast is subtracted from the
   input to the next block (so block 2 explains block 1's residual); forecasts are summed
   `Ŷ = Ŷ_1 + Ŷ_2`. The implementation uses a final `Linear → LeakyReLU → Linear` head.

6. **Chebyshev realization (eigendecomposition-free).** Honest GFT needs `O(N³)` eigendecomposition
   of a *learned, changing* Laplacian each step — slow and numerically fragile. Restricting the
   filter to a polynomial of `L` cancels the eigenvectors:
   `U(Σ_k θ_k Λ^k)U^T = Σ_k θ_k UΛ^kU^T = Σ_k θ_k (UΛU^T)^k = Σ_k θ_k L^k`. The ChebNet recurrence is
   `T_0=I, T_1=L̃, T_k = 2L̃T_{k-1}−T_{k-2}`, giving localized `K`-hop filters via cheap matmuls.
   The implementation keeps four polynomial channels but realizes them as `[0, L, 2L^2, 4L^3−L]`
   using the learned normalized Laplacian directly, not the exact rescaled Chebyshev stack.

## Spe-Seq Cell forecasting equivalence

For `x_0,…,x_{N-1}` with DFT `y_u = (1/N)Σ_t x_t e^{-i2πut/N}`, appending an unknown `x_N` gives the
length-`(N+1)` DFT `ŷ_u`. For `u<N`,
`ŷ_u − y_u = Σ_t x_t(e^{-i2πut/(N+1)}/(N+1) − e^{-i2πut/N}/N) + (x_N/(N+1))e^{-i2πuN/(N+1)}`,
and `ŷ_N = (1/(N+1))Σ_t x_t e^{-i2πNt/(N+1)} + (x_N/(N+1))e^{-i2πN^2/(N+1)}`. Each coefficient is
history plus one term in `x_N`. So learning `ŷ_N = M(y_0,…,y_{N-1})` lets one recover
`x_N = (N+1)ŷ_N e^{i2πN^2/(N+1)} − Σ_t x_t e^{i2πN(N-t)/(N+1)}` and then, for `u<N`,
`ŷ_u = y_u + Σ_t x_t(e^{-i2πut/(N+1)}/(N+1) − e^{-i2πut/N}/N) + (x_N/(N+1))e^{-i2πuN/(N+1)}`.
The length-`N+1` IDFT `x̂_t = Σ_u ŷ_u e^{i2πut/(N+1)}` then returns the known history and the
forecast `x̂_N`. Time-domain forecasting is equivalent, given the history, to predicting one new
frequency-domain coefficient.

## Loss and complexity

Full method objective: forecasting loss plus backcasting reconstruction:
`L = Σ_t ‖X̂_t − X_t‖² + Σ_{t}Σ_i ‖B_{t-i}(X) − X_{t-i}‖²`. Costs: latent correlation `O(N²d)`,
Spe-Seq Cell `O(NT log T)` (FFT), GFT `O(N³)` if eigendecomposed (avoided by Chebyshev). Note: the
BasicTS implementation drops the backcast loss term, uses GLU-only (no 1D conv) in the Spe-Seq Cell,
uses the backcast only as the first block's residual passed to the second block, and uses the
four-channel polynomial approximation above in place of eigendecomposition; the MLS-Bench edit keeps
that structure while adapting the class names and forward signature to the benchmark harness.

## Working code

Faithful to the MLS-Bench edit of the BasicTS implementation: a learned-graph latent correlation
layer with polynomial expansion, two blocks (Spe-Seq Cell = FFT + gated processing of real/imag +
iFFT, embedded in a spectral graph convolution), doubly-residual forecast aggregation, and a final
FC head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=5)
    num_blocks: int = field(default=2)
    dropout: float = field(default=0.5)


class GLU(nn.Module):
    """Gated linear unit: (W1 x) * sigmoid(W2 x)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.left = nn.Linear(in_dim, out_dim)
        self.right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.left(x) * torch.sigmoid(self.right(x))


class StockBlock(nn.Module):
    """One StemGNN block: Spe-Seq Cell embedded in a spectral graph convolution,
    with forecast and (for the first block) backcast heads."""
    def __init__(self, input_len, num_features, hidden_size, layer_idx):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.output_hidden_size = 4 * hidden_size

        # graph-conv kernel; 4 = polynomial channels used by the implementation
        self.weight = nn.Parameter(
            torch.Tensor(1, 4, 1, input_len * hidden_size, hidden_size * input_len))
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(input_len * hidden_size, input_len * hidden_size)
        self.forecast_result = nn.Linear(input_len * hidden_size, input_len)
        if layer_idx == 0:
            self.backcast = nn.Linear(input_len * hidden_size, input_len)
        self.backcast_short_cut = nn.Linear(input_len, input_len)

        # gated frequency-domain processing; real and imaginary parts get separate GLUs
        self.GLUs = nn.ModuleList()
        for i in range(3):
            in_d = input_len * 4 if i == 0 else input_len * self.output_hidden_size
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))

    def spe_seq_cell(self, inputs):
        # Spectral Sequential Cell: DFT -> gate real/imag -> IDFT
        B, _, _, N, L = inputs.size()
        inputs = inputs.view(B, -1, N, L)
        ffted = torch.fft.fft(inputs, dim=-1)
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        imag = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            imag = self.GLUs[i * 2 + 1](imag)
        real = real.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
        imag = imag.reshape(B, N, 4, -1).permute(0, 2, 1, 3).contiguous()
        return torch.fft.ifft(torch.complex(real, imag), dim=-1).real

    def forward(self, x, graph):
        graph = graph.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(graph, x)                       # polynomial graph filtering
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)  # temporal-spectral learner
        igfted = torch.matmul(gconv_input, self.weight).sum(dim=1)  # learned channel mix
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.layer_idx == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class Custom(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        N = config.num_features
        L = config.input_len
        self.num_blocks = config.num_blocks

        # latent correlation layer (learned graph via additive self-attention)
        self.weight_key = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(L, N)

        self.stock_block = nn.ModuleList(
            [StockBlock(L, N, config.hidden_size, i) for i in range(config.num_blocks)])

        self.fc = nn.Sequential(
            nn.Linear(L, L), nn.LeakyReLU(), nn.Linear(L, config.output_len))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(config.dropout)

    def _latent_graph(self, x):
        h, _ = self.GRU(x.permute(2, 0, 1))      # node-indexed history vectors -> node reps
        h = h.permute(1, 0, 2).contiguous()
        h = h.permute(0, 2, 1).contiguous()
        key = torch.matmul(h, self.weight_key)
        query = torch.matmul(h, self.weight_query)
        N = h.size(1)
        attn = key.repeat(1, 1, N).view(-1, N * N, 1) + query.repeat(1, N, 1)
        attn = self.leakyrelu(attn.squeeze(2).view(-1, N, N))
        attn = self.dropout(F.softmax(attn, dim=2))
        attn = torch.mean(attn, dim=0)
        attn = 0.5 * (attn + attn.T)             # symmetrize before Laplacian normalization
        degree = torch.sum(attn, dim=1)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-7))
        L = D_inv_sqrt @ (torch.diag(degree) - attn) @ D_inv_sqrt   # normalized Laplacian
        # four-channel polynomial stack: [0, L, 2L^2, 4L^3-L]
        L = L.unsqueeze(0)
        T0 = torch.zeros_like(L)
        T1 = L
        T2 = 2 * torch.matmul(L, T1) - T0
        T3 = 2 * torch.matmul(L, T2) - T1
        return torch.cat([T0, T1, T2, T3], dim=0)

    def forward(self, inputs, inputs_timestamps):
        graph = self._latent_graph(inputs)
        x = inputs.unsqueeze(1).transpose(-1, -2)   # [B, 1, N, L]
        results = []
        for i in range(self.num_blocks):
            pred, x = self.stock_block[i](x, graph)  # x becomes the backcast residual
            results.append(pred)
        prediction = sum(results)                    # doubly-residual aggregation
        return self.fc(prediction).transpose(1, 2)   # [B, H, N]
```

## Relation to prior methods

- **ChebNet / GCN (Defferrard 2016; Kipf & Welling 2016):** supply the spectral graph convolution and
  the Chebyshev-polynomial trick that removes the eigendecomposition; StemGNN adds a temporal axis and
  a learned graph.
- **DCRNN / STGCN / GraphWaveNet (Li 2017; Yu 2017; Wu 2019):** stack graph conv with *time-domain*
  temporal modules and (mostly) need a prior graph; StemGNN fuses both axes in the spectral domain and
  learns the graph.
- **N-BEATS (Oreshkin 2019):** source of doubly residual stacking and basis-expansion forecast/backcast
  heads; StemGNN adapts that residual idea to a multivariate spectral graph setting.
- **GLU (Dauphin 2017):** the gated operator used inside the Spe-Seq Cell.
- **SFM / Spectral Residual (Zhang 2017; Ren 2019):** prior DFT-in-time-series; StemGNN combines DFT
  with GFT in one block.
