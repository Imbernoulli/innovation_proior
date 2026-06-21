Multi-scale mixing improved on the floor everywhere, but barely, and the *pattern* of the barely is the diagnosis. TimeMixer moved METR-LA from 4.0599 to a tight 3.9497 mean across three seeds, PEMS-BAY from 2.1334 to 2.0641, PEMS04 from 28.4748 to 27.7508 — gains of a few percent, and crucially *uniform* across the three datasets rather than concentrated on PEMS04 where I had predicted the scale axis should pay off most. That uniformity is the tell. I expected flow data to reward multi-scale decomposition disproportionately and it did not; the coarsest scale was a length-three reed, and reading a macro trend off three averaged points in a twelve-step window simply does not recover much structure. So the residual all three rungs leave is *not* on the temporal-scale axis. Both DLinear and TimeMixer are channel-independent by construction — every node its own univariate sequence, with no path for one sensor's recent state to inform another's future. And that is exactly the structure traffic has and these models refuse to model: congestion on one segment shows up downstream minutes later, so node $j$'s recent past genuinely predicts node $i$'s near future. The next rung has to add cross-node coupling, the only axis left with un-reached signal on it.

I propose StemGNN: a spectral temporal graph network that learns the graph from the window and models cross-node and within-node structure in *one* representation. The harness hands me only the value tensor and timestamps — no adjacency — and even if I had the road graph the physical topology may not be what drives the forecast, since two distant roads can behave identically at rush hour while two adjacent ones decouple at a junction. So the graph must be *learned*. I compress each node's twelve-step history into a vector with a GRU run over the node-indexed sequences, then score every ordered pair $(i,j)$ with learned additive attention: each node gets a scalar key and a scalar query, the pair score is $\text{LeakyReLU}(\text{key}_i + \text{query}_j)$, a row-wise softmax normalizes the scores into affinities, and averaging over the batch gives a single shared $N\times N$ structure for the window. This is data-driven, needs no prior topology, and is interpretable.

The central idea is to model both axes spectrally so the cross-node interaction is *absorbed into a basis* rather than stapled on. For a symmetric nonnegative $W$, the normalized Laplacian $L = I - D^{-1/2}WD^{-1/2}$ is real, symmetric, and positive-semidefinite, so it diagonalizes, and projecting the $N$ coupled node signals onto its eigenvectors — the graph Fourier transform — rotates them into orthogonal graph modes that are individually *less entangled in time* and therefore easier to forecast. That is the fusion the stacked graph-conv-plus-recurrent baselines could not do. Once in the graph-spectral domain I still have $N$ univariate mode-series to model in time, and here I use the *temporal* Fourier transform rather than an RNN, because traffic periodicity is real and a basis that names frequency directly is the right place to model it — now both transforms act on the same representation. The temporal learner is a gate that suppresses noise frequencies and keeps the ones carrying sequential pattern — a gated linear unit, $\text{GLU}(z) = (W_1 z)\odot\sigma(W_2 z)$, applied to the real and imaginary parts of the DFT *separately*, because cosine and sine content encode different information and should not share weights. DFT $\to$ gated processing on real and imaginary parts $\to$ inverse DFT is the spectral sequential cell, embedded *inside* the spectral graph convolution so the two spectral computations are fused.

There is one wall, and the implementation is built to climb it. The honest graph Fourier transform needs the eigenvectors of $L$ — an $O(N^3)$ eigendecomposition every forward pass, on a matrix that *changes* every batch because $W$ is learned, with fragile backpropagation through `eig` on a near-singular learned Laplacian. The escape is the ChebNet move: restrict the graph filter to a *polynomial* of the Laplacian, and the eigenvectors cancel — a polynomial in the eigenvalues conjugated by the eigenvectors is the same polynomial in $L$, so only repeated matrix multiplication by $L$ remains, which also localizes the filter to $K$-hop neighborhoods. The `_latent_graph` builds exactly this: it symmetrizes the learned attention with $0.5(W + W^\top)$ (since $L$ needs a symmetric $W$), forms the normalized Laplacian with a $10^{-7}$ floor under the square root so an isolated low-degree node does not divide by zero, then expands four polynomial channels through $T_0,\ T_1=L,\ T_2=2L T_1 - T_0,\ T_3=2L T_2 - T_1$. Against the canonical method this is the *lean* realization: $T_0$ is initialized to zeros rather than $I$ on a spectrally-rescaled $\tilde{L}$, so the four channels are effectively $[0,\ L,\ 2L^2,\ 4L^3 - L]$ built from the *unrescaled* learned Laplacian. It keeps the four-channel polynomial-filter shape and the eigendecomposition-free property — the load-bearing ideas — while dropping the exact Chebyshev normalization, and I derive against that because it is what the edit surface runs.

The block ties it together. Per StockBlock: graph-filter the input by the four polynomial channels, run the spectral sequential cell to learn temporal patterns in the graph-spectral domain, apply a learnable channel-mixing weight that plays the role of the eigenvalue filter, and produce a forecast through a $\sigma$-gated linear head. The first block additionally emits a *backcast* — a reconstruction of its own input window — and the second block consumes the residual $x - \text{backcast}$, so it only has to explain what the first could not (the doubly-residual stacking idea). Here too I name what the harness drops: the canonical version adds an explicit backcast-reconstruction *loss term* and emits the forecast through a learnable-basis expansion, whereas this edit keeps the backcast purely as the residual passed forward — no separate backcast supervision, plain linear-with-sigmoid heads — so the regularization-by-reconstruction story is weaker. Two blocks, forecasts summed, then a final $\text{Linear}\to\text{LeakyReLU}\to\text{Linear}$ head to the twelve-step horizon. No timestamps are used, periodicity read straight off the values through the DFT. The GRU-plus-spectral stack is unstable at the harness default, so `CONFIG_OVERRIDES = {'lr': 0.001}`, `weight_decay` at default. I expect the clearest win on PEMS04, where flow propagation between sensors is strongest and is precisely the cross-node signal the channel-independent rungs threw away, with more modest gains on the smoother, more self-predictable speed series.

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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.left = nn.Linear(in_dim, out_dim)
        self.right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.left(x) * torch.sigmoid(self.right(x))


class StockBlock(nn.Module):
    def __init__(self, input_len, num_features, hidden_size, layer_idx):
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.output_hidden_size = 4 * hidden_size

        self.weight = nn.Parameter(
            torch.Tensor(1, 4, 1, input_len * hidden_size, hidden_size * input_len))
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(input_len * hidden_size, input_len * hidden_size)
        self.forecast_result = nn.Linear(input_len * hidden_size, input_len)
        if layer_idx == 0:
            self.backcast = nn.Linear(input_len * hidden_size, input_len)
        self.backcast_short_cut = nn.Linear(input_len, input_len)

        self.GLUs = nn.ModuleList()
        for i in range(3):
            in_d = input_len * 4 if i == 0 else input_len * self.output_hidden_size
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))
            self.GLUs.append(GLU(in_d, input_len * self.output_hidden_size))

    def spe_seq_cell(self, inputs):
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
        gfted = torch.matmul(graph, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight).sum(dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.layer_idx == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class Custom(nn.Module):
    """StemGNN: Spectral Temporal Graph Neural Network baseline.

    Learns a latent graph via self-attention, applies Chebyshev graph
    convolution, and processes temporal patterns in the spectral domain.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        N = config.num_features
        L = config.input_len
        self.num_blocks = config.num_blocks

        # Latent graph via self-attention
        self.weight_key = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(N, 1))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(L, N)

        # Backbone
        self.stock_block = nn.ModuleList([
            StockBlock(L, N, config.hidden_size, i)
            for i in range(config.num_blocks)
        ])

        # Output
        self.fc = nn.Sequential(
            nn.Linear(L, L), nn.LeakyReLU(),
            nn.Linear(L, config.output_len))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(config.dropout)

    def _latent_graph(self, x):
        # x: [B, T, N]
        h, _ = self.GRU(x.permute(2, 0, 1))  # [N, B, N]
        h = h.permute(1, 0, 2).contiguous()  # [B, N, N]
        h = h.permute(0, 2, 1).contiguous()  # [B, N, N] transposed for attention
        key = torch.matmul(h, self.weight_key)    # [B, N, 1]
        query = torch.matmul(h, self.weight_query) # [B, N, 1]
        N = h.size(1)
        attn = key.repeat(1, 1, N).view(-1, N * N, 1) + query.repeat(1, N, 1)
        attn = self.leakyrelu(attn.squeeze(2).view(-1, N, N))
        attn = self.dropout(F.softmax(attn, dim=2))
        attn = torch.mean(attn, dim=0)
        attn = 0.5 * (attn + attn.T)
        degree = torch.sum(attn, dim=1)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-7))
        L = D_inv_sqrt @ (torch.diag(degree) - attn) @ D_inv_sqrt
        # Chebyshev polynomials (order 3)
        L = L.unsqueeze(0)
        T0 = torch.zeros_like(L)
        T1 = L
        T2 = 2 * torch.matmul(L, T1) - T0
        T3 = 2 * torch.matmul(L, T2) - T1
        return torch.cat([T0, T1, T2, T3], dim=0)

    def forward(self, inputs, inputs_timestamps):
        graph = self._latent_graph(inputs)
        x = inputs.unsqueeze(1).transpose(-1, -2)  # [B, 1, N, T]
        results = []
        for i in range(self.num_blocks):
            pred, x = self.stock_block[i](x, graph)
            results.append(pred)
        prediction = sum(results)
        prediction = self.fc(prediction).transpose(1, 2)  # [B, T', N]
        return prediction


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.001}
```
