**Problem.** The first two rungs are channel-independent: no node can inform another. But traffic couples
across sensors — congestion propagates downstream minutes later — so node `j`'s recent past predicts node
`i`'s near future. The residual the floor and the multi-scale rung leave is cross-node coupling, and the
harness gives no adjacency, so the graph must be learned.

**Key idea.** Learn an `N × N` graph from the window and model both axes in one spectral representation.
A GRU compresses each node's history; additive self-attention (`LeakyReLU(key_i + query_j)`, row-softmax,
batch-averaged) gives a data-driven, symmetrized adjacency. Build the normalized Laplacian and convolve
across nodes in the **graph-spectral** domain (graph Fourier rotates coupled node series into less-entangled
modes), while modeling time on those modes in the **temporal-spectral** domain — a Spe-Seq Cell: DFT →
gated linear unit on real and imaginary parts separately → inverse DFT. Two StockBlocks stacked
doubly-residually (the first emits a backcast; the second models `x − backcast`), forecasts summed, a final
`Linear → LeakyReLU → Linear` head to the horizon.

**Avoiding the O(N³) wall.** The honest graph Fourier transform needs the eigenvectors of a *learned,
changing* Laplacian every step — fragile and slow. Restricting the filter to a polynomial of `L` cancels
the eigenvectors (a polynomial in `Λ` conjugated by `U` is the same polynomial in `L`), so only matmuls
against four Chebyshev-style channels remain, also localizing the filter to `K`-hop neighborhoods.

**Lean realization (what the harness drops).** This is the *lean* realization: `T0` is zeros (not `I`
on a rescaled `L̃`), so the four channels are `[0, L, 2L²−0, 4L³−L]` from the *unrescaled* learned Laplacian.
The backcast is kept only as the residual passed to the next block — there is **no separate
backcast-reconstruction loss** and **no learnable-basis-expansion head** (plain linear-with-sigmoid),
so the regularization-by-reconstruction is weaker here.

**Hyperparameters.** `hidden_size = 5`, `num_blocks = 2`, `dropout = 0.5`;
`CONFIG_OVERRIDES = {'lr': 0.001}` (the harness default 2e-3 is unstable for the GRU+spectral stack),
`weight_decay` at default.

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
