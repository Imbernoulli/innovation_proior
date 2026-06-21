The problem is forecasting a network of correlated sensors: $N$ series — traffic-speed loops on adjacent highway segments, meters across a grid, daily case counts across regions — stacked into a matrix $X \in \mathbb{R}^{N\times T}$, where from the last $K$ columns we must predict the next $H$ for every series at once. What makes this hard, and what every clean univariate forecaster ignores, is that the series are coupled: congestion on one segment shows up a few minutes later on the next, so series $i$'s future genuinely depends on series $j$'s recent past. There are two kinds of structure to capture — the temporal pattern *within* each series (trend, periodicity, autocorrelation) and the correlation *across* series — and the accuracy lives in the coupling between them, not in either one alone. The defaults each handle only one half. Recurrent nets and temporal convolutions model a single series' dynamics well, and there is a frequency-domain lens worth keeping: the discrete Fourier transform $y_u = \tfrac{1}{N}\sum_t x_t e^{-i2\pi ut/N}$ collapses periodic and autocorrelation structure that is smeared across the time axis onto a handful of trigonometric coefficients — and traffic periodicity (the daily rush-hour cycle) is real, so a basis that names frequency directly is the right basis to model on. For the cross-series part the natural object is a graph: nodes are series, edge weights $w_{ij}$ are correlation strengths in an adjacency $W$, and on a symmetric nonnegative $W$ the normalized Laplacian $L = I_N - D^{-1/2}WD^{-1/2}$ diagonalizes as $L = U\Lambda U^T$, giving a graph Fourier transform $\mathrm{GF}(x) = U^T x$ and a spectral graph convolution $y = U g_\theta(\Lambda) U^T x$.

The state of the art — the stacked graph-conv-plus-recurrent models, DCRNN and STGCN and GraphWaveNet — does the obvious thing: it interleaves a graph-convolution module to mix across nodes with a temporal module (a GRU, gated temporal convolutions, dilated causal convolutions) to march in time. But the graph module is spectral while the temporal module runs entirely in the *time* domain, so the two computations never share a representation. There are two Fourier views available, and these architectures use one of them spatially and throw the other away temporally — leaving the whole point of having a transform on the table. There is a second sore spot too: DCRNN and STGCN demand the adjacency $W$ as a prior (the road network), which is not always available, and even when it is, the physical network may not be the structure that actually drives the forecast — two roads can be far apart yet behave identically at rush hour. So we have two complaints to fix at once: model both axes in a *single* spectral representation, and learn the graph from data instead of being told it.

I propose StemGNN, the Spectral Temporal Graph Neural Network. It learns the dependency graph from data, then in each block applies a Graph Fourier Transform across the series and a Discrete Fourier Transform across time so both axes are processed in one spectral representation before being transformed back. Take the graph-learning problem first, since everything downstream consumes a $W$. Each node's history window is compressed into a representation $r_i$ (a GRU over the node-indexed history vectors), and the affinity between nodes is a learned function of those representations — exactly what self-attention computes. The clean version is query-key attention, $Q = R W^Q$, $K = R W^K$, $W = \mathrm{softmax}(QK^T/\sqrt{d})$; the code-friendly variant makes the same idea additive, giving each node a scalar key $k_i$ and scalar query $q_j$, scoring each ordered pair as $\mathrm{LeakyReLU}(k_i + q_j)$ and row-normalizing with a softmax. That $N\times N$ matrix *is* the adjacency: data-driven, needing no prior topology, and interpretable — one can read off which sensors the model thinks drive which. I symmetrize $W \leftarrow \tfrac{1}{2}(W + W^T)$ before taking degrees so the normalized Laplacian $L = D^{-1/2}(D-W)D^{-1/2}$, with $D_{ii} = \sum_j W_{ij}$, is built from a genuinely symmetric $W$, and I add a small $10^{-7}$ under the square root of the degree so an isolated low-degree node does not divide by zero. The attention is averaged over the batch so the window has one shared graph rather than $B$ different ones.

Now the central move: model both axes jointly *in the spectral domain* rather than stapling a spectral spatial module to a time-domain temporal one. Apply the GFT first, to the whole multivariate input — $\mathrm{GF}(X) = U^T X$ projects the $N$ coupled node series onto the Laplacian eigenbasis. Projecting correlated node signals onto the eigenvectors rotates the mixed node coordinates into orthogonal graph modes: low-frequency eigenvectors pick out slowly varying components over strongly connected nodes, higher-frequency eigenvectors pick out sharper contrasts, and a raw node trace that mixes several such components is decomposed into modes that are individually smoother and less entangled in time. That is the key insight — the GFT is not a spatial mixer bolted on; it is the step that turns "$N$ coupled node series" into graph-mode series with cleaner temporal structure, an easier forecasting problem. With $N$ univariate series now sitting in the graph-spectral domain, I model each of them *with the temporal Fourier transform* instead of handing them to a time-domain RNN — DFT each mode, learn on the trigonometric coefficients, transform back. Both transforms now act on the same representation, and the temporal modeling happens on the graph-spectral series, which is the joint spectral modeling the stacked baselines could not do.

It matters that the DFT step genuinely forecasts and is not merely a feature extractor. Take a real series $x_0,\dots,x_{N-1}$ with DFT $y_u = \tfrac{1}{N}\sum_t x_t e^{-i2\pi ut/N}$, and imagine the unknown next value $x_N$ appended. The DFT of the length-$(N+1)$ sequence, $\hat y_u$, satisfies, for $u<N$,
$$\hat y_u - y_u = \sum_{t=0}^{N-1} x_t\!\left(\frac{e^{-i2\pi ut/(N+1)}}{N+1} - \frac{e^{-i2\pi ut/N}}{N}\right) + \frac{x_N}{N+1}\,e^{-i2\pi uN/(N+1)},$$
and for the new top coefficient $\hat y_N = \tfrac{1}{N+1}\sum_{t=0}^{N-1} x_t e^{-i2\pi Nt/(N+1)} + \tfrac{x_N}{N+1}e^{-i2\pi N^2/(N+1)}$. Every coefficient is a quantity computable from the history plus a single term carrying the unknown $x_N$. So if a learnable map produces $\hat y_N = M(y_0,\dots,y_{N-1})$, the last relation inverts to recover $x_N = (N+1)\hat y_N e^{i2\pi N^2/(N+1)} - \sum_t x_t e^{i2\pi N(N-t)/(N+1)}$, the remaining $\hat y_u$ back-substitute, and the ordinary length-$(N+1)$ inverse DFT reproduces the history exactly and hands back $\hat x_N$ as the forecast. Predicting the next time-domain value is therefore equivalent, given the history, to predicting one new frequency-domain coefficient — and unlike a period-detector this assumes nothing about periodicity; it just learns the frequency-domain map.

So what is $M$? The DFT output is complex, a real part and an imaginary part, and these encode different information (cosine versus sine content), so they are processed by the same kind of operator but with *separate* parameters rather than forced to share weights. The operator itself should be able to suppress the frequency components that are noise and keep the ones carrying the sequential pattern — that is a gate, exactly the gated linear unit $\mathrm{GLU}(z) = (W_1 z)\odot\sigma(W_2 z)$, whose sigmoid branch learns per-component how much of the linear branch to pass. Symbolically each part runs $M^*(\hat X_u^*) = \theta_\tau^*(\hat X_u^*)\odot\sigma^*(\theta_\tau^*(\hat X_u^*))$ for $*\in\{r,i\}$; in the lean implementation the optional frequency-mixing convolution is dropped and stacked GLUs do the mixing directly. Recombining the real and imaginary results and applying the inverse DFT gives the four-step pipeline — DFT, gated processing on real and imaginary separately, IDFT — which is the Spectral Sequential (Spe-Seq) Cell, the temporal-spectral learner sitting on each graph-spectral series. Embedding that cell inside the spectral graph convolution gives one StemGNN block, per output channel $j$:
$$Z_j = \mathrm{GF}^{-1}\!\left(\sum_i g_{\Theta_{ij}}(\Lambda_i)\cdot S\big(\mathrm{GF}(X_i)\big)\right),$$
read inside-out: GFT each input channel into the graph-spectral domain, run the Spe-Seq Cell $S$ to learn temporal patterns there, apply the learnable eigenvalue filter $g_\Theta(\Lambda)$, sum over input channels, and inverse-GFT back to node space. The cell is embedded *inside* the convolution, so the temporal and spatial spectral computations are fused rather than stacked.

For the block output I borrow from N-BEATS: rather than emit raw numbers, each block emits coefficients $\theta$ from a fully-connected layer and combines learnable basis vectors $V$, so the output is $Y = V\theta$, the basis being shared learned waveforms and the block only saying how much of each to use. Each block carries *two* such heads — a forecasting head and a backcasting head that reconstructs the block's own input window. The backcast earns its place twice over: forcing the block to reconstruct what it consumed regularizes its representation, and subtracting the backcast from the input before passing to the next block lets the second block see $X - \hat X_1$, the part the first could not explain, so it only learns the residual; the forecasts then sum, $\hat Y = \hat Y_1 + \hat Y_2$. I use two blocks in this bilevel residual stack and feed the summed forecasts through a final $\text{Linear}\to\text{LeakyReLU}\to\text{Linear}$ head. The full objective is a forecasting error plus a backcast reconstruction error, $\mathcal{L} = \sum_t \|\hat X_t - X_t\|^2 + \sum_t\sum_i \|B_{t-i}(X) - X_{t-i}\|^2$, though the lean implementation uses the backcast only as the residual passed forward and carries no separate backcast-loss term.

One step would have sunk the whole design. The honest GFT is $U^T X$, and $U$ is the eigenvector matrix of the Laplacian, so doing GFT literally means *eigendecomposing* $L$ — an $O(N^3)$ operation, every forward pass, on a matrix that *changes* every batch because $W$ is being learned. That is not merely slow; backpropagating through the eigendecomposition of a learned, possibly near-singular Laplacian is notoriously fragile. The escape is the one ChebNet found years ago. The convolution $U g_\theta(\Lambda) U^T x$ needs the eigenvectors only because $g_\theta$ is an arbitrary function of the eigenvalues; restrict $g_\theta$ to a *polynomial* of the Laplacian and the eigenvectors cancel,
$$U\Big(\sum_k \theta_k \Lambda^k\Big)U^T = \sum_k \theta_k\, U\Lambda^k U^T = \sum_k \theta_k\,(U\Lambda U^T)^k = \sum_k \theta_k\, L^k,$$
using $U^T U = I$. A polynomial in $\Lambda$ conjugated by $U$ is just the same polynomial in $L$ — no eigendecomposition, only repeated multiplication by $L$ — and a degree-$K$ polynomial mixes each node with only its $K$-hop neighborhood, so the filter is automatically localized, a feature rather than a cost. For conditioning the polynomial basis of choice is Chebyshev, with the three-term recurrence $T_0 = I$, $T_1 = \tilde L$, $T_k = 2\tilde L T_{k-1} - T_{k-2}$ on a rescaled $\tilde L$. The implementation keeps four polynomial channels but, in lean form, realizes them as $[\,0,\ L,\ 2L^2,\ 4L^3 - L\,]$ using the learned normalized Laplacian directly rather than the exact rescaled Chebyshev stack. The eigenvector matrix $U$ is never formed; the spectral convolution becomes cheap, stable matmuls against polynomial terms of $L$. The costs settle out as $O(N^2 d)$ for the latent correlation layer, $O(NT\log T)$ for the FFT-based Spe-Seq Cell, and the would-be $O(N^3)$ GFT avoided entirely.

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
