SOFTS landed exactly the split I predicted. Routing cross-node information through a robust global core reclaimed both speed datasets — METR-LA MAE fell to $3.8546$ (back below the channel-independent TimeMixer's $3.9497$ and well under StemGNN's fragile $4.0045$), and PEMS-BAY reached $1.9621$, under $2.00$ at last. But on PEMS04 it came in at $25.6572$, *worse* than StemGNN's $24.2040$, precisely as I worried: a hub that averages all nodes into one core blurs the directed upstream→downstream propagation that highway flow actually exhibits. So I have two facts. Cross-node coupling helps PEMS04, but neither a fragile spectral graph nor a robust-but-global core extracts it cleanly. And here is what reframes the whole climb: StemGNN's best-so-far PEMS04 of $24.20$ was won by a model whose *temporal* side — the spectral sequential cell — was its weak point. What if the PEMS04 residual is not primarily a cross-node problem but a *temporal* one — flow's sharp, multi-scale periodic structure that no rung so far has modeled in the right space?

Flow differs from speed in exactly this way. Speed is smooth and self-predictable; twelve steps ahead from twelve back per node captures most of it, which is why even DLinear did respectably. Flow spikes sharply at the commute peaks, the spikes have internal shape, and the same physical process shows fine structure *within* a cycle and a different structure *across* cycles. Every rung so far models the temporal axis as a flat function of absolute window position — linear maps over time, or SOFTS's single linear embedding — with no explicit handle on *which periodicities* a series carries or on the two distinct dependencies a periodic point has. So I attack the temporal representation directly. I propose TimesNet, which models temporal 2D-variation.

Be precise about the two dependencies, because vague talk of "temporal variation" goes nowhere. A point now depends on its immediate neighbors — the short-term movement *within* the current cycle, the intraperiod variation. It also depends on the same phase one cycle ago and ahead — the same five-minute slot in yesterday's rush, the interperiod variation. These are genuinely different, and real traffic has both, at several periods at once. The structural problem is the layout. My data is a 1D sequence indexed by time. Adjacency along that axis gives the intraperiod neighbors for free: $t-1$ and $t+1$ sit next to $t$. But the interperiod neighbor, the same phase one period back, lives at $t-p$ — and in the 1D layout that point is $p$ steps away with the entire previous cycle crammed between. A 1D convolution with any sane kernel never sees $t$ and $t-p$ together. So 1D presents one variation cleanly and effectively hides the other, and every prior rung inherited this bottleneck and had to recover the interperiod relationship implicitly, fighting the layout.

The fix is to change the layout so *both* become locality. Take the sequence and a period $p$. Chop it into consecutive blocks of length $p$ and stack those blocks as rows of a matrix. Now walking along a row is walking through one cycle (intraperiod), and walking down a column is walking through the same phase across successive cycles (interperiod). Both dependencies are adjacencies in a 2D array — the thing that was $p$ apart in 1D is now one step apart along the cross-period axis. And the instant the data is a 2D grid with two meaningful local axes, the entire mature toolbox of 2D convolution applies: a 2D kernel sees, in one receptive field, a few adjacent steps within the cycle *and* the same window of phases in neighboring cycles, modeling intra- and interperiod variation simultaneously — what no 1D kernel can do. The bottleneck was never that flow's variation is hard; it is that 1D was the wrong space.

Which $p$? Real series carry several periods and I do not know them a priori, so I discover them from the spectrum. The FFT amplitude at a frequency measures how strongly that periodic component is present. Compute the FFT of the window, take amplitudes, average over channels to get one shared amplitude profile (I want the *same* reshape applied to every node so they stay aligned in the grid), zero the DC term (its "period" is meaningless and its amplitude huge), consider frequencies up to Nyquist, and take the top-$k$ peaks as the dominant frequencies $f_i$ with periods $p_i = T // f_i$. Keep the $k$ amplitudes — they will serve as importance weights. With $\text{input\_len}=12$ the discoverable periods are short, but that is fine: flow's intra-spike shape lives at exactly those short periods, and $k=3$ gives three 2D views, each exposing a different periodicity.

What runs over each 2D tensor is a 2D conv — but the variation has structure at several scales (a fine wiggle over two or three steps, a broader hump over a third of a cycle, a slow tilt across cycles), and a single fixed kernel commits to one. That is what the Inception block is for: several kernel sizes in parallel inside one block, combined, so the block is multi-scale by construction and parameter-efficient. I use an Inception block of 2D convs with increasing kernel sizes $2i+1$ (padded to preserve spatial size), run in parallel and averaged, made into a small two-layer channel-bottleneck feedforward in 2D — Inception $\text{hidden}\to 4\cdot\text{hidden}$, GELU, Inception $4\cdot\text{hidden}\to\text{hidden}$. One decision matters for cleanliness: I share *one* Inception block across all $k$ periods. Each period gets a different reshape geometry, but the same conv weights process every grid, so model size is invariant to $k$ — $k$ becomes a pure width-of-search knob — and conceptually the Inception learns "how to read 2D temporal variation," which should not depend on which period produced the grid.

After the shared Inception transforms each 2D tensor, reshape each back to 1D and truncate to the working length, giving $k$ candidate representations. Fuse them by importance: a plain sum would ignore that some periods are far more present in this window than others, and I kept the amplitudes, which *are* the relative-importance signal. Push the $k$ amplitudes through a softmax over periods to get convex weights and take the weighted sum — confidence-weighted aggregation with the FFT amplitude as the confidence of each period. That is one TimesBlock — discover periods, reshape to $k$ 2D tensors, shared Inception on each, reshape back, amplitude-softmax aggregate — wrapped in a residual with a layer norm so each block learns only the correction to the variation representation and depth composes stably.

Making it a *forecaster* in this harness adds two limbs beyond the canonical reconstruction model. There is a horizon: I must produce twelve future steps, not just reconstruct the window. So the model first embeds the input with a 1D conv that maps the $N$ node features at each timestep to $\text{hidden}$ channels — and that conv is itself a *dense per-timestep cross-node mixing*, which is how TimesNet touches the spatial axis: not through any graph, but by letting every node feed every hidden channel at each step. Then a $\text{predict\_linear}$ stretches the sequence along time from $\text{input\_len}$ to $\text{input\_len}+\text{output\_len}$, so the TimesBlocks see and continue the cycles into the future; the residual TimesBlocks refine that extended representation; and a final linear projection maps $\text{hidden}$ back to the $N$ node outputs, of which I keep the last $\text{output\_len}$ positions. RevIN brackets the whole thing for non-stationarity. The model wants $\text{lr}=10^{-3}$ through `CONFIG_OVERRIDES`, with weight decay at default. This is a different bet from SOFTS: drop the explicit cross-node mechanism (keep only the dense embedding mixing) and instead model time in the 2D period space. I expect the clearest PEMS04 win on the ladder if the flow residual was really temporal; the open question is whether it holds the dense PEMS-BAY grid, where SOFTS's robust cross-node core may already capture what a 2D temporal model adds.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from dataclasses import dataclass, field
from basicts.configs import BasicTSModelConfig


@dataclass
class CustomConfig(BasicTSModelConfig):
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    num_features: int = field(default=207)
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    num_kernels: int = field(default=3)
    top_k: int = field(default=3)
    dropout: float = field(default=0.1)


class RevIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev
        else:
            return x * self.stdev + self.mean


class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution with different kernel sizes."""
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super().__init__()
        self.num_kernels = num_kernels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return torch.stack([conv(x) for conv in self.convs], dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_size, num_kernels, top_k):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.top_k = top_k
        intermediate = hidden_size * 4
        self.conv = nn.Sequential(
            InceptionBlock(hidden_size, intermediate, num_kernels),
            nn.GELU(),
            InceptionBlock(intermediate, hidden_size, num_kernels),
        )

    def forward(self, x):
        B, T, D = x.size()
        # FFT to find dominant periods
        xf = torch.fft.rfft(x, dim=1)
        freq_amp = xf.abs().mean(dim=(0, -1))
        freq_amp[0] = 0  # ignore DC
        _, top_idx = torch.topk(freq_amp, self.top_k)
        periods = T // top_idx.detach().cpu().numpy()
        period_weight = xf.abs().mean(dim=-1)[:, top_idx]

        # Process each period
        results = []
        for p in periods:
            if T % p != 0:
                pad_len = ((T // p) + 1) * p - T
                out = F.pad(x, (0, 0, 0, pad_len))
            else:
                pad_len = 0
                out = x
            out = out.reshape(B, -1, p, D).permute(0, 3, 1, 2)  # [B, D, rows, p]
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :T, :]
            results.append(out)

        results = torch.stack(results, dim=-1)
        weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).expand_as(results)
        return (results * weights).sum(-1) + x


class Custom(nn.Module):
    """TimesNet: Temporal 2D-Variation Modeling baseline.

    Transforms 1D time series to 2D based on detected periodicity,
    then applies 2D Inception convolution to capture temporal patterns.
    """

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.output_len = config.output_len
        self.revin = RevIN()

        # Embedding: feature -> hidden
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.value_embed = nn.Conv1d(
            config.num_features, config.hidden_size,
            kernel_size=3, padding=padding, padding_mode="circular", bias=False)

        # Temporal alignment for forecasting
        total_len = config.input_len + config.output_len
        self.predict_linear = nn.Linear(config.input_len, total_len)

        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(config.input_len, config.output_len,
                       config.hidden_size, config.num_kernels, config.top_k)
            for _ in range(config.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Output projection
        self.projection = nn.Linear(config.hidden_size, config.num_features)

    def forward(self, inputs, inputs_timestamps):
        x = self.revin(inputs, "norm")

        # Embed: [B, T, N] -> [B, T, D]
        h = self.value_embed(x.transpose(1, 2)).transpose(1, 2)

        # Extend to input_len + output_len
        h = self.predict_linear(h.transpose(1, 2)).transpose(1, 2)

        for block in self.blocks:
            h = self.layer_norm(block(h))

        pred = self.projection(h[:, -self.output_len:, :])
        pred = self.revin(pred, "denorm")
        return pred


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.001}
```
