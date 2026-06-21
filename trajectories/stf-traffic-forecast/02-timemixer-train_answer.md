The linear floor told me exactly where it sits, in numbers: DLinear landed MAE 4.0599 on METR-LA, 2.1334 on PEMS-BAY, and 28.4748 on PEMS04. That a single affine map per node gets this far confirms the floor's premise — most of the forecastable structure on these roads is trend plus daily phase. But the flow dataset, PEMS04, is the worst of the three by a wide relative margin, and the reason is specific: flow spikes sharply at the commute peaks and is less smoothly periodic than speed, so a single moving-average split at the native five-minute resolution cannot cleanly separate the commute-spike structure from the slow daily drift when both live inside one twelve-step window. DLinear decomposes *once*, at *one* resolution, with *one* kernel. The structure it leaves on the table is the part that only becomes legible when the same series is read at a *different* temporal scale — a gap I can attack while staying almost as cheap as the floor, before spending any capacity on cross-node coupling.

The driving observation is that one physical process presents a completely different pattern at each sampling scale, because averaging is a low-pass filter: the same road sampled every five minutes shows a sharp commute spike; averaged coarser, that spike softens into weekday-versus-weekend structure; averaged further, only a slow seasonal drift remains. Fine and coarse views are not redundant copies — they are complementary descriptions, and the floor throws all but one away. There is a second fact specific to forecasting: the next hour depends *jointly* on immediate five-minute momentum and on the slow drift of the day, and a fine-view predictor and a coarse-view predictor have different *skills* — the fine one good at the next few high-frequency wiggles, the coarse one at the slow drift. Collapsing everything into one representation before the prediction head merges the skills before I can use them separately. So the shape of the method is forced: build the series at several scales, mix across scales while keeping them separate, then predict from each scale and combine the forecasts.

I propose TimeMixer: decomposable multiscale mixing. The scales come from a downsampling ladder $x_0, x_1, x_2$, where $x_0$ is the raw twelve-step input and $x_{m+1}$ is an average pool of $x_m$ with window 2 — lengths $12 \to 6 \to 3$. I deliberately use a *parameter-free* average pool rather than a learned strided convolution, because a coarsener's whole job is to be a faithful change of resolution, not to bake in a learned filter that distorts what "coarse" even means before the rest of the model sees it; max-pool is wrong too, since max tracks peaks rather than the central tendency a coarse view should show. Each scale is embedded to $D$ channels. The core is how the scales talk to each other, and the key move is to *decompose first, at every scale*, with the same replicate-padded moving-average split DLinear used, because even the coarsest scale is not pure trend — a coarsely-averaged traffic series still carries seasonality on its drift. So inside each block, for every scale $m$, $s_m, t_m = \text{SeriesDecomp}(x_m)$, and I mix the clean components rather than the tangled raw scales.

The load-bearing design choice is that season and trend mix with *opposite information flow*. A coarse-scale season is an *aggregation* of finer seasons — a weekly cycle is several daily cycles stacked — so the detail that defines a coarse seasonal pattern lives down in the fine scale, and I push seasonality *upward*, fine $\to$ coarse:
$$s_m \leftarrow s_m + \text{BottomUpMixing}(s_{m-1}).$$
For trend the asymmetry reverses. Fine-scale detail is the *enemy* of a slow drift — the wiggles I keep for season are exactly the noise I suppress for trend — so the clean read on the trend comes from the coarse scale, and information flows the other way, coarse $\to$ fine:
$$t_m \leftarrow t_m + \text{TopDownMixing}(t_{m+1}).$$
This is structural, not aesthetic: mixing season top-down would smear the detail-poor coarse view onto the fine scale and destroy the microscopic seasonal information that *was* the fine scale's value; mixing trend bottom-up would inject fine wiggles into the coarse trend, the exact noise its cleanliness depended on. Both reversed directions attack what made each scale useful. The mixing primitive is a two-layer MLP with a GELU between, acting along the temporal length — not a single linear resample, because a linear map can only rescale, whereas I want the finer season *transformed* into a useful supplement. Each mix is residual, keeping each scale's own component and *adding* the cross-scale contribution. After seasonal and trend mixing I recombine per scale, $\text{out}_m = s_m + t_m$, run a small cross-channel FFN over the $D$ channels (everything so far mixed along time, not channels) with its own residual, and wrap the block in a residual on the original scale features. That is one Past-Decomposable-Mixing block; I stack two so repeated re-decomposition gives deeper cross-scale interaction without ballooning cost.

For the prediction I refuse to collapse the scales. Each scale, still at its own length, gets its own head: one linear layer maps that scale's temporal length straight to the twelve-step horizon — the DLinear move I know is strong — then a channel-independent projection to one value per node, giving a full $[B,12,N]$ forecast from each scale alone. The final forecast is the *sum* over scales, a deliberately heterogeneous ensemble where the fine predictors carry seasonal detail and the coarse predictors carry the macro drift; sum and average differ only by a constant the network absorbs, so I use the plain sum. Two more pieces the harness and the many-node setting force. First, per-scale reversible instance normalization (RevIN): traffic level and scale drift across the window, so I standardize each input window per node — subtract its mean, divide by its std, invert on the output — and I do it *per scale*, because averaging changes the variance, with an affine term to re-introduce a learned level. Second, channel independence: with 207–325 loosely-coupled nodes, embedding them jointly would overfit spurious cross-node correlations and tie the embedding width to $N$, so after per-scale normalization I reshape $[B,L_m,N]$ into $[B{\cdot}N, L_m, 1]$, treat every node as its own univariate sequence sharing one set of weights, and project back at the end. This keeps the model size independent of $N$ and, like the floor, models *no* explicit cross-node coupling — the deliberate continuity with DLinear, so the comparison isolates whether multi-scale *temporal* modeling alone is worth its cost. The deeper MLP stack is unstable at the harness default rate, so I halve it via `CONFIG_OVERRIDES = {'lr': 0.001}`, leaving `weight_decay` at the default. I expect the largest gain on PEMS04, where fine spike and coarse drift are both strong and tangled, though the twelve-step window leaves the coarsest scale only length three, which may mute the scale-axis advantage everywhere.

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
    hidden_size: int = field(default=64)
    num_layers: int = field(default=2)
    down_sampling_layers: int = field(default=2)
    down_sampling_window: int = field(default=2)
    dropout: float = field(default=0.1)
    moving_avg: int = field(default=5)


class RevIN(nn.Module):
    def __init__(self, num_features, affine=True):
        super().__init__()
        self.eps = 1e-6
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        else:
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            return x * self.stdev + self.mean


class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1)
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

    def forward(self, x):
        # x: [B, T, C]
        trend = self.avg(F.pad(x.transpose(1, 2),
                                (self.pad_left, self.pad_right),
                                mode='replicate')).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class MLPMixer(nn.Module):
    '''2-layer MLP for scale mixing on the temporal dimension.'''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiScaleSeasonMixing(nn.Module):
    '''Bottom-up mixing: finer -> coarser scales via learned MLP.'''
    def __init__(self, input_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.down_layers = nn.ModuleList([
            MLPMixer(
                input_len // (down_sampling_window ** i),
                input_len // (down_sampling_window ** (i + 1))
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, seasonal_list):
        # seasonal_list[i]: [B, D, L_i] (permuted)
        out_high = seasonal_list[0]
        out_low = seasonal_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(seasonal_list) - 1):
            out_low_res = self.down_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(seasonal_list) - 1:
                out_low = seasonal_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    '''Top-down mixing: coarser -> finer scales via learned MLP.'''
    def __init__(self, input_len, down_sampling_layers, down_sampling_window):
        super().__init__()
        self.up_layers = nn.ModuleList([
            MLPMixer(
                input_len // (down_sampling_window ** (i + 1)),
                input_len // (down_sampling_window ** i)
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        # trend_list[i]: [B, D, L_i] (permuted)
        trend_rev = trend_list.copy()
        trend_rev.reverse()
        out_low = trend_rev[0]
        out_high = trend_rev[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_rev) - 1):
            out_high_res = self.up_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    '''Decompose each scale, mix seasonal bottom-up and trend top-down.'''
    def __init__(self, input_len, hidden_size, down_sampling_layers,
                 down_sampling_window, moving_avg):
        super().__init__()
        self.decomp = MovingAvgDecomp(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(
            input_len, down_sampling_layers, down_sampling_window)
        self.trend_mixing = MultiScaleTrendMixing(
            input_len, down_sampling_layers, down_sampling_window)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x_list):
        seasonal_list, trend_list = [], []
        for x in x_list:
            seasonal, trend = self.decomp(x)
            seasonal_list.append(seasonal.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        seasonal_list = self.season_mixing(seasonal_list)
        trend_list = self.trend_mixing(trend_list)

        out_list = []
        for x, seasonal, trend in zip(x_list, seasonal_list, trend_list):
            out = seasonal + trend
            out = x + self.out_cross_layer(out)
            out_list.append(out)
        return out_list


class Custom(nn.Module):
    '''TimeMixer: Decomposable Multiscale Mixing baseline.

    Channel-independent mode: each variate processed separately.
    Multi-scale decomposition + Past-Decomposable Mixing across scales.
    '''

    def __init__(self, config: CustomConfig):
        super().__init__()
        self.num_features = config.num_features
        self.output_len = config.output_len
        self.down_layers = config.down_sampling_layers
        self.down_window = config.down_sampling_window
        D = config.hidden_size

        self.down_pool = nn.AvgPool1d(config.down_sampling_window)

        # Per-scale RevIN
        self.norm_layers = nn.ModuleList([
            RevIN(config.num_features, affine=True)
            for _ in range(self.down_layers + 1)
        ])

        # Embedding (channel-independent: 1 feature -> D)
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.embed = nn.Conv1d(1, D, kernel_size=3, padding=padding,
                               padding_mode="circular", bias=False)

        # PDM blocks (decomposition happens inside each block)
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                config.input_len, D, self.down_layers,
                self.down_window, config.moving_avg)
            for _ in range(config.num_layers)
        ])

        # Per-scale prediction heads
        self.predict_layers = nn.ModuleList([
            nn.Linear(config.input_len // (self.down_window ** i), config.output_len)
            for i in range(self.down_layers + 1)
        ])

        # Channel-independent projection
        self.projection = nn.Linear(D, 1)

    def forward(self, inputs, inputs_timestamps):
        # inputs: [B, T, N]
        B, T, N = inputs.shape

        # Multi-scale inputs
        x_list = [inputs]
        sample = inputs.permute(0, 2, 1)  # [B, N, T]
        for _ in range(self.down_layers):
            sample = self.down_pool(sample)
            x_list.append(sample.permute(0, 2, 1))

        # Per-scale normalization + channel independence
        for i in range(len(x_list)):
            x_list[i] = self.norm_layers[i](x_list[i], "norm")
            _, Li, _ = x_list[i].shape
            x_list[i] = x_list[i].transpose(1, 2).reshape(-1, Li, 1)  # [B*N, Li, 1]

        # Embedding
        h_list = []
        for x in x_list:
            h = self.embed(x.transpose(1, 2)).transpose(1, 2)  # [B*N, Li, D]
            h_list.append(h)

        # Past Decomposable Mixing (decomposition inside blocks)
        for block in self.pdm_blocks:
            h_list = block(h_list)

        # Per-scale prediction and sum
        pred_list = []
        for i, h in enumerate(h_list):
            # h: [B*N, Li, D] -> predict -> [B*N, T', D] -> project -> [B*N, T', 1]
            p = self.predict_layers[i](h.permute(0, 2, 1)).permute(0, 2, 1)
            p = self.projection(p)  # [B*N, T', 1]
            p = p.reshape(B, N, self.output_len).permute(0, 2, 1)  # [B, T', N]
            pred_list.append(p)

        prediction = sum(pred_list)
        prediction = self.norm_layers[0](prediction, "denorm")
        return prediction


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: lr, weight_decay.
CONFIG_OVERRIDES = {'lr': 0.001}
```
