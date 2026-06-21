The patch Transformer landed at mean F1 0.8135, but its per-dataset numbers tell a sharper story. PSM is fine (0.9617, recall 0.9361) — smooth, strongly periodic server metrics the patch tokens reconstruct cleanly. But MSL came in at 0.7904 (recall 0.7130) and SMAP at 0.6883 (recall a poor 0.5557). The recall is the tell: on both telemetry datasets precision stays high (when it flags, it is usually right) but recall sags, which for a reconstruction detector means the model is reproducing the *abnormal* points almost as well as the normal ones, so their error never clears the threshold. That is the signature of an over-flexible reconstructor — a 512-wide, two-layer attention encoder with a flatten head over $D\cdot N = 512\cdot 12$ features is powerful enough to partly fit anomalies, smoothing the very error spikes the score depends on. So the move now is not more capacity but *less*, with the capacity spent only on the structure that genuinely characterizes a normal window: its trend and its periodicity.

I propose **DLinear**: a deliberately under-flexible reconstructor that splits the window into a moving-average trend and a seasonal residual, reconstructs each with its own one-layer linear map along time, and sums them. The reasoning starts from what carries the reconstructable signal. A normal window is, to first order, a slow trend-cyclical component plus an oscillation riding on top, and both pieces are at heart "the value at a step is a fixed linear functional of the rest of the window": a trend is captured by extrapolating a slow drift (a weighted combination of values), a periodicity by reading off the same-phase values (again a weighted combination). So the most minimal reconstructor that can represent normal structure is a single linear map along time — for one channel, $\hat{X} = W X$ with $W$ a $\text{seq\_len}\times\text{seq\_len}$ matrix, every input step connected to every output step with a learned weight, signal-path length one, no recurrence to forget through, no attention to be distracted by the dominant normal mass. The decisive property is that a linear map has *exactly* the capacity to reproduce trend-plus-periodicity and almost none to fit an anomaly's idiosyncratic shape. Where the patch Transformer reconstructed abnormal points too faithfully and killed recall, a linear reconstructor reproduces the normal periodic structure and leaves the anomaly as a clean residual error spike. The cure for the over-flexible backbone is a deliberately under-flexible one.

A single linear map, though, fights itself when a window has a strong trend *and* a seasonal oscillation, which is the common case. One $W$ must fit both, and they want opposite weight patterns: the trend wants broadly smooth weights extrapolating a slow drift, the seasonality wants weights sharply concentrated at the periodic lags. Worse, the trend component is large in magnitude and the seasonal small, so in a squared-error fit the trend dominates the gradient — $W$ spends itself getting the big ramp roughly right and under-fits the small oscillation that carries the fine structure. And the fine oscillation is exactly where anomalies show: an anomaly is usually a break in the *shape*, not a gross level move. So I separate the components the classical way — seasonal-trend decomposition. Decompose the window once with a moving average: $\text{trend} = \text{MovingAvg}(x)$, $\text{seasonal} = x - \text{trend}$. The trend stream is the smooth baseline, the seasonal stream the oscillation around it, cleanly separated and on comparable footing within each stream. Give each its own linear map — $\text{Linear\_Seasonal}$ and $\text{Linear\_Trend}$, both $\text{seq\_len}\to\text{seq\_len}$ — so the trend linear learns to reproduce a slow baseline and the seasonal linear learns to read off the periodic lags, neither contaminated by the other and neither's gradient swamped by the other's magnitude. Then sum the two reconstructed streams.

I should be honest that this looks like added depth, which would betray the point of going simpler. But two linear maps plus a sum are affine, and a moving average is itself linear, so decompose-then-two-linears-then-sum is end to end a *single* affine map from window to reconstruction — no added representational capacity in the function-class sense. What I added is *conditioning*: the decomposition is a fixed, parameter-free reparameterization that separates the loud trend from the quiet seasonality so gradient descent on the squared error fits each well instead of letting the trend's magnitude dominate. It is preconditioning, not depth — same solution set, far better-behaved learning — and it pays off precisely when there is a clear trend, the case the bare linear map handled worst. The model stays as weak in capacity as I want it (it cannot fit an anomaly's idiosyncratic shape) while reconstructing normal trend-plus-seasonality cleanly.

I keep the channel decision the patch backbone got right, expressed here as one shared map. The channels share their weights — one $W$ applied identically to every channel — rather than a separate map per channel, because within a single dataset the channels usually share temporal dynamics (the daily rhythm of all server metrics, the diurnal cycle of all telemetry sensors). A shared map encodes that prior, slashes the parameter count, and keeps the model from fitting spurious cross-channel coincidences in the normal training data — the same bet the patch backbone's channel-independence made.

Two details decide whether it helps. The decomposition must be length-preserving and well-behaved at the edges: a stride-1 average pool over length $L$ with kernel $k$ produces $L-k+1$ outputs, so I pad — but zero-padding would drag the trend toward zero at the two ends, creating spurious dips where I have least information and hence false anomalies at every window boundary. Instead I replicate the endpoints, padding the front with $(k-1)/2$ copies of the first value and the back with $(k-1)/2$ copies of the last, so with $k$ odd the length is restored exactly and the trend stays flat-but-faithful at the edges. I keep the established smoothing scale (kernel 25), which averages out sub-cycle wiggles while preserving the cycle-and-slower trend, reusing the harness's `series_decomp` rather than tuning a new knob. The second wrinkle is scaffold-specific: for anomaly detection the harness sets $\text{pred\_len} = \text{seq\_len}$, so both maps are $\text{seq\_len}\to\text{seq\_len}$, and — unlike the patch backbone and the next rung — DLinear applies **no** reversible instance normalization. The TS-Lib anomaly path feeds the raw (already per-dataset Z-scored) window straight into the decomposition because the moving-average split itself handles the level: the trend linear absorbs the window's baseline directly. The weights are initialized to the uniform $1/\text{seq\_len}$ average (every entry equal), so each linear starts as an identity-ish averaging map and learns away from it — a sensible warm start for a reconstructor.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """Decomposition + two linear maps, reconstruction, channel-shared."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # for anomaly detection (reconstruction) the output length equals the input length
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' \
                or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # parameter-free seasonal-trend split (length-preserving moving average, replicate-padded)
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:                       # one linear pair per channel (rarely needed)
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:                                     # default: weights shared across variates
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x):                          # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        # move time to the last axis so the linear maps act along time (L -> L)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output         # recombine the streams
        return x.permute(0, 2, 1)                  # [B, L, C]

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)                 # [B, L, C] reconstruction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                         # [B, L, D]
        return None
```
