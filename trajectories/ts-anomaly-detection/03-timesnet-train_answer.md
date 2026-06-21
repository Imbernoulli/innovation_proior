DLinear's numbers confirmed the diagnosis and exposed the next one. Its mean F1 (0.8194) cleared the patch backbone (0.8135) exactly where I expected: MSL jumped from 0.7904 to 0.8187 (recall 0.7130 to 0.7530) and PSM ticked up to 0.9663 — the under-flexible linear reconstructor left the steady-period anomalies as cleaner residual. But SMAP went the *other* way, F1 falling from 0.6883 to 0.6733 and recall to 0.5383, the worst on the board. That is the tell. A single linear map is one flat, position-indexed function of the window: the weight $W[i,j]$ for "reconstruct step $i$ from step $j$" is fixed regardless of *which* period the current window carries. On PSM and MSL the dominant period is steady enough that one fixed map works; on SMAP the satellite telemetry shifts its rhythm window to window, so a single fixed lag pattern is wrong for most windows, the reconstruction smears the periodic structure, and anomalies stop standing out. The linear map has no handle on which periodicities a given window carries.

To see what must change, I have to be precise about the dependency a point in a normal window actually has. A point now depends on its immediate neighbors — the short-term movement *within* the current cycle — but also on the same phase one cycle back and one cycle ahead — how that phase *changes from period to period*. Call these intraperiod variation (within a cycle, short-range) and interperiod variation (across cycles, same-phase, long-range). Real telemetry has both, and several periods at once. The trouble is the layout: the data is a 1D sequence indexed by $t$, where adjacency gives me the intraperiod neighbors $t-1, t+1$ for free, but the interperiod neighbor — the same phase one period back — sits at $t-p$ with the whole previous cycle crammed in between. A linear map can place weight at a *fixed* lag $p$; a convolution never sees $t$ and $t-p$ together; attention can reach $t-p$ but treats it as just another of $L$ points with no notion that it is the same phase. The 1D representation presents one variation cleanly and hides the other, and the linear map's SMAP failure is exactly that bottleneck made visible.

I propose **TimesNet**, a period-aware reconstruction backbone built on one idea: change the *layout* so both variations become locality. For a period $p$, chop the window into consecutive blocks of length $p$ and stack them as rows of a matrix. Walking along a row is walking through one cycle (intraperiod); walking down a column is walking through the same phase across successive cycles (interperiod). The point that was $p$ apart in 1D is now one step apart along the cross-period axis. Both dependencies are adjacencies in a 2D array — and the instant the window is a 2D grid with two meaningful local axes, the entire mature toolbox of 2D convolution applies: one kernel sliding over the grid sees, in a single receptive field, a few adjacent steps within the cycle *and* the same band of phases in neighboring cycles, modeling intra- and interperiod variation simultaneously, which no 1D operator could.

But which $p$? Windows have several periods at once, unknown a priori and shifting window to window — exactly the variability that broke the fixed map. So I *discover* the periods per window from the spectrum. The amplitude of the Fourier transform at frequency $j$ measures how strongly a periodic basis of period $T/j$ is present, so I take the real FFT along time, its amplitude, average over batch and channels into one length-$T/2{+}1$ profile, and read off the dominant frequencies. Pooling spectral energy across channels is deliberate: with 25–55 channels an anomaly may show in one channel while the periods are best estimated jointly, and I want the *same* reshape applied to every channel so they stay phase-aligned in the grid. Three details matter. The DC term (frequency 0) is the window's mean energy, corresponds to no period and is usually huge, so I zero it before picking peaks. A real signal's spectrum is conjugate-symmetric, so the real FFT's frequencies up to $T/2$ are all I consider. And the spectrum is sparse with a noisy high-frequency tail, so I keep only the top-$k$ amplitudes, giving periods $p_i = T \,/\!/\, f_i$ — and I keep the $k$ amplitudes around to serve as importance weights. This `FFT_for_Period` is the per-window period discovery the linear map lacked.

The reshape must come out exactly. For period $p_i$ I want a grid of $(\text{length}/p_i)$ rows and $p_i$ columns, but $T$ is generally not a multiple of $p_i$, so I zero-pad along time up to the next multiple of $p_i$, reshape to $(\text{length}/\!/p_i)\times p_i$, process, then truncate back to $T$ — the padding is harmless filler the truncation discards. Over each 2D grid I run not a single kernel but a multi-scale one, because the variation has structure at several scales — a fine wiggle over two or three steps, a broader hump over a third of the cycle, a slow tilt across cycles. That is what the Inception block was built for: parallel convolutions of increasing kernel size $2i+1$, each padded to preserve size, run together and mean-combined — multi-scale by construction and far cheaper than one giant kernel. I make the block a small two-layer feedforward in 2D: inception expanding $d_{\text{model}}\to d_{\text{ff}}$, a GELU, inception contracting $d_{\text{ff}}\to d_{\text{model}}$. One decision keeps the model size clean: I share *one* inception across all $k$ periods. Each period gets a different reshape (different grid geometry) but the same conv weights, so model size is invariant to $k$ — and conceptually right, since the inception learns "how to read 2D temporal variation," which should not depend on which period produced the grid.

After the shared inception transforms each grid, I reshape each back to 1D and truncate to $T$, giving $k$ candidate representations. Fusing them by a plain sum would throw away that some periods are far more present in this window than others — and "more present" is exactly the per-window adaptivity the fixed map lacked. The amplitude *is* how strongly a period is expressed, so I push the $k$ amplitudes through a softmax over the $k$ periods to get convex weights and take the weighted sum: a window dominated by one cycle puts most weight on that period's representation, a window with two strong rhythms splits it, and a SMAP window whose rhythm shifts gets a *different* convex combination than the previous window. The softmax's normalization is the principled choice over raw, scale-sensitive amplitudes. This whole pipeline — discover periods, reshape to $k$ grids, shared inception on each, reshape back, amplitude-softmax aggregate — is one TimesBlock. I stack several with a residual $X^l = \text{TimesBlock}(X^{l-1}) + X^{l-1}$ and a LayerNorm, so each block learns only the *correction* to the variation representation and the stack refines with depth (early blocks coarse, later ones fine).

The reconstruction path matches the prior two rungs: I reconstruct the window I was given, not a future horizon, so $\text{pred\_len}$ is 0, the pad/reshape uses just $\text{seq\_len}$, and there is no decoder. I embed the window into a $d_{\text{model}}$ sequence (a value embedding — a 1D conv lifting channels to $d_{\text{model}}$ — plus a fixed positional embedding, no time marks), run the residual TimesBlocks over its own length, and project every position back to the input channels. One thing DLinear deliberately omitted comes back here: because channels and datasets span orders of magnitude and the level wanders, I wrap the backbone in reversible per-window instance normalization — subtract the temporal mean, divide by the std (biased variance plus $1\mathrm{e}{-5}$), detach both, run the model, then de-normalize the output with the same statistics. DLinear could skip this because its trend linear absorbed the level; a deep conv backbone cannot, so the normalization returns for the same reason the patch backbone needed it. Per-window centering could partly absorb a pure level-shift anomaly, but the data is already globally Z-scored, the windows are short, and the anomalies that matter are violations of the *variation* — cycle breaks, shape going wrong — which survive centering precisely because they are deviations from the periodic structure the model reconstructs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)     # amplitude over batch and channels
    frequency_list[0] = 0                         # drop DC (the mean, not a period)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len          # 0 for anomaly detection
        self.k = configs.top_k
        # one shared inception, reused for every period -> size independent of k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()                                        # T = seq_len (pred_len = 0)
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # [B, length, N] -> [B, N, num_periods, period]: cols=intraperiod, rows=interperiod
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                                  # multi-scale 2D conv
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)       # back to 1D
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)                            # [B, T, N, k]
        # adaptive aggregation by amplitude-softmax
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                            # residual


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # reconstruction: project the representation straight back to the input channels
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)                 # [B, seq_len, d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))     # residual TimesBlocks + LayerNorm
        dec_out = self.projection(enc_out)                        # [B, seq_len, c_out]

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                                        # [B, seq_len, c_out] reconstruction
        return None
```
