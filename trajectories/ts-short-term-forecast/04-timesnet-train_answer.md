The multi-scale mixer paid off where I bet it would: TimeMixer came in at SMAPE 12.80 on Monthly, 10.21 on Quarterly, 13.38 on Yearly, beating PatchTST (12.97 / 10.22 / 13.68) on all three, with the biggest gains on Monthly and especially Yearly — the regimes where trend and season live at different resolutions and top-down trend mixing had real work to do. But look at Quarterly: TimeMixer's 10.21 was barely below PatchTST's 10.22, essentially flat. That is the regime with the *cleanest* single dominant period (the 4-quarter cycle) and the shortest window after Yearly (16 steps, two cycles), so one average-pooling step to a coarse 8-step view does not separate much. The fixed pooling ladder is the wrong lens when the structure is *one sharp period* rather than a trend/detail split across scales. TimeMixer exposes scale by *blind* pooling — it never asks which periodicity a series carries; it just halves the length and hopes trend and season fall out. On a single-strong-period series, the informative thing is not "coarse vs fine," it is "the value one period back," and a pooling ladder represents that only incidentally.

So the move is to discover the period structure *inside the window* and represent the series around it. Consider what dependencies a forecast point actually has. It depends on its immediate neighbors — the local shape within the current cycle, short-range, *intraperiod*. It also depends on the same phase one cycle ago and one cycle ahead — the same quarter last year, the same month in adjacent years — which is how that phase changes from cycle to cycle, long-range, *interperiod*. These are two genuinely different dependencies, and real M4 series carry both, often more than one period at once. The structural problem is that the data is a 1D sequence indexed by $t$: adjacency along that axis gives the intraperiod neighbors for free ($t-1$, $t+1$), but the interperiod neighbor at $t-p$ is $p$ steps away with an entire cycle crammed in between. A 1D convolution with any sane kernel never sees $t$ and $t-p$ together; an MLP over absolute window position (DLinear, TimeMixer's per-scale predictors) has no handle on *which* period a series carries; attention relates point pairs but, over a window of mostly ordinary points, the similarity is dominated by them. Every tool so far inherits the same bottleneck: the 1D layout presents intraperiod variation as locality but effectively hides interperiod variation. That is exactly why Quarterly stalled — the one-period-back relationship that *is* the Quarterly signal is never made local.

I propose **TimesNet**: discover the dominant periods of the window with an FFT, reshape the 1D series around each period into a 2D grid so that both kinds of locality appear at once, and process the grid with a 2D inception convolution. For a period $p$, chop the series into consecutive blocks of length $p$ and stack them as rows of a 2D array: walking along a row traverses one cycle (intraperiod shape), walking down a column traverses the same phase across successive cycles (interperiod change). The point that was $p$ apart in 1D is now one step apart along the cross-period axis, so *both* dependencies become adjacencies a 2D convolution reads simultaneously — the thing a 1D layout and a pooling ladder structurally cannot do.

Which $p$? Discover it, do not assume it. Take the FFT of the window; the amplitude spectrum tells how much energy sits at each frequency, and the peaks are the dominant periodicities. Average the amplitude over channels (one here), zero the DC term (it is the window mean, not a period), and take the top-$k$ frequencies by amplitude. Each frequency $f$ gives a period $p = \lfloor T/f \rfloor$ and an amplitude that is its *confidence*. Top-$k$ rather than all frequencies because the spectrum of a short real series is sparse and small-amplitude bins are noise. On Quarterly the FFT should put a sharp peak at the 4-step period and reshape the window into a grid whose columns are within-quarter shape and whose rows are year-over-year change — making the one-period-back relationship local for the first time on the ladder. There is a subtlety with short windows I do not gloss over: the amplitude spectrum of a real series is conjugate-symmetric, so I look only at non-negative frequencies up to $T/2$, and $p = \lfloor T/f \rfloor$ can collapse distinct frequencies onto the same integer period or onto periods so short (2 or 3) that the 2D reshape is a degenerate grid of many two-column rows. That is fine — a degenerate short-period grid lets the inception block act almost like a 1D convolution, so the model gracefully falls back to local smoothing where there is no real long period; the genuine win only materializes when a real period (Quarterly's 4, Monthly's 12) has enough cycles to fill several rows.

For each discovered period, reshape (zero-padding the time axis up to a multiple of $p$ first), then process the 2D tensor with a parameter-efficient inception block — several 2D kernels of increasing size in parallel, averaged — so the block reads intra- and interperiod variation at multiple 2D scales at once. One inception is **shared across all $k$ periods**, so model size is independent of $k$; reshape back to 1D and truncate. Fuse the $k$ period-specific representations by a softmax over their amplitudes — amplitude is the confidence of each period, so a convex combination weighted by confidence is the principled aggregation (the same confidence-softmax idea the auto-correlation forecasters used, now over genuinely separate 2D representations rather than 1D rolls). Stack a couple of these TimesBlocks residually with LayerNorm, wrapped in the reversible per-window instance normalization that has helped since PatchTST.

For forecasting specifically the horizon has to be *created* before the period machinery runs, because the future steps do not exist in the input window. So after instance-normalizing and embedding the length-$\text{seq\_len}$ window to $d_{\text{model}}$ (a value embedding plus positional embedding; the harness passes no time marks, so the temporal-feature branch contributes nothing and the embedding must accept `x_mark=None`), one linear map along time extends the sequence $\text{seq\_len} \to \text{seq\_len} + \text{pred\_len}$. The TimesBlocks then operate on the full extended sequence, discovering periods over $\text{seq\_len} + \text{pred\_len}$ and convolving across it, so the forecast region is filled by the same period-aware 2D convolution that models the observed region; a final linear projection maps $d_{\text{model}} \to 1$, and the output is denormalized and truncated to the last $\text{pred\_len}$ steps.

On the protocol: the fixed Custom settings give $d_{\text{model}}=512$, $d_{\text{ff}}=512$, $e_{\text{layers}}=2$, with unset $\text{top\_k}/\text{num\_kernels}$ defaulting to 5 and 6 — much wider than TimesNet's own short-term script ($d_{\text{model}}=32$). On very short windows the FFT has few bins: Yearly's window is 12 steps (extended to 18), so the spectrum is coarse, the top-5 periods include near-meaningless short ones, and the 2D reshape is thin. So I expect the period machinery to deliver most where the window holds several clean cycles of a real period and least where there is essentially no period to find. The falsifiable expectation against TimeMixer's numbers is therefore *non-uniform* — this is not a rung I expect to win everywhere. I am most confident on **Quarterly**, the regime that stalled for TimeMixer, because the FFT should lock onto the sharp 4-step period and make the interperiod relationship local — I expect to clear 10.21 with the clearest margin of the three. **Monthly** I expect roughly tied near 12.80: the 12-step seasonality is real and the 36-step window holds three cycles so the 2D reshape is informative, but TimeMixer's decomposition-mixing already handles the trend+season superposition well, so two strong period-aware models land close. **Yearly** is where I expect *not* to beat TimeMixer: there is no meaningful period in a trend-dominated six-step horizon, so FFT period-discovery has little to discover and top-down trend mixing is the better tool — I would not be surprised to come in slightly above 13.38. So the bar this rung must clear to be the strongest baseline is a *mean* win driven by Quarterly outweighing a possible Yearly loss, which would confirm that the binding constraint was making interperiod structure local.

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
    frequency_list = abs(xf).mean(0).mean(-1)      # amplitude over batch and channels
    frequency_list[0] = 0                          # drop DC (the mean, not a period)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # one shared inception, reused for every period -> size independent of k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
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
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)                            # [B, T, N, k]
        # adaptive aggregation by amplitude-softmax
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                            # residual


class Model(nn.Module):
    """TimesNet: FFT period discovery + 2D inception conv, forecasting."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)                 # [B, seq_len, d_model]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # extend time
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))          # residual TimesBlocks
        dec_out = self.projection(enc_out)                             # [B, seq_len+pred_len, c_out]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
