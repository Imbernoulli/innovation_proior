The task in front of me is unsupervised anomaly detection on multivariate monitoring streams — server metrics, spacecraft and satellite telemetry — where labels are useless because anomalies are rare and labeling at scale is hopeless. The standard way to make this operational is reconstruction: train a model only on normal windows to reproduce them faithfully, then at test time flag the points it reconstructs poorly, because a model that has only ever seen normal variation should choke on the abnormal. The anomaly score is just the per-point reconstruction error, and so the whole problem collapses to one thing — how well can I model the temporal variation of a normal window? A better variation model gives a tighter reconstruction on normal points and a cleaner spike on anomalies.

Modeling that variation is the genuine difficulty, because a single time point is only a handful of scalars and carries almost no standalone meaning; the information lives entirely in how the signal moves. Let me be precise about what dependencies a point actually has, since vague talk of "temporal variation" gets me nowhere. A point depends on its immediate neighbors — the short-range shape *within* the current cycle, which I will call intraperiod variation — and it also depends on the same phase one cycle ago and one cycle ahead — the long-range, same-phase relationship *across* cycles, which I will call interperiod variation. Real series carry both, and worse, they carry several periods at once (a server has daily and weekly rhythms; telemetry has several), so multiple intra/inter pairs are superimposed. The structural trap is that the data is a 1D sequence indexed by $t$: adjacency along that axis hands me the intraperiod neighbors $t-1$ and $t+1$ for free, but the interperiod neighbor sits at $t-p$ where $p$ is the period length, with an entire cycle crammed in between. A convolution along time with any sane kernel width never sees $t$ and $t-p$ together; to reach across a period it would need a kernel as wide as the period, and then it averages over everything in between instead of relating the two same-phase points. The 1D layout can present one variation as locality and effectively hides the other. This is why every tool on the shelf struggles in the same way: RNNs have to carry the dependency on $t-p$ through $p$ sequential state transitions, exactly the long-range washout they are known for, and they compute slowly; TCNs slide a 1D kernel and, even with dilation, only combine points near each other along time, so same-phase points a period apart are in no single kernel's window; MLP models like DLinear bake a single fixed linear map over absolute window position, shared across all series, with no handle on which periodicities a given series carries; and point-wise Transformer attention, used for reconstruction anomaly detection, is risky because a window is mostly normal points, so the similarity structure is dominated by normal-normal comparisons and the rare abnormal pattern gets washed out. Autoformer at least takes periodicity seriously — its Auto-Correlation estimates period lengths from the autocorrelation function via the FFT, rolls the series by those lags, and aggregates the rolled copies weighted by a softmax over their correlation confidences — but it still operates on the 1D series, folding the intraperiod shape and the interperiod link into one roll-and-correlate rather than treating them as two axes I can convolve over independently. None of these separates intra and inter into two distinct, simultaneously-modelable structures.

So the question sharpens: I have two kinds of locality, within a period and across periods, and a 1D layout that can express only one of them as locality. The resolution is to change the layout so that *both* become locality, and the method that does this I call TimesNet. Take the 1D sequence and a period $p$: chop the sequence into consecutive blocks of length $p$ and stack those blocks as the rows of a matrix. Now walking along a row traverses one cycle (the intraperiod, within-cycle shape) and walking down a column traverses the same phase across successive cycles (the interperiod, across-cycle structure). The point that was $p$ apart in 1D is now one step apart along the cross-period axis. Both dependencies have become adjacencies in a 2D array, and the instant the data is a 2D grid with two meaningful local axes I inherit the entire mature toolbox of 2D convolution: a kernel sliding over this grid sees, in one receptive field, a few adjacent time points within the cycle *and* the same window of phases in neighboring cycles, modeling intra and inter variation simultaneously — precisely what the 1D layout could never do with one kernel. The bottleneck was never that the variation is hard; it was that 1D was the wrong space to look at it in.

The one thing I do not know in advance is the period $p$, and it differs across data and across mini-batches, so I discover it from the spectrum. The amplitude of the Fourier transform at frequency $j$ measures how strongly a periodic basis function with about $T/j$ samples per cycle is present, so a strong period shows up as a tall amplitude peak. I compute the real FFT along time for the mini-batch of length-$T$, $C$-channel windows, average the amplitudes over the batch and channel axes into one shared peak list, and keep the per-example channel-averaged amplitudes at the selected frequencies for later weighting. Three details have to be right. The DC term, frequency $0$, is just the window's mean energy, corresponds to a meaningless infinite "period," and is usually huge, so I zero it out before picking peaks or it dominates the selection. The spectrum of a real signal is conjugate-symmetric, so I consider only frequencies up to $T/2$, which the real FFT gives directly. And the spectrum is sparse with a noisy high-frequency tail, so I keep only the top $k$ amplitudes, yielding the $k$ most significant nonzero frequencies $f_1,\dots,f_k$; the integer period the runnable block uses is $p_i = \lfloor T/f_i \rfloor$, with any mismatch absorbed by padding and later truncation. Concretely the reshape pads the series with zeros along time up to the next multiple of $p_i$, reshapes the padded length into $(\text{length}/p_i) \times p_i$, and truncates back to $T$ after processing — the padding is harmless filler the truncation discards. This gives $k$ different 2D views of the same window, each exposing a different periodicity's intra/inter structure.

What runs over each 2D tensor is a 2D convolution, but with a deliberate choice of kernel. The reshape fixes one period, yet the variation within and between cycles has structure at several scales — a fine wiggle over two or three steps, a broader hump over a third of the cycle, a slow tilt across several cycles — and a single fixed kernel size commits to just one. This is exactly what the Inception block was invented for in vision: run several kernel sizes in parallel inside one block and combine them, so the block is multi-scale by construction while staying parameter-efficient versus one giant kernel. I use a parameter-efficient inception block of 2D convolutions with kernel size $2i+1$ and padding $i$ for $i=0,\dots,\text{num\_kernels}-1$, run in parallel and mean-pooled across the kernel dimension, and I make it a small two-layer affair — inception expanding $d_{\text{model}} \to d_{\text{ff}}$, a GELU, then inception contracting $d_{\text{ff}} \to d_{\text{model}}$ — so it has the capacity to genuinely transform the 2D features rather than smear them. The decision that keeps the design clean is to share one inception block across all $k$ periods: each period gets a different reshape geometry, but the same conv weights process every view, so model size is invariant to $k$ and I can dial $k$ as a pure width-of-search knob. This is also conceptually right, since the inception is learning *how to read 2D temporal variation*, which should not depend on which period produced the grid.

After the shared inception transforms each 2D tensor, I reshape each back to 1D and truncate to $T$, leaving $k$ candidate representations, one per period, that I must fuse. A plain sum would throw away the fact that some periods are far more present in this window than others, but I already kept the amplitudes $A_{f_i}$, and an amplitude *is* a measure of how strongly that period is expressed — the same confidence-weighted aggregation Auto-Correlation used, with the FFT amplitude playing the role of confidence. Raw amplitudes are unnormalized and scale-sensitive, so I push them through a softmax over the $k$ periods to obtain convex weights and take the weighted sum,

$$X' = \sum_{i=1}^{k} \operatorname{softmax}(A_{f_1},\dots,A_{f_k})_i \cdot Y^{(i)}_{\text{1D}},$$

so a window dominated by a daily cycle puts most of its weight on that period's representation while a window with two strong rhythms splits the weight. One block therefore does: discover periods, reshape to $k$ 2D tensors, run the shared inception on each, reshape back, and amplitude-softmax aggregate — this is the TimesBlock. To refine the representation with depth I stack several of these, and because naively stacking transformations destabilizes training I wrap each in a residual connection with a layer norm,

$$X^{l} = \operatorname{LayerNorm}\big(\operatorname{TimesBlock}(X^{l-1}) + X^{l-1}\big),$$

which also means each block only has to learn the *correction* to the variation representation: early blocks catch coarse structure, later ones refine it.

The anomaly-detection setting then makes the surrounding architecture unusually simple. I am not predicting any future steps — I am reconstructing the window I was given — and a TimesBlock already maps a length-$T$ feature sequence to a length-$T$ feature sequence, which is exactly a reconstruction map. So there is no horizon to invent, no learned temporal extension, no decoder: the look-back and output lengths are the same $\text{seq\_len}$, $\text{pred\_len}=0$, and the pad/reshape inside the block uses that length with no future positions. I embed the window (value embedding plus a fixed sinusoidal positional embedding; no calendar marks are useful here, so the time-feature branch is simply unused), run the residual TimesBlocks over its own length, and project every position back to the input channels; the framework reads the anomaly score off the squared difference between reconstruction and input, point by point. The one remaining problem is distribution shift — channels and datasets span very different magnitudes and the level wanders over time — which I handle with per-window instance normalization: subtract the window's temporal mean, divide by its standard deviation (computed as the biased variance with a small $\epsilon=10^{-5}$ under the root for numerical safety, both statistics detached because they are normalization constants and not parameters to backprop through), run the backbone, then de-normalize the output with the same mean and std broadcast over the window length. This is what lets one shared backbone reconstruct heterogeneous multichannel streams. It is worth being honest that normalizing per window could partly absorb an anomaly that is purely a level shift, but the data is already globally Z-scored per dataset, the windows are short, and the anomalies I care about are violations of the *variation* — the cycle breaks, the shape goes wrong — which survive the per-window centering precisely because they are deviations from the periodic structure the model has learned to reconstruct.

Trained with Adam at learning rate $10^{-4}$, $(\beta_1,\beta_2)=(0.9,0.999)$, batch size 128, up to 10 epochs, on an MSE reconstruction objective over normal windows, with $k=3$, three residual layers, six inception kernels, $\text{seq\_len}=100$, and a model width sized to the channel count by $d_{\text{model}}=d_{\text{ff}}=\min(\max(2^{\lceil \log C \rceil}, 32), 128)$, the causal chain is complete: the anomaly score is reconstruction error, so the task reduces to modeling normal-window variation; that variation has two distinct components, short-range intraperiod and long-range interperiod, which the 1D layout cannot both express as locality; reshaping per discovered period into a 2D grid whose axes *are* those two directions turns both into 2D locality and unlocks shared multi-scale 2D convolution to model them at once; amplitude-softmax fusion weights the periods by their spectral confidence, residual stacking refines with depth, and per-window instance normalization restores each window's own scale so the residuals form a clean anomaly signal.

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
    frequency_list = abs(xf).mean(0).mean(-1)     # amplitude averaged over batch and channels
    frequency_list[0] = 0                         # drop DC (the mean, not a period)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
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
        super().__init__()
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

        dec_out = dec_out.mul(
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                                        # [B, seq_len, c_out] reconstruction
        return None
```
