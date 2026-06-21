PatchTST confirmed its diagnosis, including by failing exactly where I flagged. Handwriting moved 0.2306 → 0.2541 — local-shape tokens and a nonlinear encoder do help the 26-class gesture problem, so local temporal shape was indeed a bottleneck. FaceDetection went 0.6822 → 0.6853, parity within noise, because channel-independence forbids in-encoder cross-channel modeling on the one dataset that is all cross-channel covariance. And EthanolConcentration slipped 0.2890 → 0.2852, a hair below the floor, because patching with overlapping windows plus per-instance normalization washes out the slow global trend the decomposition captured directly. So PatchTST is a sideways trade — a win where local shape matters, flat where cross-channel matters, a small loss where the global trend matters — and averaged it is about even with the floor. The common structural defect across both rungs is now visible: both present time as a *single* axis and recover everything from adjacency (the floor) or attention over local patches (PatchTST) along it. But a patch never relates the same phase one period earlier to the same phase one period later — those points are `period` steps apart in the 1-D layout, with a whole cycle crammed between, and no single patch or kernel spans them. PatchTST reads within-stretch shape but not *across*-period shape. And neither rung consults the padding mask.

I propose **TimesNet**: reshape the 1-D series into 2-D so that cross-period structure becomes ordinary locality, then convolve. Stare at a series with period $p$. The within-period neighbour ($t-1$, $t+1$) is adjacent, but the cross-period neighbour ($t-p$, the same phase one cycle back) is $p$ steps away. Chop the series into consecutive blocks of length $p$ and stack them as the rows of a matrix: walking along a row is walking through one cycle (the intraperiod, within-cycle shape), and walking down a column is walking through the same phase across successive cycles (the interperiod, cross-cycle variation). Both dependencies are now adjacencies in a 2-D array — the thing that was $p$ apart in 1-D is one step apart along the column axis. The instant the data is a grid with two meaningful local axes I inherit the entire mature toolbox of 2-D convolution: a 2-D kernel sees, in one receptive field, a few adjacent timesteps within the cycle *and* the same window of phases in neighbouring cycles, modelling intra- and inter-period variation simultaneously, which neither earlier rung could do with one operator.

Which period, though? These series have several periodicities at once and I do not know them a priori, so I discover them from the data. The amplitude of the Fourier transform at frequency $j$ measures how strongly a periodic component of period $T/j$ is present, so I take the real FFT of the window along time, take amplitudes, average over batch and channels to one length-$(T/2+1)$ profile, zero the DC term (frequency 0 is the window mean, not a period, and it is usually huge enough to dominate peak selection), and pick the top-$k$ peaks ($k=3$, because the spectrum is sparse and its high-frequency tail is mostly noise). Each peak gives a dominant frequency $f_i$ and a period $p_i=T // f_i$; I keep the $k$ amplitudes too, because an amplitude measures how strongly that period is expressed and I will use it as an importance weight. For each $p_i$ I lay the $T$ timesteps into a $(T // p_i)\times p_i$ grid — $T$ is generally not a multiple of $p_i$, so I zero-pad along time to the next multiple, reshape to $[\text{B},\,\text{length}/p_i,\,p_i,\,N]$, permute to read as an image $[\text{B},N,\text{num\_periods},\text{period}]$, and after processing reshape back and truncate the padding.

What runs over each 2-D view is a multi-scale convolution. The variation has structure at several spatial scales — a fine wiggle over two steps, a broad hump over a third of the cycle, a slow tilt across several cycles — so a single fixed kernel commits to one scale; that is exactly the problem the Inception block solves in vision by running several kernel sizes in parallel and combining them, multi-scale by construction and parameter-efficient. So the conv is a parameter-efficient Inception block (parallel 2-D convs of increasing kernel size, $\text{num\_kernels}=6$, mean-combined) used as a two-layer bottleneck $d_\text{model}\to d_\text{ff}\to d_\text{model}$ with a GELU between. One decision that matters: I share *one* Inception block across all $k$ periods. Each period gets a different reshape (different grid geometry) but the same conv weights, so the model size is invariant to $k$ — I can dial $k$ purely as a width-of-search knob — and the Inception is conceptually learning "how to read 2-D temporal variation," which should not depend on which period produced the grid. After the shared Inception transforms each of the $k$ views, I reshape each back to 1-D, truncate to $T$, and fuse. A plain sum would ignore that some periods are far more present in this window than others; instead I push the $k$ amplitudes through a softmax (turning raw, scale-sensitive amplitudes into convex weights) and take the amplitude-weighted sum, so a window dominated by one rhythm puts most weight on that period's view. That is one TimesBlock: discover periods → reshape to $k$ 2-D views → shared Inception on each → reshape back → amplitude-softmax aggregate, with a residual connection so each block learns only a correction and the stack stays stable. I stack $e_\text{layers}=3$ residual TimesBlocks with a LayerNorm between them.

In front of the stack, an embedding lifts the raw `enc_in` channels into a $d_\text{model}=128$ feature sequence: a value embedding (a 1-D conv over the channel axis) plus a fixed positional embedding, dropped out, with no time marks (the windows are bare series). One important contrast with the two earlier rungs: TimesNet does *not* per-instance subtract/divide normalize for classification. The discriminative cue can live in the absolute level and scale (a spectral curve's overall absorbance, a gesture's acceleration magnitude), and the embedding's conv plus the in-block LayerNorm already handle scale heterogeneity without erasing the level — whereas PatchTST's per-window normalization, which I argued washed out EthanolConcentration's global trend, is exactly the move I want to *not* make here. That alone is a concrete reason to expect EthanolConcentration to recover. And the head finally does the thing both earlier rungs skipped. After the stack I have a per-timestep representation $[\text{B},\text{seq\_len},d_\text{model}]$; I apply a GELU and dropout, then — the load-bearing line — multiply by the mask, `output = output * x_mark_enc.unsqueeze(-1)`, zeroing the feature vectors at padded positions before flattening, so the padded tail contributes exactly zero to the flattened $\text{seq\_len}\cdot d_\text{model}$ vector that the final `Linear(d_model · seq_len, num_class)` projects to logits. The classifier can no longer learn spurious weights on padding or drift with where a window happens to end — a clean, free gain on the variable-length datasets. Channels are still mixed only in the value embedding and the head; what is new is the cross-period 2-D modelling and the mask-aware pooling. I expect the largest gain on Handwriting (phase-recurrent gestures, plus variable length the mask now respects), a recovery and clearing of 0.29 on EthanolConcentration, and at best parity on FaceDetection, where channels are still not mixed inside the encoder.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]; find the k dominant periods by spectral amplitude
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0                              # drop DC: window mean, not a period
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]       # periods, per-batch amplitudes


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len               # 0 for classification
        self.k = configs.top_k
        # ONE shared inception (model size independent of k): d_model -> d_ff -> d_model
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # pad along time so length divides the period, reshape to 2-D
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # [B, length, N] -> [B, num_periods, period, N] -> [B, N, num_periods, period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                       # multi-scale 2-D conv: intra + inter period
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # amplitude-softmax aggregation: amplitude = period confidence
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x                                 # residual connection


class Model(nn.Module):
    """TimesNet classification fill: 2-D period reshape + Inception, mask-aware pooling head."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)      # value + positional, no marks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)     # zero out padded positions before pooling
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)               # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
