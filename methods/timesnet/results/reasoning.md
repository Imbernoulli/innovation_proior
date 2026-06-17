Let me start from what actually hurts. I have streams of monitoring data — server metrics, spacecraft telemetry, satellite channels — and I have to flag the time points where something has gone wrong. The labels are useless for training because anomalies are rare and labeling at scale is hopeless, so this is unsupervised: I train on data that is overwhelmingly normal and I want to notice when the signal stops behaving normally. The classical way to make that operational is reconstruction — teach a model to reproduce normal windows faithfully, then at test time the points it *fails* to reproduce are the suspects, because a model that has only ever seen normal variation should choke on the abnormal. So the anomaly score is just the per-point reconstruction error, and the whole game collapses to one thing: how well can I model the temporal variation of a normal window? A better variation model gives a tighter reconstruction on normal points and a cleaner spike on anomalies. That is the real problem hiding under "anomaly detection."

And modeling temporal variation in real data is a mess. One isolated time point is a few scalars and tells me almost nothing; the signal lives in how the values move. A server's load climbs and falls through its daily cycle, but the daily shape is not the same from one day to the next, and there is slow drift on top. Rising, falling, fluctuation, trend, repeated cycles: several variations overlap in the same one-dimensional record, and a model reading that record step by step has to disentangle all of them at once with no help from the data layout.

So let me be precise about what kinds of dependency a single point actually has, because vague talk of "temporal variation" is not going to get me anywhere. Take a point now. It depends on its immediate neighbors — the step before, the step after — which is the short-term movement inside the current cycle. But it also depends on the same phase one cycle ago and one cycle ahead: the same hour yesterday and tomorrow, the same phase of adjacent periods. Those are two genuinely different dependencies. One is the local shape *within* a period; the other is how the corresponding phase *changes from period to period*. Let me name them so I can keep them straight: intraperiod variation (within a cycle) and interperiod variation (across cycles, same phase). The first is short-range, the second long-range. Real series have both, and worse, they have several periods at once — daily and weekly for a server, several rhythms in telemetry — so there are several intra/inter pairs superimposed.

Now here is the structural problem, and I want to feel exactly where it bites. My data is a 1D sequence indexed by t. Adjacency along that axis gives me intraperiod neighbors for free: t-1 and t+1 sit right next to t. But the interperiod neighbor — the same phase one period back — is at t-p where p is the period length, and in the 1D layout that point is p steps away, with the entire previous cycle crammed in between. A convolution along the time axis with a kernel of any sane width never sees t and t-p together. To reach across a full period I would need a kernel as wide as the period, and then it is averaging over everything in between rather than relating the two same-phase points. So the 1D representation can present one of the two variations cleanly (intra) and effectively hides the other (inter). That is the bottleneck. Whatever I build on top of the raw 1D sequence inherits this: it has to recover the interperiod relationship implicitly, fighting the layout.

Let me see how each tool I have on the shelf runs into this, because I want to know what is genuinely missing and not just reinvent one of them. RNNs — LSTM and its descendants — carry a state forward step by step, so the dependency on t-p has to survive p steps of state transition. With overlapping cycles and long periods that is exactly the long-range washout LSTMs are known for; and the sequential recurrence is slow. TCNs slide a 1D kernel along time; dilation can widen the receptive field, but the kernel still only combines points that are near *along the time axis*, so same-phase points a period apart are not in any one kernel's window — same wall as above, just convolutional. MLP models like DLinear bake a single fixed linear map over the temporal dimension; that map is one flat function of absolute window position, shared across every series, with no explicit handle on which periodicities a given series carries, so it cannot pull the within-cycle and across-cycle parts apart or adapt its period structure series by series. Transformers relate pairs of points with attention; on this very task the plain attention encoder is actually the *worst* reconstruction backbone, and the reason is instructive — attention scores similarity between every pair of time points, but a window is mostly normal points, so the similarity is dominated by them and the rare abnormal pattern gets washed out, which is precisely the thing reconstruction-for-anomaly needs to keep sharp. Autoformer is the interesting one because it already takes periodicity seriously — its Auto-Correlation estimates likely period lengths from the autocorrelation function (which it computes through the FFT), rolls the series by those lags, and aggregates the rolled copies weighted by a softmax over their correlation confidences. That is a real acknowledgment that periodicity is the structure to exploit, and the confidence-weighted softmax aggregation is a clean idea I will want to remember. But Auto-Correlation still operates on the 1D series — it relates a point to its lagged copy by a roll-and-correlate, which captures the interperiod link but folds the intraperiod shape into the same 1D mechanism rather than treating the two as separate axes I can convolve over independently. None of these separates intra and inter into two distinct, simultaneously-modelable structures.

So the question sharpens: I have two kinds of locality — within a period and across periods — and a data layout (1D) that can only express one of them as locality. What if I just change the layout so that *both* become locality? Stare at the 1D sequence and the period p. If I chop the sequence into consecutive blocks of length p and stack those blocks as the rows of a matrix, then walking along a row is walking through one cycle (intraperiod, the within-period shape), and walking down a column is walking through the same phase across successive cycles (interperiod). Both dependencies are now *adjacencies* — one along columns within a period, one along rows across periods — in a 2D array. The thing that was p apart in 1D is now one step apart along the cross-period axis. That is the move. Reshape the 1D series, for a given period p, into a 2D tensor whose two axes are exactly intraperiod and interperiod, and the two variations I care about both become local in 2D.

And the instant the data is a 2D grid with two meaningful local axes, I get an enormous gift for free: the entire mature toolbox of 2D convolution. A 2D kernel sliding over this grid sees, in one receptive field, a few adjacent time points within the cycle *and* the same window of phases in a couple of neighboring cycles. It models intra and inter variation simultaneously, which is precisely what the 1D layout could never do with one kernel. So the bottleneck was never that the variation is hard — it is that the 1D layout was the wrong space to look at it in. Lift it to 2D and ordinary vision backbones — ResNet, Inception, ConvNeXt — can chew on temporal variation the way they chew on images. That connection is worth a lot on its own: it means time-series modeling can ride on the whole development of computer vision.

But which p? Real series have several periods at once, and I do not know them a priori — they differ by dataset and even by window. I need to discover the dominant periods from the data. This is where the spectrum comes in. The amplitude of the Fourier transform at frequency j measures how strongly a periodic basis function of period T/j is present in the window; a strong period shows up as a tall amplitude peak. So compute the FFT of the window, take amplitudes, and the tallest peaks tell me the dominant frequencies, hence the dominant periods p = T/f. Concretely, for a window of length T with C channels, take the FFT along time, its amplitude, and average the amplitude over the C channels so I get one length-T amplitude profile A shared across channels — I want the *same* reshape applied to every channel so they stay aligned in the 2D grid, so a single set of periods for the whole window is exactly right. This matters more here than in a univariate setting: the monitoring streams have many channels (25, 55) and an anomaly may show in one channel while the periods are best estimated by pooling the spectral energy across all of them.

A couple of details I have to get right or this misbehaves. First, the DC term, frequency 0: its amplitude is just the window's mean energy, it corresponds to "period T/0" which is meaningless (an infinite, non-repeating constant), and it is usually huge — if I let it through it dominates the peak selection and tells me nothing about periodicity. So zero it out before picking peaks. Second, the spectrum of a real signal is conjugate-symmetric, so the second half of the frequencies is a mirror of the first; I only consider frequencies up to T/2, which the real FFT gives me directly. Third, I do not want all the peaks — the spectrum is sparse and the high-frequency tail is mostly noise (a long-known fact in spectral analysis, and the same denoising motive behind keeping only a few frequency components in frequency-domain forecasters). So take only the top-k amplitudes, getting the k most significant frequencies f_1..f_k and their periods p_i = T // f_i — keep the k amplitudes A_{f_1}..A_{f_k} around too; I have a hunch they will be useful as importance weights, exactly the way Auto-Correlation used its correlation confidences. So one function, call it Period, maps the window to its k dominant periods and their amplitudes.

Now the reshape itself, carefully, because the dimensions have to come out exactly. For period p_i and its frequency f_i, I want to lay the T time steps into a grid with f_i rows and p_i columns — roughly f_i cycles each of length p_i. But T is generally not equal to f_i × p_i (integer division guarantees a mismatch). So pad the series with zeros along the time axis up to the next multiple of p_i, reshape the padded length into (length // p_i) × p_i, and after I am done processing, truncate back to the original T. The padding is harmless filler that the truncation discards. So I get, for each of the k periods, a 2D tensor of shape (number-of-periods) × (period-length) per channel — k different 2D views of the same window, each exposing a different periodicity's intra/inter structure.

What do I run over each 2D tensor? A 2D conv, obviously, since that is the whole point of going to 2D. But what kernel size? Here is the subtlety: the reshape fixes *one* period, but the actual variation within and between cycles has structure at several scales — a fine wiggle over two or three time steps, a broader hump over a third of the cycle, a slow tilt across several cycles. A single fixed kernel size commits to one of those scales. I want several at once. That is exactly the problem the Inception block was invented for in vision: run several kernel sizes in parallel inside one block and combine them, so the block is multi-scale by construction while staying parameter-efficient compared to one giant kernel. So I will use a parameter-efficient inception block — a set of 2D convolutions with increasing kernel sizes (1×1, 3×3, 5×5, …, each padded to preserve the spatial size so the outputs can be combined), run in parallel and averaged together. Concretely a few convolutions with kernel size 2i+1 and padding i for i = 0..num_kernels-1, stacked and mean-pooled across the kernel dimension. And I will make it a small two-layer affair with a nonlinearity: inception expanding d_model → d_ff, a GELU, then inception contracting d_ff → d_model, a little channel-bottleneck feedforward in 2D, so it has the capacity to actually transform the 2D features rather than just smear them.

One more decision that matters for efficiency and for keeping the design clean: I have k reshaped tensors, one per period. Do I give each its own inception weights? If I did, the model size would grow with k, and k is a hyperparameter I would like to tune freely without changing the parameter count. So share one inception block across all k periods. Each period gets a different *reshape* (different grid geometry), but the same conv weights process all of them. Model size is then invariant to k — I can dial k up or down purely as a width-of-search knob. That is the parameter-efficient part, and it is also conceptually right: the inception is learning "how to read 2D temporal variation," which should not depend on which period produced the grid.

After the shared inception transforms each 2D tensor, I reshape each back to 1D (flatten the grid back along time) and truncate to T, giving me k candidate 1D representations of the window — one per period. Now I have to fuse them into a single representation for the next stage. A plain sum throws away the fact that some periods are far more present in this window than others. I already kept the amplitudes A_{f_i}, and an amplitude *is* a measure of how strongly that period is expressed — exactly the relative-importance signal I want. So weight the k representations by their amplitudes. To turn raw amplitudes into proper convex weights I push them through a softmax over the k periods, then take the weighted sum: sum_i softmax(A)_{f_i} · (1D representation from period i). This is the same confidence-weighted aggregation idea Auto-Correlation used, repurposed: the FFT amplitude is the confidence of the period. A window dominated by a daily cycle puts most of the weight on that period's representation; a window with two strong rhythms splits the weight. I can imagine the alternatives — sum the representations directly, or weight by raw amplitudes without the softmax — and both feel worse: the direct sum ignores importance entirely, and raw amplitudes are unnormalized and scale-sensitive, so the softmax's normalization to convex weights is the principled choice; a quick mental ablation says directly summing should cost a bit of F1 and dropping the softmax a little less, and that is roughly what I would expect to measure.

So one block does: discover periods → reshape to k 2D tensors → shared inception on each → reshape back → amplitude-softmax aggregate. Call it a TimesBlock. Now, this transforms the feature sequence, but I want to stack several of them so the representation is refined with depth, and naively stacking transformations is a recipe for unstable training. The standard fix is residual: the block computes TimesBlock(X) and I add the input back, X^l = TimesBlock(X^{l-1}) + X^{l-1}, with a layer norm to keep activations well-scaled. The residual also means a block only has to learn the *correction* to the variation representation, which composes nicely across layers — early blocks catch coarse structure, later ones refine it. That is the heart of the architecture: an embedding to lift the raw inputs into a d_model feature sequence, a stack of residual TimesBlocks, and a projection back out.

Now I have to make this actually produce the thing the task needs, and here the anomaly-detection setting is simpler than a forecaster in one clean way. I am not predicting any future steps; I am *reconstructing the window I was given*. The TimesBlock maps a length-T feature sequence to a length-T feature sequence — it transforms what is there, same length in, same length out — and that is exactly a reconstruction map. So there is no horizon to invent, no learned temporal extension, no decoder bolted on the end: the look-back length and the output length are the same seq_len, and the pad/reshape inside the TimesBlock uses that length with no extra future positions. A forecaster would have to first stretch the sequence to seq_len + horizon so the cycles continue into the future; here that whole limb just is not present. I embed the window, run the residual TimesBlocks over its own length, and project every position back to the input channels. The reconstruction is the projection of the refined representation, and the framework will read the anomaly score off the squared difference between this reconstruction and the input, point by point.

Before that, one real problem I have not addressed: distribution shift. The channels and datasets span very different magnitudes and the level wanders over time; if I feed raw values into a shared backbone, it burns capacity learning to absorb each window's offset and scale instead of modeling variation. The known remedy is per-window instance normalization — subtract the window's temporal mean, divide by its standard deviation, run the model on the normalized series, then undo it at the output by multiplying the std and adding the mean back. So compute the mean of the window along time, subtract it; compute the variance along time (biased, dividing by the count, since this is a normalization statistic not an estimate I will do inference on) and divide by its square root with a small epsilon under the root for numerical safety; detach both statistics from the graph because they are normalization constants, not parameters to backprop through. After the backbone and projection, de-normalize the output with the same mean and std, broadcasting them over the window length. This is what lets one shared backbone fit windows spanning orders of magnitude. There is a subtlety worth being honest about for the anomaly task: normalizing per window means an anomaly that is purely a level shift could be partly absorbed by the subtraction — but the data is already globally Z-scored per dataset, the windows are short, and the anomalies I care about are violations of the *variation* (the cycle breaks, the shape goes wrong), which survive the per-window centering precisely because they are deviations from the periodic structure the model has learned to reconstruct.

Let me also pin down the embedding, since the backbone needs a d_model-dimensional feature sequence. For anomaly detection there are no useful calendar marks to feed, so the embedding is the value embedding (a 1D conv over the channel axis lifting the raw channels to d_model) plus a fixed sinusoidal positional embedding so the block has absolute-position information, summed with dropout — the temporal-feature branch is simply not used here (I pass no time marks). None of that is the contribution; it is the generic harness the variation block sits inside.

Let me now write the whole thing as the code I would actually ship, filling in that one empty variation-block slot with the TimesBlock I just derived and wiring the reconstruction path with the instance normalization. The period discovery first:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1   # parallel multi-scale 2D convs, mean-combined


def FFT_for_Period(x, k=2):
    # x: [B, T, C]; find the k dominant periods by spectral amplitude
    xf = torch.fft.rfft(x, dim=1)                 # real FFT along time -> freqs in {0..T/2}
    frequency_list = abs(xf).mean(0).mean(-1)     # amplitude averaged over batch and channels -> [T/2+1]
    frequency_list[0] = 0                         # drop DC: it is the mean, not a period
    _, top_list = torch.topk(frequency_list, k)   # k most significant frequencies
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list               # period length p_i = T // f_i for each
    return period, abs(xf).mean(-1)[:, top_list]  # periods, and per-(batch) amplitudes for weighting
```

Then the TimesBlock — reshape to 2D, shared inception, reshape back, amplitude-softmax aggregate, residual; for reconstruction the length is just seq_len (pred_len is 0), so the same length goes in and comes out:

```python
class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len          # 0 for anomaly detection (reconstruction)
        self.k = configs.top_k
        # ONE shared inception, reused for every discovered period -> model size independent of k.
        # Two layers with a GELU bottleneck: d_model -> d_ff -> d_model, multi-scale 2D kernels each.
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()                         # T = seq_len + pred_len = seq_len here
        period_list, period_weight = FFT_for_Period(x, self.k)   # k periods + their amplitudes

        res = []
        for i in range(self.k):
            period = period_list[i]
            # pad along time so the length is divisible by this period, then reshape to 2D
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros([B, length - (self.seq_len + self.pred_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # [B, length, N] -> [B, num_periods, period, N] -> [B, N(=channels), num_periods, period]
            # columns (period) = intraperiod axis, rows (num_periods) = interperiod axis
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # multi-scale 2D conv reads intra- and interperiod variation together
            out = self.conv(out)
            # back to 1D: [B, N, num_periods, period] -> [B, length, N], then truncate the padding
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)                            # [B, T, N, k]

        # adaptive aggregation: amplitude = confidence of each period -> softmax -> convex weights
        period_weight = F.softmax(period_weight, dim=1)          # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)                  # weighted sum over the k periods
        return res + x                                            # residual connection
```

And the full reconstruction model for anomaly detection, with the instance normalization wrapping the stack of TimesBlocks — note there is no temporal extension and no decoder, the window comes in and the reconstruction of that same window comes out:

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len           # 0 for anomaly detection
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # reconstruction: project the representation straight back to the input channels
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # per-window instance normalization (subtract mean, divide by std) over the window
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embed (no time marks for anomaly detection -> value + positional only)
        enc_out = self.enc_embedding(x_enc, None)                  # [B, seq_len, d_model]
        # residual TimesBlocks + layer norm, over the window's own length
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)                         # [B, seq_len, c_out]

        # de-normalize with the same window statistics, broadcast over the length
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out                                         # [B, seq_len, c_out] reconstruction
        return None
```

Trained with Adam at 1e-4 and an MSE reconstruction objective over normal windows, with k=3, a few residual layers, six inception kernels, seq_len 100, and a model width sized to the channel count by min(max(2^ceil(log C), d_min), d_max). At test time the framework takes the per-point squared reconstruction error as the anomaly score, sets a threshold at the anomaly-ratio percentile of the score distribution, and scores point-adjusted F1 — but all of that lives in the harness; my job was only to make the reconstruction good, and the reconstruction is good exactly when the variation model is.

So the causal chain is: the anomaly score is reconstruction error, so the task reduces to modeling the temporal variation of normal windows; real variation has two distinct components — short-range within a cycle and long-range across cycles at the same phase — and the 1D layout can only present one of them as locality, which is why RNNs, TCNs, MLPs, and even point-wise attention all struggle to model both at once (and why plain attention, distracted by the dominant normal points, reconstructs worst of all). Reshaping the 1D series, for a discovered period, into a 2D grid whose two axes *are* the intraperiod and interperiod directions turns both dependencies into 2D locality, which unlocks the whole 2D convolution toolbox to model them simultaneously. The periods are discovered by FFT amplitude (drop DC, top-k, denoise the spectrum); each period gives a 2D view; a single shared multi-scale inception block reads each view (parameter cost independent of k); the views are fused by a softmax over their amplitudes, treating amplitude as the period's confidence; residual stacking refines the representation with depth; and because the task is reconstruction, the window comes in and out at the same length with no horizon to invent — only a per-window instance normalization restoring each window's own scale, which is what makes one shared backbone reconstruct heterogeneous multichannel monitoring streams and turn their residuals into a clean anomaly signal.
