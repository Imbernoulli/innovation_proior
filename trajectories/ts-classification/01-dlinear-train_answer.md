The scaffold I inherit returns zero logits, so it predicts a constant for every window and scores at chance, modulated by class imbalance. That floor exists only to anchor a question: how much accuracy is available from the *cheapest honest model* that actually reads the window? I want the weakest rung to be a model I cannot accuse of cleverness, so that whatever it scores is a clean lower bound on "a real encoder helps." And the doubt that motivates the whole ladder points at exactly which cheap model to use. The standard deep recipe for multivariate time-series classification embeds the whole channel-vector at each timestamp into one token and runs self-attention over the `seq_len` tokens — but a single timestamp of a spectral trace or an MEG sweep has no standalone meaning the way a word does, so point-wise attention scores similarity between objects that individually carry no semantics, at quadratic cost in `seq_len`. What nags is that a plain linear map over the flattened window is reported competitive with these encoders. If a linear map is competitive, then either attention is the wrong tool or the leverage is in the *input representation*, not the kernel — and the right floor is precisely that linear map, both because it is the honest lower bound and because it operationalizes the doubt.

I propose a **decomposition-linear classifier** (DLinear adapted to classification). The bare floor would be: flatten the window into one $\text{enc\_in}\cdot\text{seq\_len}$ vector and apply a single linear layer to `num_class` — logistic regression on the raw window, one weight per (channel, timestep, class), no temporal notion beyond what the weights memorize, no nonlinearity. I do not ship the bare version, because there is a free, parameter-light refinement that costs nothing in capacity and that I expect to matter on at least one dataset: seasonal-trend decomposition, the oldest move in time-series analysis. Write the series additively as a slow trend plus a seasonal residual, because each piece on its own is more regular than the sum, and in a classifier the discriminative cue can live in either piece. EthanolConcentration is the clear case where the slow trend — the level and slow shape of the smooth absorbance curve — carries the label; Handwriting's accelerometer gestures, by contrast, live in the oscillatory residual once the slow drift of the hand is removed. A single flatten-and-project must fit both kinds of cue with one weight matrix, and the large-magnitude trend dominates the fit. Splitting the window into $\text{trend}=\mathrm{MovingAvg}(x)$ and $\text{seasonal}=x-\text{trend}$, mapping each with its own linear map and summing, lets each stream specialize.

What makes this safe as a *floor* is that it adds no representational capacity. A moving average is linear; two linear maps plus a sum is still affine end to end. The decomposition is preconditioning, not depth — it separates the loud component from the quiet one so the optimizer can fit each, which is exactly the move that distinguishes this rung from the bare linear map without smuggling in any nonlinearity. The moving average must preserve length so I can subtract it from the input, so I use an `AvgPool1d` with an odd kernel $k=25$ (the canonical smoothing scale), stride 1, and replicate-pad the two endpoints by $(k-1)/2$ each. Replication is deliberate: zero-padding would drag the trend toward zero at the window edges, inventing spurious dips exactly where I have least information, whereas replicating the endpoint keeps the trend flat-but-faithful there. With $k$ odd and $(k-1)/2$ on each side the pooled output has exactly the input length, and the seasonal part is $x-\text{trend}$.

The head is where the classification path departs from how a *forecaster* would use the same decomposition. A forecaster maps the time axis $L\to T$ and keeps the channel axis separate, because it must emit a future trajectory per channel. I emit class logits, not a trajectory. So each of the two streams runs a `Linear(seq_len, seq_len)` along time (classification keeps $\text{pred\_len}=\text{seq\_len}$, so the per-channel temporal map is square), the two are summed back to a $[\text{B},\text{seq\_len},\text{enc\_in}]$ representation, and then the head flattens *everything* — both time and channel axes — into one $\text{enc\_in}\cdot\text{seq\_len}$ vector and projects straight to `num_class` with a single `Linear`. That final projection is the only place the channels are mixed and the class boundary is drawn; the per-time linear maps before it are the decomposed temporal encoder. The whole thing is affine from input to logits.

I am honest about what this rung throws away, because the throwaways are exactly what the later rungs will exploit. It ignores the padding mask `x_mark_enc` entirely — the flattened projection sees padded positions as ordinary zeros and learns weights for them, which is wasted capacity and, where padding length correlates with class, a spurious cue. It models cross-channel interaction only through the final flatten-and-project, a single static linear combination, with no mechanism to learn that two MEG channels covary; FaceDetection, whose entire signal is cross-channel covariance over ~144 noisy channels, is where I expect this to hurt most. And it is linear, so any class boundary that is not a hyperplane in the flattened-window space is unreachable; Handwriting's 26 oscillatory gesture classes are where I expect the linear boundary to be too blunt. So the falsifiable profile of this floor is not a single number: clearly above chance everywhere (so it is genuinely reading the window), relatively strongest on the forgiving binary FaceDetection where a linear combination of 144 channels has plenty of features, and weakest on 26-class Handwriting where blunt linearity and the ignored mask both bite.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """DLinear classification fill: decomposition + two linear maps, flatten-and-project head."""

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # classification keeps pred_len == seq_len: the per-channel temporal map is square
        self.pred_len = configs.seq_len
        # series decomposition block (moving-average trend + seasonal residual), kernel = 25
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # classification head: flatten the (time x channel) representation, project to classes
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, seq_len, enc_in]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)            # [B, enc_in, seq_len]
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, enc_in, seq_len]
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output                         # recombine the streams
        return x.permute(0, 2, 1)                                  # [B, seq_len, enc_in]

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc)                              # [B, seq_len, enc_in]
        output = enc_out.reshape(enc_out.shape[0], -1)            # flatten time x channel
        output = self.projection(output)                          # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
