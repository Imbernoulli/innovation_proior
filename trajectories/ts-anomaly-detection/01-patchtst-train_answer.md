The harness hands me a window $x_{\text{enc}}$ of shape $[B, 100, C]$, trains me with MSE to reproduce normal windows, and reads the anomaly score off the per-point squared error between my reconstruction and the input. The scaffold default is identity — return the window unchanged — which is degenerate: the reconstruction error is zero everywhere, the score is flat, and the threshold has nothing to bite on. The first real choice is which reconstruction backbone to start from, and the one limitation that decides it is the *token*. The point-wise Transformer reconstructors on the shelf treat a single time step as a token, and a single sensor reading at one instant means nothing in isolation — the information in these streams lives in shapes over short stretches: a rising edge, a dip, a local oscillation, the slope of a ramp. Worse, a reconstruction window is almost entirely normal points, so pairwise step-attention is dominated by them and the rare abnormal pattern — exactly what the score must keep crisp — gets averaged into the normal mass. That is why on this very task the plain attention encoder is the worst backbone available: the attention map is being computed over the wrong objects.

I propose **PatchTST** as the first rung: a channel-independent, patch-tokenized vanilla Transformer that reconstructs the input window. The central move is to fix the token. Instead of one step per token, I group a contiguous stretch of $P$ steps into a single token — a local shape that attention can actually compare. Concretely, pick a patch length $P$ and a stride $S$ (the hop between consecutive patch starts), slide a width-$P$ window along the series in steps of $S$, and each placement is one patch in $\mathbb{R}^P$. The number of patches is the count of how many length-$P$, stride-$S$ windows fit in length $L$. I have to cover the window's tail for reconstruction — every point needs a reconstruction or the score is undefined there — so before patching I pad the end by repeating the last value $S$ times, which guarantees exactly one extra full window slides into existence over the true end. The count becomes $N = \lfloor (L-P)/S \rfloor + 2$, and with $L=100$, $P=16$, $S=8$ this is $\lfloor 84/8 \rfloor + 2 = 12$ patches. The payoff is both representational and quantitative: the attention map is $N\times N$, so going from $N=L=100$ to $N\approx 12$ drops the cost by roughly $S^2 \approx 64\times$, and more importantly the tokens are now local shapes. I keep $S$ a bit smaller than $P$ so consecutive patches overlap by half and no edge shape gets cleanly split.

The second decision is independent of patching: how to handle the $C$ channels. The standard multivariate Transformer mixes them — projects the whole $C$-vector at each step into one token — forcing every channel under one shared attention pattern. I do the opposite and run one backbone independently on each univariate series, for three reasons. *Adaptability*: under one shared map a single attention pattern is a compromise across channels with completely different temporal behavior; channel-independence lets each channel form its own attention map. *Data efficiency*: learning genuine cross-channel correlation jointly with temporal structure is a much larger hypothesis space, and these datasets are not huge, so a channel-along-time-only model fits with less data. *Overfitting*: a channel-mixing model can fit spurious cross-channel coincidences that hold in the normal training window and do not generalize — poison for a reconstruction backbone whose job is to model *only* normal structure cleanly. But channel-independent does not mean $C$ separate models; that would $C\times$ the parameters and share nothing. The right design is *one* backbone with one set of weights, run independently per series: channel-independent in the forward pass, weight-shared in the parameters. The implementation cost is a reshape — permute $[B,L,C]\to[B,C,L]$, patch to $[B,C,N,P]$, fold channels into the batch axis to get $[B\cdot C, N, P]$, and the encoder simply sees a batch of $B\cdot C$ independent length-$N$ token sequences. As a structural gift, the weights never learn how many channels there are, so the same backbone serves PSM's 25, MSL's 55, and SMAP's 25 unchanged.

The encoder itself I keep deliberately *vanilla*, because the whole thesis is that the input representation, not the attention kernel, was the problem — I must not sneak in a fancy kernel and confound the test. Each patch is linearly projected to width $D$ with no bias (an instance-normalized patch already has its level removed, so a per-patch additive offset buys nothing), a learnable positional embedding is added (patches are otherwise an unordered set, while temporal order is everything), then standard multi-head scaled dot-product attention runs over the $N$ patch tokens, followed by a position-wise feed-forward $D\to d_{\text{ff}}\to D$, with residual connections around both sublayers. The one non-default choice inside is the normalization: the usual Transformer uses LayerNorm, which normalizes within each token — risky here, because a spike landing in a patch skews that token's own statistics and drags LayerNorm around. I use **BatchNorm** instead, which normalizes each feature across the batch of patch positions, so a single outlier patch is diluted by all the others rather than corrupting its own normalization (transpose the feature axis into place, BatchNorm1d, transpose back).

The head is where the anomaly setting is cleaner than forecasting: I am not predicting a future horizon, I am reconstructing the window I was given, so the target window *is* $\text{seq\_len}$. Flatten the per-series $[D, N]$ representation into a $D\cdot N$ vector and linear-project it to $\text{seq\_len}$, then permute back to $[B, \text{seq\_len}, C]$. The head input width is $\text{head\_nf} = D\cdot N = D\cdot(\lfloor (L-P)/S\rfloor + 2)$. No decoder, no learned temporal extension — the window comes in and the reconstruction of that same window comes out.

One last thing the data forces: distribution shift. Channels and datasets span very different magnitudes and the level wanders, so feeding raw values would burn capacity absorbing each window's offset and scale instead of modeling shape. I wrap the backbone in reversible per-instance normalization: subtract the window's temporal mean, divide by its standard deviation (the biased variance plus $1\mathrm{e}{-5}$ for a flat series), detach both as constants, run the model on the normalized window, and at the output multiply the std back and add the mean back, broadcast over the window length. There is a subtlety — per-window centering can partly absorb a pure level-shift anomaly — but the data is already globally Z-scored per dataset, the windows are short, and the anomalies that matter are violations of the *shape* (the cycle breaks, the local dynamics go wrong), which survive per-window centering precisely because they are deviations from the structure the model has learned to reconstruct.

```python
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)        # (D, N) -> (D*N)
        self.linear = nn.Linear(nf, target_window)      # (D*N) -> seq_len
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                               # x: [bs, nvars, D, N]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride                                # pad S copies of the last value (one extra patch)

        # patch + linear-project each patch to d_model (no bias) + positional embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over the N patch tokens; BatchNorm (not LayerNorm)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2),
                                     nn.BatchNorm1d(configs.d_model),
                                     Transpose(1, 2))
        )

        # head input width = d_model * number of patches N; N = (L - P)/S + 2
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        # reconstruction: project the encoded patches back to the seq_len window
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                head_dropout=configs.dropout)

    def anomaly_detection(self, x_enc):
        # reversible per-instance normalization against distribution shift
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # channel-independence: channels on dim 1, folded into the batch inside patch_embedding
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, N, D]

        # vanilla encoder over patch tokens
        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, N, D]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [bs, nvars, N, D]
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, D, N]

        # reconstruction head -> seq_len, then back to [bs, seq_len, nvars]
        dec_out = self.head(enc_out)                    # [bs, nvars, seq_len]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, seq_len, nvars]

        # de-normalize with the same window statistics
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        return None
```
