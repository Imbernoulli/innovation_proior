The decomposition-linear floor landed where its nature predicted: EthanolConcentration 0.2890 (four classes, barely above the ~0.25 chance), FaceDetection 0.6822 (binary, its relative best, a linear combination of ~144 channels being plenty for a forgiving two-class problem), and Handwriting 0.2306 (26 classes, well above the ~0.04 chance but the weakest of the three). Reading those numbers sharpens the diagnosis. The flatten-and-project head is linear, so the decision boundary is a hyperplane and the 26-class gesture geometry is unreachable; and the model never forms a *local temporal feature* — it has no notion that a rising edge over a few steps is a thing, only per-(channel, timestep) weights. Both are representation problems, not optimization problems, and both point at the same fix: stop weighting raw timesteps, and start from local sub-sequences. This is the positive half of the doubt that motivated starting at the floor — the floor confirmed a per-timestep linear map is weak; now I act on what the token *should* be.

I propose **PatchTST**: patching plus a channel-independent vanilla Transformer. In vision, a pixel is meaningless and the answer was to cut the image into patches, each a local visual concept; the analogue is immediate. Take one channel's series of length $L=\text{seq\_len}$, pick a patch length $P=16$ and stride $S=8$, and slide a width-$P$ window along the series in steps of $S$; each placement is one patch, a vector in $\mathbb{R}^P$. Before patching I replication-pad the end by $S$ so one more full window slides into existence and reaches the last timestep, giving $N=\lfloor (L-P)/S\rfloor + 2$ patches. Two payoffs fall out. Each token now carries a local shape — a ramp, a bump, an oscillation — so attention is finally comparing meaningful objects ("does this ramp resemble that ramp $k$ patches later?") instead of individual scalars, exactly what the linear floor could not form. And the token count drops from $L$ to about $L/S$, so the $N\times N$ attention map shrinks by roughly $S^2$ — with $S=8$ a ~64× cut in attention cost, which is what makes three layers of 16-head attention affordable. Stride below patch length ($8<16$) means consecutive patches overlap by half, so no local shape is cleanly split down the middle between two patches.

The second decision the floor forces is the channels, and here I go *against* the instinct to "model cross-channel correlation with attention." The linear evidence cuts the other way: the channel-*independent* linear map is what is competitive with mixing Transformers on these benchmarks, and mixing models overfit on the smaller datasets. So I make it **channel-independent** — one shared Transformer backbone run over each channel's patch sequence separately, weights shared across channels, no cross-channel attention inside the encoder at all. Three reasons, each traceable. *Adaptability*: a single mixed token stream would force one attention pattern on all channels, but a spectral channel, an MEG channel, and an accelerometer axis have completely different temporal behavior, so each should form its own attention map. *Data efficiency*: learning cross-channel interaction jointly with temporal structure is a far larger hypothesis space, and these UEA sets are small, so the temporal-only space converges with the data I have. *Overfitting*: mixing can fit spurious cross-channel coincidences in the training split. The cost in code is nearly nothing — patch the batch $[\text{B},\text{enc\_in},L]$ into $[\text{B},\text{enc\_in},N,P]$, fold the channel axis into the batch to get $[\text{B}\cdot\text{enc\_in},N,P]$, and the Transformer just sees a larger batch of length-$N$ sequences; reshape back at the end.

The backbone is a vanilla Transformer encoder, deliberately, since the floor told me the leverage is in the representation, not the kernel. Each $P$-vector patch is projected to $d_\text{model}=128$ with a *bias-free* linear embedding (a normalized patch has its level removed, so a per-patch additive offset buys nothing), a positional embedding is added because attention is permutation-invariant and patch order is everything, then $e_\text{layers}=3$ layers of multi-head self-attention ($n_\text{heads}=16$) over the $N$ tokens, each with a position-wise feed-forward $d_\text{model}\to d_\text{ff}=256\to d_\text{model}$ and residual connections. One non-default choice I make on purpose is the normalization: the default uses LayerNorm across features per token, but time series carry outliers — a sensor glitch, a regime jump — and an outlier patch would skew its own per-token statistics, so I use BatchNorm (transpose, `BatchNorm1d` over $d_\text{model}$, transpose back), which normalizes each feature across the batch of patch positions and dilutes a single outlier; this is measured to beat LayerNorm for time-series Transformers. And an input-side fix the floor lacked: per-instance normalization. The channels span very different magnitudes — MEG microvolts versus normalized absorbance versus acceleration — so before patching I subtract each window's temporal mean and divide by its temporal standard deviation (biased variance, detached, with a $10^{-5}$ floor under the root), decoupling shape-learning from level-tracking, which is what lets one shared backbone read heterogeneous channels. For classification there is no output trajectory to de-normalize, so this is purely input conditioning.

The head again departs from how a forecaster uses the same backbone. A forecaster flattens each channel's $[d_\text{model},N]$ and projects per channel to the horizon, keeping channels separate to emit a per-channel trajectory. I emit one class decision for the whole window, so after the encoder gives $[\text{B},\text{enc\_in},d_\text{model},N]$ (channels unfolded back out of the batch) I flatten across *both* the patch axis and the channel axis — $\text{head\_nf}=d_\text{model}\cdot N$ per channel, times `enc_in` channels — apply dropout, and project to `num_class` with one linear layer. That flatten across channels is the only place channels meet, the channel-independence story made literal: independent inside the encoder, joined once at the decision. I carry forward one known gap, like the floor: this path does not consult `x_mark_enc`, so patches over the right-padded tail are embedded and attended like any other and the flatten head learns weights for them — I leave it for the next rung, because folding in mask-aware pooling now would confound it with the patching change I am actually testing. So I expect the clearest gain on Handwriting (local-shape tokens and a nonlinear boundary are exactly its missing ingredients), at best parity on FaceDetection (channel-independence forbids the in-encoder cross-channel modeling its signal is made of), and a real risk of a small *loss* on EthanolConcentration, where per-window normalization can wash out the slow global trend the decomposition captured directly.

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
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    """PatchTST classification fill: patch + channel-independent Transformer, flatten head."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len            # 0 for classification
        padding = stride

        # patching + bias-free patch embedding + positional embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over the patch tokens, BatchNorm instead of LayerNorm
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # per-channel flattened width = d_model * number_of_patches
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        # classification head: flatten across patches AND channels, project to classes
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # per-instance normalization (subtract/divide by window mean/std)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # patch + embed; channels are folded into the batch -> channel-independent encoder
        x_enc = x_enc.permute(0, 2, 1)                          # [B, enc_in, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)           # [B*enc_in, N, d_model]
        enc_out, attns = self.encoder(enc_out)                  # [B*enc_in, N, d_model]
        # unfold channels back out: [B, enc_in, N, d_model] -> [B, enc_in, d_model, N]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # flatten across patches and channels, project to classes
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)                        # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
