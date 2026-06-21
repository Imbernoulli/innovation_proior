The provocation I cannot ignore is that a one-layer linear model — a per-channel linear map from the length-$L$ look-back window to the length-$T$ horizon, with a trend/seasonal split and no attention at all — matches or beats every published Transformer forecaster on the standard long-horizon benchmarks. That is a direct attack on the premise the whole efficient-Transformer line was built on. Informer, Autoformer, FEDformer, Pyraformer each performed a clever surgery on the attention kernel — ProbSparse query-key pruning, auto-correlation over period-shifted subseries, a Fourier-domain block, a pyramid of inter- and intra-scale edges — all in service of beating down the $O(L^2)$ cost of point-wise attention so that a long look-back becomes affordable. Yet not only does the linear model win, the complex Transformers do not even improve when handed a longer window: their error stays flat or rises as $L$ grows, while the linear model keeps improving. So the field's diagnosis — "attention is too expensive, make it cheaper" — is either wrong or beside the point. We need something that is genuinely competitive with the linear model on the same fixed protocol, that actually exploits a longer history instead of overfitting it, that survives the train/test distribution drift these non-stationary series exhibit, and that retains the one thing a linear map structurally cannot offer: a multi-layer representation that can be pre-trained on unlabelled series and transferred. The linear baseline is a strong but expressively shallow yardstick; the question is whether attention, applied correctly, can clear it.

My claim is that attention was never the bug — the token was. Look at what a token is in every one of these models: a single time step, either one scalar $x_t$ for a univariate series or the full $M$-channel vector for a multivariate one. In language a token is a word or sub-word, a unit with standalone meaning, so attention between two tokens compares two semantic objects. The value of a sensor at 14:03, in isolation, means nothing; the structure of a time series lives in *shapes over short stretches* — a rising edge, a dip, a local oscillation, the slope of a ramp — not in any single point. Point-wise attention asks how the scalar at one instant relates to the scalar at another, and the answer is almost always noise, because neither scalar carries local meaning; the attention map is being computed over the wrong objects. The linear model wins precisely because it never does point-wise comparison: it reads the whole window through learned weights and so sees the shape. A second symptom points the same way — with one token per step the token count is $N=L$, the attention map is $L\times L$, and lengthening the look-back is quadratically punished, while the time axis is heavily redundant so neighboring points are near-duplicates that waste the attention budget. Both symptoms have one cure.

I propose **PatchTST**, a Transformer forecaster resting on two ideas — **patching** and **channel-independence** — wrapped around a deliberately vanilla encoder. The first idea is to stop tokenizing per step and instead group a contiguous sub-series into one token, exactly as the Vision Transformer cut an image into $16\times16$ patches when it hit the identical wall. Take the $i$-th univariate series of length $L$, pick a patch length $P$ and a stride $S$ (the hop between consecutive patch starts), and slide a length-$P$ window along the series in steps of $S$; each placement is a patch in $\mathbb{R}^P$. Before patching I pad the end by appending $S$ repeated copies of the last value $x_L$ — because if $(L-P)$ is not a multiple of $S$ the final window never reaches $x_L$, and the most recent value is exactly the one a forecaster least wants to drop. Padding by precisely the stride slides one more full window into existence that includes the true end, adding exactly one patch, so the patch count is
$$N = \left\lfloor \frac{L-P}{S} \right\rfloor + 2.$$
With $L=336,\,P=16,\,S=8$ this gives $\lfloor 320/8\rfloor + 2 = 42$; with $L=512$ it gives $64$. The payoff is quantitative: attention is $O(N^2 D)$ per layer, and patching takes $N$ from $L$ down to $\approx L/S$, so the attention cost drops by a factor of about $S^2$ — with $S=8$, roughly $64\times$. The quadratic-in-$L$ wall everyone else attacked by mutilating the kernel, I simply walk around by changing what a token is. And the saving compounds: because long windows are now affordable, I can feed the longer, more informative history the data wants instead of being forced to $L=96$. I take $S$ a little smaller than $P$ so consecutive patches overlap and no local shape is split cleanly between two patches; $P=16,\,S=8$ gives half-overlap. Too small a $P$ drifts back toward point tokens with no local shape; too large and $N$ collapses so attention has almost nothing to attend over — the $8$–$16$ range is the balance, and it is robust across patch-length sweeps.

The second idea answers the other half of why the linear model wins: how to handle the $M$ channels. The standard multivariate Transformer mixes them, projecting the whole $M$-vector at each step into one token, so all channels are fused at the input and forced under a single attention pattern. The winning linear model does the opposite — each channel through its own shared-form map, no cross-channel mixing — so the empirical fact is that mixing does not help. Reasoning out why mixing is worse despite being more expressive: first, *adaptability* — one shared attention map cannot be right for a slow trend, a sharp daily cycle, and near-noise all at once, whereas if each channel passes through the backbone separately each series produces its own attention map, free to lock onto its own lag structure; second, *data efficiency* — learning genuine cross-channel correlation jointly with temporal structure is a far larger hypothesis space that needs much more data than these modest benchmarks provide, while a channel-independent model only learns along the time axis and converges with far less; third, *overfitting* — mixing fits spurious cross-channel coincidences that hold in the training window and do not generalize, and indeed mixing models overfit after a few epochs while independent ones keep improving. So I process each channel independently — but not with $M$ separate models, which would multiply the parameters and share nothing. The right design is *one* backbone with one weight set, applied independently to each univariate series: channel-independent in the forward pass, weight-shared in the parameters — a global univariate model that sees each series alone yet learns from all of them. A useful consequence is that the weights are blind to the channel count, so the number of series at train time need not match test time, and the univariate case is just the native $M=1$ instance. The implementation cost is a single reshape: start from $[B, M, L]$, patch to $[B, M, N, P]$, fold the channel axis into the batch to get $[B\cdot M, N, P]$ — an ordinary batch of $B\cdot M$ length-$N$ token sequences any Transformer consumes — run the encoder, reshape back. The $M$ weight copies are conceptual; the channels merely ride in the batch dimension.

The backbone is a *vanilla* Transformer encoder, and I keep it vanilla on purpose: the whole thesis is that the input representation, not the attention kernel, was the problem, so sneaking in a fancy kernel would muddy what is responsible. Each patch is linearly projected into the model space of dimension $D$ by a trainable $W_p\in\mathbb{R}^{D\times P}$. Because attention is permutation-invariant the patches are otherwise an unordered set, so I add a positional encoding $W_{pos}$, one vector per patch slot, to restore temporal order. Then standard multi-head scaled dot-product attention over the $N$ patch tokens,
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,$$
the division by $\sqrt{d_k}$ keeping the logits from growing with head dimension and saturating the softmax, with multiple heads so different heads capture different lag relationships, followed by a position-wise feed-forward sublayer $D\to F\to D$ and residual connections around both sublayers. One choice I do not make on autopilot is the normalization: the default Transformer uses LayerNorm, which normalizes within each token, but time series carry outlier steps — spikes, sensor glitches, regime jumps — and an outlier landing in a patch will drag that token's own mean and variance around. BatchNorm instead normalizes each feature across the batch (here the $B\cdot M\cdot N$ patch positions), diluting a single outlier patch among all the others, and it is measured to beat LayerNorm for time-series Transformers; mechanically BatchNorm1d wants the feature axis at dimension 1, so I transpose $D$ into place and back.

The head turns the per-series encoder output $z^{(i)}\in\mathbb{R}^{D\times N}$ into the next $T$ values by flattening the $D\times N$ representation into a single $D\cdot N$ vector and applying one linear layer to $T$. Per series this matrix is only $(D\cdot N)\times T$, modest and shared across series — deliberately *not* the monstrous $(L\cdot D)\times(M\cdot T)$ joint head a point-wise channel-mixing model needs to map all $L$ step-representations of all $M$ channels to all outputs at once, which is exactly the thing that overfits when downstream data is scarce. Finally the data forces a wrapper for distribution shift: these series are non-stationary, so a test window can sit at a level the training windows never visited, and feeding raw values would make the model track an offset and scale it has no stable handle on while also learning shape. I normalize each instance reversibly — over the look-back axis subtract the mean $\mu=\frac1L\sum_t x_t$, then divide by $\sqrt{\sigma^2+\epsilon}$ with the biased variance ($\texttt{unbiased=False}$, the actual window scale) and $\epsilon=10^{-5}$ to keep a flat series from dividing by zero — so the encoder always sees roughly zero-mean unit-variance shapes regardless of where the window sits, decoupling shape-learning from level-tracking. The mean is detached so it acts as a fixed offset; at the end the forecast is de-normalized, multiplied by the stored stdev and added back to the mean, broadcast over the $T$ steps. The training objective is mean squared error per channel averaged over channels, $\mathbb{E}_x \frac{1}{M}\sum_i \lVert \hat{x}^{(i)} - x^{(i)} \rVert_2^2$, which needs no autoregressive decoder and keeps the comparison against the linear forecaster clean. The same architecture supports self-supervised pre-training by swapping the head for $D\to P$ and masking whole *non-overlapping* patches (non-overlapping so a visible patch cannot leak a masked neighbor's contents) — and masking whole patches rather than single steps is what makes the task non-trivial, since a lone masked step is recoverable by interpolating its neighbors while a whole missing sub-series demands genuine signal understanding; with the shared channel-independent weights the pre-training and fine-tuning channel counts can even differ. That is why the patch token, not the kernel, is the load-bearing choice.

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
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
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
        padding = stride

        # patching + per-patch linear embedding + positional embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # vanilla Transformer encoder over patch tokens, BatchNorm normalization
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

        # prediction head: width = d_model * number-of-patches
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # channel-independent patching: channels folded into the batch axis
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, patch_num, d_model]

        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [bs, nvars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                    # [bs, nvars, T]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, T, nvars]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [bs, T, nvars]
        return None
```

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)   # W_p
        self.position_embedding = PositionalEmbedding(d_model)             # W_pos
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                # x: [bs, nvars, L]
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
```
