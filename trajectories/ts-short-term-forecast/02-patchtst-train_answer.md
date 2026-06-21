The linear floor came in at SMAPE 13.39 on Monthly, 10.50 on Quarterly, 14.36 on Yearly — and the ordering across regimes is exactly the one the affine hypothesis predicted: best where there are the most seasonal cycles to read linearly (Quarterly), worst on Yearly, the trend-dominated regime with the twelve-step look-back and six-step horizon. So the floor is real and honest under the SMAPE loss. But two things in that result are diagnostic of what an affine model *cannot* do. The Yearly number sits far above the other two — a regime where a fixed linear combination of a short window simply runs out of structure to exploit. And more generally, one shared affine map has to serve M4's tens of thousands of series, each at its own level and scale, with no per-window normalization, so it cannot decouple "what shape" from "what level," and any series far from the training average is systematically off. The question, then, is whether I can keep the linear model's robustness on these short series while adding the two things DLinear left on the table — a learned nonlinear representation and per-instance normalization — without reviving the overfitting that sank the heavy attention forecasters.

The sharpest version of that question is one the whole attention line refused to ask. A linear map beat Informer, Autoformer, FEDformer — every clever surgery on the attention kernel. The usual reading is "attention is the wrong tool for this data." I want to chase the opposite reading: attention is fine, and it was being fed garbage. Look at what a token *is* in those models — a single time step, one scalar at time $t$. A scalar has no standalone meaning the way a word does; the information in a series lives in *shapes over short stretches* — a ramp, a dip, a local oscillation. Point-wise attention asks "how does the scalar at step 14 relate to the scalar at step 3?" and the answer is mostly noise. That would explain DLinear winning: it never does point-wise comparison, it reads the whole window at once through learned weights, so it sees the shape. The fix is not a new attention kernel; it is to change the token.

I propose **PatchTST**: cut each univariate series into contiguous sub-series *patches*, let each patch be a token, and run a deliberately vanilla channel-independent Transformer over those patch tokens, wrapped in reversible instance normalization. A length-$P$ patch is a little shape — exactly the object attention should compare — and this is the vision move (cut the image into $16\times16$ patches) carried over to a series. Patching buys three things, but I keep them straight because two barely matter here. It gives local semantic tokens; it cuts attention cost by $S^2$ since the token count drops from $L$ to $\sim L/S$; and it makes a longer look-back affordable. But on M4 the window is $\text{seq\_len}=36$ at most so cost was never the constraint, and the harness fixes $\text{seq\_len}=2\cdot\text{pred\_len}$ so a longer look-back is unavailable. The load-bearing benefit here is purely the first one — shape-tokens instead of scalar-tokens — which is the only honest way to test whether depth-plus-attention can beat the linear floor on short series. With $\text{patch\_len}=16$, $\text{stride}=8$ (half-overlap, so no local shape is split cleanly down the middle), and end-padding $\text{stride}$ copies of the last value so the most recent step is never dropped, the patch count is $N = \lfloor (\text{seq\_len} - P)/S \rfloor + 2$: four for Monthly, two for Quarterly. For Yearly the window (12) is *shorter than the patch length* (16), so the replication-pad-then-unfold path still yields the two padded patches — a thin attention, but it does not crash.

The second half of why the linear model wins is the channel axis, and M4 makes the decision. The heavy multivariate Transformers *mix* channels — at each step they fuse the whole channel vector into one token, forcing every channel under one shared attention pattern — and DLinear does the opposite, each channel through its own map, and wins. Mixing is worse for three reasons that all bite harder on short data: one shared attention map cannot be right for a slow-trend channel and a sharp-cycle channel at once; learning cross-channel *and* temporal structure jointly is a far larger hypothesis space, starved on short series; and mixing fits spurious cross-channel coincidences and overfits in a few epochs. On M4 this is not even a tradeoff — every series is its own univariate channel ($\text{enc\_in}=c_{\text{out}}=1$), so channel-mixing has nothing to mix. So I process each channel **independently** through one shared backbone, channels folded into the batch axis ($[B, 1, L] \to [B, N, P]$): it costs nothing but a reshape, and the univariate M4 setting is its native case.

The backbone is kept *vanilla* on purpose — the thesis is that the token was the bug, so a fancy kernel would muddy what is responsible. Each patch is linearly projected to $d_{\text{model}}$ with a no-bias embedding (an instance-normalized patch already has its level removed, so a per-patch additive offset buys nothing), a learned positional embedding is added (patches are an unordered set to attention, but order is everything in a series), then standard multi-head scaled dot-product attention over the $N$ patch tokens with the usual $1/\sqrt{d_k}$ scaling, a position-wise FFN $d_{\text{model}} \to d_{\text{ff}} \to d_{\text{model}}$, and residuals. One non-default choice I do *not* make on autopilot: the normalization is **BatchNorm**, not LayerNorm. Time series have outliers — a spike, a glitch — and an outlier step inside a patch skews that token's *within-token* statistics, dragging LayerNorm around; BatchNorm normalizes each feature *across* the batch of patch positions, so a single outlier patch is diluted rather than corrupting its own normalization. The head is the simplest faithful thing: flatten the $d_{\text{model}} \times N$ encoder output per series and project with one linear layer to the horizon ($\text{head\_nf} = d_{\text{model}} \cdot N$), shared across series, sidestepping the oversized joint head a mixing model needs.

The piece that directly answers DLinear's level-tracking failure is the **reversible instance normalization**. Before patching, per instance I subtract the look-back mean and divide by $\sqrt{\text{var} + 10^{-5}}$ using the *biased* variance — I want the window's actual scale, not an unbiased estimator — with both statistics detached, since they are normalization constants and not parameters. The encoder then always sees roughly zero-mean unit-variance shapes regardless of where the window sits, the exact decoupling of shape-learning from level-tracking DLinear could not do; at the end the forecast is denormalized by multiplying by the std and adding back the mean.

A caution before committing, because the harness protocol is not PatchTST's own. The fixed Custom settings pass $d_{\text{model}}=512$, $d_{\text{ff}}=512$, $e_{\text{layers}}=2$ — far wider than PatchTST's own M4 script ($d_{\text{model}}=128$, $e_{\text{layers}}=3$) — so a 512-wide, 2-layer encoder with a flatten head of width $512\cdot N \to \text{pred\_len}$ is a large model fitting two-to-four tokens from very short series. The instance normalization and channel-independent shared weights are precisely the regularizers that should keep this from blowing up, and SMAPE training plus 10-epoch early stopping (patience 3) cap the overfitting window. Still, this is the rung most exposed to the mismatch: a wide attention model on two-token sequences is capacity the data may not be able to use. I expect to beat the linear floor on *every* regime, with the largest absolute gain on Yearly where DLinear was most strained. What I am not confident about is by how much — if the gain over DLinear is only a few tenths rather than a clear separation, that is the tell that attention over so few short-window tokens is near its ceiling, and that the next rung should stop adding generic capacity and instead exploit the *multi-scale* structure of these series directly.

```python
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
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
    """PatchTST: patch tokens + vanilla channel-independent Transformer + RevIN."""

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
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
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        # prediction head: width = d_model * number-of-patches
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev

        # channel-independent patching: channels folded into the batch axis
        x_enc = x_enc.permute(0, 2, 1)                  # [bs, nvars, L]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [bs*nvars, patch_num, d_model]

        enc_out, attns = self.encoder(enc_out)          # [bs*nvars, patch_num, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)           # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                    # [bs, nvars, T]
        dec_out = dec_out.permute(0, 2, 1)              # [bs, T, nvars]

        # de-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]       # [bs, T, nvars]
        return None
```
