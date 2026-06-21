The linear floor told me what I suspected, in numbers. DLinear landed ETTh1 at MSE 0.0644 / MAE 0.1878 — respectable, the smooth target is largely recoverable from its own history — but Weather sat at 0.005652 / 0.0627 and ECL at 0.3873 / 0.4512, loose in exactly the way a no-fusion control should be on the datasets where the target leans on its covariates. Yet the ETTh1 number is the tell that stops me from leaping straight to fusion. A channel-independent linear map gives up *two* distinct things — cross-channel information *and* any nonlinear read of the target's own temporal structure — and ETTh1 is the dataset where fusion has the least to add, so its residual gap points at the *per-channel temporal model itself*. If I jumped to a cross-variate machine now and it beat DLinear, I would not know whether the win came from fusion or from finally giving the target a nonlinear temporal model. So the disciplined next rung upgrades the temporal model to its strongest channel-independent form and holds channel-independence fixed.

That forces a question: what is the strongest way to model one channel's temporal structure without looking at the others? The Transformer is the obvious reach, but there is an embarrassing fact in the way — the linear map I just ran *beats* the deployed temporal-attention Transformers (Informer, Autoformer, FEDformer) on these benchmarks. If attention over time earned its keep, a linear map would have no business winning. The diagnosis is that the failure is not in the attention kernel but in *what a token is*. In those models a token is a single time step — one scalar at time $t$. What is the meaning of a sensor's value at 14:03 in isolation? Almost nothing; the information in a series lives in *shapes over short stretches* — a rising edge, a dip, the slope of a ramp. Point-wise attention asks how the scalar at 14:03 relates to the scalar at 09:17, and the answer is mostly noise. Two symptoms confirm it: with one token per step the attention map is $L \times L$ so a long look-back is quadratically punished, and you can down-sample the window — keep every fourth step — and forecast about as well, which says neighboring steps carry overlapping, compressible structure rather than independent information.

I propose **PatchTST**: a channel-independent patch Transformer. The move is the one vision made at the same wall — the Vision Transformer cut an image into $16\times16$ patches because one pixel is meaningless and per-pixel attention is hopeless. So I cut each channel's length-$L$ look-back into contiguous sub-series **patches**, one patch per token. Take one channel's history $x^{(i)} \in \mathbb{R}^L$, pick a patch length $P=16$ and stride $S=8$, slide a width-$P$ window in steps of $S$, and each placement is a patch in $\mathbb{R}^P$ — a little local shape, a ramp or bump or level, which is exactly the kind of object attention should compare. The patch count $N = \lfloor (L-P)/S \rfloor + 1$ (plus one for the end padding the embedding adds) is far smaller than $L$, so attention is cheap *and* each token is meaningful. I embed each patch with a shared linear map $\mathbb{R}^P \to \mathbb{R}^d$ plus a positional embedding, run a stack of vanilla Transformer encoder layers over the patch tokens, and flatten the resulting $N$ patch representations through a linear head to the 96-step horizon.

The load-bearing decision — the one that keeps this rung a clean control — is that the backbone is **channel-independent**: the same patch-embedding, the same encoder, the same head are applied to every channel separately, and channels never attend to each other. I implement that by folding the channel axis into the batch — reshape $(B,L,C)$ so each channel becomes its own sequence of patch tokens, run the shared backbone, then reshape back. This does for the temporal model what DLinear could not — patches plus attention give a strong nonlinear, multi-scale read of one channel's own dynamics — while holding the channel-independence axis at exactly where DLinear had it, so whatever this rung gains over DLinear is attributable to *better per-channel temporal modeling*, not fusion.

Two pieces of plumbing matter and cost little. First, **per-instance normalization** in the Non-stationary-Transformer style: subtract the per-instance mean and divide by the per-instance std of the look-back before patching, then add them back after the head. These benchmarks carry heavy distribution shift between train and test windows, and normalizing each instance removes the shift the model would otherwise waste capacity chasing — and because it is applied per channel, it smuggles in no cross-channel coupling. Second, **BatchNorm** in the transpose-BN-transpose form instead of LayerNorm inside the encoder, the reference choice for patch tokens, which trains more stably here. With `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`, `stride=8`, dropout 0.1, the model stays small and fast.

I expect this to beat DLinear on all three datasets — better temporal model — with ETTh1 dropping into the high 0.05s as pure evidence the per-channel model was the bottleneck there. On Weather and ECL it should improve but *leave a residual gap*, because it still cannot read the other channels; that gap is precisely the quantity the next, cross-variate rung must close, and I want it on the record.

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

    def forward(self, x):  # [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class Model(nn.Module):
    """Channel-independent patch Transformer (PatchTST)."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # per-instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # patch + embed (channel folded into batch -> channel-independent)
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)        # [bs*nvars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)               # [bs, nvars, d_model, patch_num]

        dec_out = self.head(enc_out).permute(0, 2, 1)        # [bs, pred_len, nvars]

        # de-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
