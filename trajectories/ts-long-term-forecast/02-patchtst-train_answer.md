The linear control came back competitive: on ETTh1, the small, strongly trend-and-seasonal dataset, DLinear landed MSE $0.3962$ / MAE $0.4108$, with Weather at $0.1962$ / $0.2561$ and ECL at $0.2104$ / $0.3016$, a mean MSE of $0.2676$. ECL is the tell — its 321 channels and its worst-by-far MAE expose exactly the cross-channel structure the channel-blind linear map leaves on the table. But before I reach for channels I want to settle the temporal half: a plain linear projection matched or beat all of Informer/Autoformer/FEDformer, which means either self-attention is the wrong tool for this data, or — the possibility I will chase — attention is fine and we have been feeding the series into it in a way that destroys the structure attention is good at finding. I refuse to touch the attention kernel and instead interrogate the *token*.

In every model in that lineage, and in the scaffold's per-step convention, a token is a single time step: one scalar at time $t$. But the meaning of a sensor's value at 14:03 is nothing in isolation — less than a character, which at least belongs to an alphabet. The information in a series lives in *shapes over short stretches*: a rising edge, a dip, the slope of a ramp. Point-wise attention asks "how does the scalar at 14:03 relate to the scalar at 09:17?" and the answer is almost always noise, because neither scalar means anything alone — the attention map is computed over the wrong objects. A second symptom points the same way: with one token per step the token count $N$ equals the sequence length $L$, so the attention map is $L\times L$ and lengthening the look-back is quadratically punished, which is why the field defaults to short windows; and people who kept a long window cheaply by *down-sampling* still forecast well, which says neighboring steps carry overlapping, compressible structure, not independent information. So I propose **PatchTST**: change what a token is. Group a contiguous local stretch into one token, exactly as vision cut an image into patches when per-pixel attention proved hopeless.

Concretely, take the $i$-th channel's univariate series, pick a patch length $P$ and a stride $S$ (the hop between patch starts), and slide a width-$P$ window in steps of $S$; each placement is one patch in $\mathbb{R}^P$, and a patch of sixteen consecutive steps is a little shape — a ramp, a bump, a level — which is precisely what attention should be comparing. The last steps of the look-back matter most for forecasting, so I pad the end with $S$ copies of the last value before patching; this guarantees one extra full window reaching the true end and adds exactly one patch, giving $N = \lfloor (L-P)/S\rfloor + 2$. With $L=96$, $P=16$, $S=8$ that is $\lfloor 80/8\rfloor + 2 = 12$ patches per channel, so the attention map is $12\times 12$ instead of $96\times 96$ — an $S^2\approx 64\times$ cut in attention cost, walking around the quadratic-in-$L$ wall by changing the token rather than mutilating the kernel. The channel decision is decided by the linear result: DLinear was channel-shared and channel-blind yet competitive, so channel-independence was not what cost it. I keep it, processing each of the $C$ channels through the *same* shared Transformer backbone as its own patch sequence, never mixing channels in attention — the cleanest possible test of "is patched temporal attention better than a linear temporal map," holding the channel treatment fixed and changing only the temporal operator. This makes the $C$ channels a batch dimension: reshape $[B, C, L]$ so the backbone sees $B\cdot C$ independent patch sequences, $C\times$ more training sequences, a real regularizer on the small datasets.

Patching does not absorb level drift the way the loader partly did, so I fold in reversible per-window instance normalization: per window and channel, subtract the look-back mean and divide by the look-back standard deviation before patching, then multiply back the std and add back the mean after the prediction head, so the backbone only ever sees zero-mean, unit-variance shapes. The backbone itself I keep stock, because the fix is the tokenization, not a new kernel: embed each patch with a linear map $P\to d_{\text{model}}$ plus a positional embedding over the $N$ patch positions (now that the tokens are order-bearing local shapes, the positional encoding does honest work instead of papering over a permutation mismatch), then run $e_{\text{layers}}$ standard encoder blocks — multi-head self-attention over the $N$ patch tokens, then a position-wise feed-forward. One detail from the assembly matters: the encoder norm is a **BatchNorm** over the feature axis rather than LayerNorm, which is steadier when the token count $N$ is small (12 here) and per-token statistics are noisy. The head takes the encoded $[B\cdot C, N, d_{\text{model}}]$ patches, flattens the $N\cdot d_{\text{model}}$ features per channel, and maps them with one linear layer to the full horizon — direct multi-step again, no decoder, the same generation step that already worked for the linear model. I keep $P=16$ from the config and hardcode $S=8$ (the loop passes no stride); I am running this at a *short* $L=96$ and one fixed capacity ($d_{\text{model}}=512$, $e_{\text{layers}}=2$, $n_{\text{heads}}=8$) for all three datasets, where the method's own recipe would tune layers and heads per dataset and use a much longer look-back — so patching's signature long-history headroom is unavailable here, and the architecture is faithful while the regime is the scaffold's.

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

    def forward(self, x):                                       # x: [B, nvars, d_model, patch_num]
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """PatchTST: channel-independent self-attention over subseries patches, direct multi-step."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride                                        # pad end so the final value is kept

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # reversible per-window instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)                          # [B, nvars, L]; channels -> batch
        enc_out, n_vars = self.patch_embedding(x_enc)          # [B*nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                  # [B, nvars, d_model, patch_num]

        dec_out = self.head(enc_out)                           # [B, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                     # [B, pred_len, nvars]

        # de-normalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
