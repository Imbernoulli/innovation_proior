**Problem.** TimesNet's nonlinearity fixed ETTh1 (0.0803 MSE) but ECL stayed its worst (0.0915) — the
`DataEmbedding` forces all 321 ECL channels into one shared `d_model` feature per timestep, averaging away
the per-client identity needed to reconstruct a specific masked client. The next rung must stop mixing
channels and give attention units worth attending to.

**Key idea (channel-independent patch Transformer).** A single timestep has no standalone meaning; a
length-16 *patch* is a local shape worth comparing. Split each channel's window into overlapping patches
(`patch_len=16`, `stride=8`), embed each patch, and run a standard Transformer encoder over the patch
sequence — but with batch and channel folded together (`[B·C, num_patches, d_model]`), so every channel is
its own sequence through *shared* weights. Cross-channel strength is shared through the parameters, not
mixed in the features, so per-channel identity survives — the fix for ECL. A flatten head projects the
encoded patches back to a length-96 reconstruction per channel.

**Why (and the imputation-specific parts).** Channel independence removes the input-mixing cap that held
ECL back; patch tokens make self-attention compare real shapes instead of point-wise noise; overlapping
patches give every masked position multiple patch contexts. Normalisation is over *observed entries only*
(`sum(mask == 1)` denominator, holes re-zeroed after centring, detached, undone after the head, repeated
over `seq_len` — not the forecasting `pred_len`). No time features: each channel is reconstructed from its
own shape sequence.

**Hyperparameters (the Custom.py eval config).** `patch_len=16`, `stride=8`, `d_model=512`, `d_ff=512`,
`e_layers=2`, `n_heads=8`, `factor=3`, `dropout=0.1`, `activation='gelu'`, `BatchNorm` encoder norm; loop
Adam `lr=1e-3`, `batch_size=16`, `train_epochs=10`, masked-only MSE.

```python
# models/Custom.py — step 3: PatchTST (channel-independent patch Transformer) for imputation
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

    def forward(self, x):                                   # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """PatchTST for imputation: channel-independent patch Transformer."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len                     # imputation: pred_len = seq_len
        padding = stride

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
                    dropout=configs.dropout, activation=configs.activation)
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                head_dropout=configs.dropout)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Non-stationary normalisation over OBSERVED entries only
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)                      # [B, C, T]
        enc_out, n_vars = self.patch_embedding(x_enc)       # [B*C, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)               # [B, C, d_model, patch_num]

        dec_out = self.head(enc_out)                        # [B, C, seq_len]
        dec_out = dec_out.permute(0, 2, 1)                  # [B, seq_len, C]

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return None
```
