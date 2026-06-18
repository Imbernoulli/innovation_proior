**Problem (from step 1).** The decomposition-linear floor is above chance everywhere but blunt: linear
boundary, no local temporal feature, channels mixed only in the head. It is weakest on 26-class Handwriting
(0.2306) and barely above chance on 4-class EthanolConcentration (0.2890), strongest relatively on binary
FaceDetection (0.6822). The missing ingredients are local-shape tokens and a nonlinear encoder.

**Key idea.** Patch + channel-independent vanilla Transformer. Cut each channel's series into overlapping
length-`P` sub-series patches (`P = 16`, stride `S = 8`), each a local-shape token; fold channels into the
batch so one shared encoder runs over each channel independently (no cross-channel attention — channel
mixing overfits the small UEA sets); attend over the `N` patch tokens with a vanilla BatchNorm Transformer;
flatten the per-channel `[d_model, N]` representations across *both* patch and channel axes and project to
`num_class`. Per-instance normalization (subtract/divide by window mean/std) conditions the heterogeneous
channels.

**Why it works (and where it won't).** Local-shape tokens are the objects attention can actually compare,
and a nonlinear encoder gives a non-hyperplane boundary — both attack exactly what the linear floor lacked,
so Handwriting and EthanolConcentration are where gains should appear. Channel-independence forbids
in-encoder cross-channel modeling, so FaceDetection (all cross-channel covariance) should at best match the
floor — its signal must be recovered only in the final flatten-across-channels head. Like the floor, this
path does not consult `x_mark_enc` (padding patches are embedded and attended) — a gap the next rung attacks.

**Scaffold edit / hyperparameters.** `PatchEmbedding(d_model=128, patch_len=16, stride=8, padding=8,
dropout=0.1)`; `e_layers=3`, `n_heads=16`, `d_ff=256`, `factor=1`, BatchNorm encoder norm;
`head_nf = d_model · (⌊(seq_len−16)/8⌋ + 2)`; head `Linear(head_nf · enc_in, num_class)`. Frozen protocol:
RAdam, `lr 1e-3`, `batch 16`, CrossEntropy, patience 10.

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
