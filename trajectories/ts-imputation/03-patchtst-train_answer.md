The periodic 2-D model moved every number in the right direction, and where it moved them most is the tell. On seed 42 it cut ETTh1 from 0.1498 to 0.0803 MSE and ECL from 0.1132 to 0.0915, with Weather down to 0.0293 — the nonlinearity bought the ETTh1 transients and the embedding's implicit channel mixing bought a real chunk of ECL. But ECL is *still* the worst of the three at 0.0915, and that is the problem, because ECL is the dataset where the difficulty is least about hard temporal variation (the clients are smooth and strongly periodic) and most about exploiting correlation across hundreds of channels. The mechanism that was supposed to fix channel-blindness still does its worst work precisely there. The reason is that TimesNet's `DataEmbedding` projects all `enc_in` channels into one shared $d_\text{model}$ feature vector per timestep; on a 7-channel set like ETTh1 that mixing is benign, but on ECL it forces 321 correlated-but-distinct client series through one common basis, and the per-client idiosyncrasies — exactly the detail needed to reconstruct a *specific* masked client — get averaged away. Forced channel mixing at the input is the cap.

I propose PatchTST: a channel-independent patch Transformer. It answers the diagnosis on two fronts. First, on high-channel data I should not mix channels in the representation at all — I should model each channel independently and let parameter *sharing*, not feature mixing, do the cross-channel work. A single shared model trained on all 321 clients still learns the common dynamics from all of them, but each client is reconstructed from its own observed values with its identity intact. Second, I revisit what made the linear interpolator competitive in the first place: a single timestep carries almost no standalone meaning, and the signal lives in shapes over short stretches. TimesNet's atomic unit is still effectively a single (embedded) timestep. So I make the unit a *patch* — a short contiguous run of timesteps, say length 16 — which is a little shape: a rising edge, a dip, a local oscillation, an object worth comparing with attention. Patching does three things at once: it gives each token genuine semantic content so self-attention finally compares objects worth comparing rather than point-wise noise; it cuts the sequence length attention sees from 96 to roughly $96/\text{stride}$ patches, dropping both the quadratic cost and the noise; and it aggregates local information within a patch before any global mixing, which for imputation means a masked entry is reconstructed first from its immediate patch-mates and then from the patch-level context — exactly the locality interpolation wants.

The mechanism is to split each channel's window into overlapping patches, embed each, run a standard Transformer encoder over the patch sequence per channel independently, and project back to a length-96 reconstruction. With `patch_len=16` and `stride=8` the patches overlap by half, which matters: a masked entry near a patch boundary still sits in the interior of a neighbouring patch, so every position gets multiple patch contexts and there are no boundary blind spots. The library's `PatchEmbedding` replication-pads the end by `stride` before unfolding (giving the last positions a full patch and a clean count), projects each `patch_len`-vector to $d_\text{model}$ with a linear layer, and adds a positional code. Crucially it folds the batch and channel axes together, so the encoder runs on $[B\cdot C, \text{num\_patches}, d_\text{model}]$ — that *is* channel independence in implementation form: every channel of every batch element is its own sequence through the same shared encoder weights, shared parameters with no feature mixing across channels and per-channel identity preserved. On ECL this is the decisive change versus TimesNet.

The encoder is a plain Transformer encoder — full self-attention over the patch sequence plus feed-forward — with the one library detail that the norm is `BatchNorm1d` over the $d_\text{model}$ axis (transposed in and out) rather than LayerNorm, which is empirically steadier for patch tokens. Self-attention is now doing something sensible: it asks which other local shapes in this channel's window resemble or inform the shape around the masked region, and because the tokens are real shapes the attention map is meaningful instead of the point-wise noise that once made attention lose to a linear map. After the encoder I have $[B\cdot C, \text{num\_patches}, d_\text{model}]$; I reshape to $[B, C, \text{num\_patches}, d_\text{model}]$, permute to put $d_\text{model}$ before the patches, and a `FlattenHead` flattens $(d_\text{model}, \text{num\_patches})$ and linearly projects to the full $\text{seq\_len}=96$ per channel. The flatten-then-project head is the right choice for imputation: it lets every output timestep draw on the whole encoded patch sequence at once — a length-one path from any patch to any reconstructed position — which is what interpolation across a window wants, as opposed to a per-patch decoder that would localise too much. The head width follows directly: $\text{head\_nf} = d_\text{model} \cdot \big\lfloor (\text{seq\_len} - \text{patch\_len})/\text{stride} + 2 \big\rfloor$.

The imputation-specific normalisation is the same masked-statistics device I derived for the periodic model, and I must keep it — patching is even more sensitive to corrupted statistics, since a patch straddling a hole would otherwise carry a fake zero into its embedded shape. I compute the mean and std over observed entries only (`torch.sum(mask == 1, dim=1)` as the denominator), subtract the mean, re-zero the holes with `masked_fill(mask == 0, 0)` so the patches see real holes rather than spurious values, divide by the std, run the patch encoder and head, and undo the normalisation afterward with the stored, detached mean and std repeated over `seq_len` (not the forecasting `pred_len`, because the output length equals the input length here). Time features are deliberately *not* used: patch embedding consumes only the value patches plus positional codes, which is part of the channel-independent design — each channel is reconstructed from its own shape sequence, and the calendar phase is recoverable from the periodic structure within the patches themselves.

This answers the TimesNet bottleneck head-on: forced channel mixing capped ECL, channel independence removes that cap, so the largest gain of the whole ladder should land on ECL. The patch tokenisation should also help ETTh1 and Weather by handing attention meaningful units, but those were already in good shape so the margins there are smaller; ETTh1 is the one place I am least sure, because its sharp hourly transients are exactly the fine within-window nonlinearity that period-folding 2-D convs capture and patch-level attention might smooth over. The honest shape of the expected result is therefore a method that wins by fixing the channel bottleneck rather than the temporal one — a large ECL gain dragging the mean below TimesNet even if ETTh1 is roughly a tie.

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
