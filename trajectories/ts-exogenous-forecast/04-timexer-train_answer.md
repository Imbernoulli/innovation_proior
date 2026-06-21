The cross-variate rung drew the exact picture I predicted. On Weather, iTransformer dropped to MSE 0.001346 / MAE 0.026998, beating PatchTST's 0.001652 / 0.029337 — the fusion gap really was a fusion gap, and cross-variate attention closed it. On ECL it went to 0.301479 / 0.405040 from 0.317887 / 0.394697: MSE improved as expected from 321 channels of cross-client structure, but **MAE slipped** — the signature I flagged, since iTransformer's $O(N^2)$ attention is indiscriminate, modeling interactions into channels I never score while their noise flows into the target's token and inflates the typical-case error. And on ETTh1 it landed 0.058923 / 0.186680, essentially tied with and slightly behind PatchTST's 0.058292 / 0.182538, because collapsing each channel's whole series into one token threw away the intra-series detail PatchTST's patches preserved. So I hold two rungs that each win where the other loses: PatchTST keeps the target's own fine temporal structure but is blind to covariates; iTransformer reads the covariates but blurs the target's structure and fuses indiscriminately. The last rung has to stop choosing.

State the requirement sharply from the target's point of view, because the MS scoring makes the asymmetry the whole game — I score *one* channel, and the others are exogenous side-information that drives it. I want three things at once. One: the target's own temporal structure modeled at full resolution, the way PatchTST's patches do, because ETTh1 says that resolution is worth real error and iTransformer's single-token projection gave it up. Two: the exogenous channels allowed to *inform* the target, the way iTransformer's cross-variate attention does, because Weather says that information is worth real error. Three — the part neither rung gets right — the fusion must be *asymmetric*: the target should read *from* the exogenous variables without paying $O(N^2)$ compute to model interactions into channels it never predicts, and without their noise flowing back into it the way iTransformer's symmetric attention let it on ECL. So the design is forced: keep patching for the target's own series, keep variate-tokens for the exogenous channels, and connect them by a *directed* fusion that runs target ← exogenous and not the reverse.

I propose **TimeXer**: endogenous/exogenous separation. I split the representation by role. The **endogenous** stream is the target channel alone, treated exactly as PatchTST treats a channel — cut its length-96 look-back into patches of length 16 (stride 16), embed each as a token, and keep $\text{patch\_num} = \lfloor 96/16 \rfloor$ patch tokens carrying the target's local shapes at full resolution. The **exogenous** stream is every *other* channel, each treated as iTransformer treats a variate — collapse its whole look-back into a single variate token via the inverted embedding, giving $N-1$ tokens, one per covariate, each a clean summary of that channel's temporal profile. Two tokenizations, each matched to its role: fine for the thing I model in detail, coarse for the things I only need to read.

The connection is where the asymmetry lives. Within a layer, the endogenous patch tokens first attend among themselves — ordinary self-attention over the target's patches, which is PatchTST's intra-series modeling, untouched. Then comes the directed fusion, and the cost-control trick that makes it scale: I do *not* let every endogenous patch token cross-attend to every exogenous token (that would be $O(\text{patch\_num}\cdot N)$ and would let exogenous noise into every patch). Instead I add **one learnable global token** to the endogenous stream — a single token that, through the self-attention step, aggregates the whole target series into one summary — and I let *only that global token* cross-attend to the exogenous variate tokens. The cross-attention is directed by construction: queries come from the endogenous global token, keys and values from the exogenous tokens, so information flows exogenous → target and never the reverse. After it updates the global token, the token is folded back among the endogenous patch tokens (it sat in the same stream), a conv feed-forward block mixes everything, and the layer repeats. At the end the endogenous tokens — patches plus the now-exogenous-informed global token — are flattened through a linear head to the 96-step forecast for the target.

The crux is *why a single global token is enough* to carry all the exogenous influence. The worry is that one token is a bottleneck so tight it loses information. But consider what the target actually needs from the exogenous side: not the covariates' fine temporal detail — that is their business — but a summary of "given where the covariates are now, which way is the target being pushed." That is a low-dimensional ask, and one $d_{model}$-wide token updated by attention over the covariate tokens is amply expressive for it. The alternative, where every endogenous patch token cross-attends to every covariate token, re-weights each of the target's local shapes by 320 noisy clients with no shared denoised summary — each patch absorbs its own slice of covariate turbulence, which is precisely how iTransformer's symmetric fusion let ECL's noise inflate the target's typical-case error. The global token forces the covariate influence to be *consolidated* before it touches the target's fine structure, so the same attention that reads useful covariate signal averages away the per-channel noise. The bottleneck is not a limitation; it is the denoiser. And because only the global token queries the exogenous stream, the cost of fusion is $O(1\cdot N)$ per layer in the covariates rather than $O(N^2)$ — cheaper *and* cleaner on exactly the large panels where symmetric fusion struggled.

Checking against the two failures concretely: iTransformer's ETTh1 loss came from giving up patch-level detail; here the endogenous stream *is* patches, so that detail is preserved and ETTh1 should recover toward PatchTST's level while the small ETTh1 covariate signal stays available through the global token. iTransformer's ECL MAE regression came from symmetric fusion letting covariate noise in; here the fusion is one directed cross-attention through a single global-token bottleneck, so the exogenous influence is read, not injected wholesale, and ECL MAE should come back down while the cross-client information keeps MSE low. Weather's win, which came from cross-variate fusion, should hold or improve, because the fusion is still there, just routed more precisely.

A few MS specifics I have to get exactly right in the edit surface. The endogenous embedding (`EnEmbedding`) patches the **last** channel of `x_enc` — `x_enc[:, :, -1:].permute(0,2,1)` — unfolds it into length-16 patches with stride 16, value-embeds each plus a positional embedding, and concatenates one learnable global token per endogenous variate (exactly one, since `features=='MS'` makes `n_vars==1`). The exogenous embedding is `DataEmbedding_inverted` over `x_enc[:, :, :-1]` *together with* `x_mark_enc`, so the calendar features ride in as exogenous tokens too. The encoder layer runs endogenous self-attention, then global-token→exogenous cross-attention (queries from the last token of the endogenous stream, which is the global token; keys/values from the exogenous tokens), folds the updated global token back, and a conv feed-forward mixes. The head flattens the $\text{patch\_num}+1$ endogenous tokens ($\times d_{model}$) to `pred_len`. I keep per-instance normalization (`use_norm`), and crucially the de-normalization in MS mode uses the **target channel's** mean and std (`stdev[:, 0, -1:]`, `means[:, 0, -1:]`) since I forecast only the target. Configuration: `patch_len=16`, `e_layers=1`, `d_model=512`, `d_ff=512`, `n_heads=8`, `factor=3`, dropout 0.1, `use_norm=True`. This is the literal resolution of the PatchTST-vs-iTransformer tension the prior feedbacks measured: ETTh1 back toward the high-0.057s, Weather held at iTransformer's 0.0013, and ECL beating iTransformer on both metrics with MAE well below 0.405040.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding


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


class EnEmbedding(nn.Module):
    """Patch the endogenous (target) series + one learnable global token."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bs, n_vars, seq_len]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                     # append global token
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # endogenous self-attention over the target's patch tokens
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # directed fusion: only the global token (last) cross-attends to exogenous tokens
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)    # fold updated global token back
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    """Endogenous/exogenous separation forecaster (TimeXer), MS mode."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = getattr(configs, 'features', 'MS')
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.patch_num = int(configs.seq_len // self.patch_len)
        self.n_vars = 1 if self.features == 'MS' else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # endogenous = target channel patched; exogenous = the rest as variate tokens
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)               # [bs, nvars, d_model, patch_num+1]

        dec_out = self.head(enc_out).permute(0, 2, 1)        # [bs, pred_len, 1]

        if self.use_norm:
            # de-normalize with the target channel's statistics (MS)
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
