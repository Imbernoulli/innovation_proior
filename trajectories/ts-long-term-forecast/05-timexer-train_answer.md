TimeMixer landed where I predicted, and the prediction's logic now tells me what is still missing. ETTh1 fell to $0.3767$ — best on the ladder, recovering the ground iTransformer lost and edging past PatchTST; Weather fell to $0.1621$ / MAE $0.2089$, clearly best, the ideal multi-scale dataset paying off. So the bet held: multiscale disentanglement generalizes across datasets better than cross-variate attention. But ECL is the residue I expected: $0.1561$, better than PatchTST's $0.1819$ but still behind iTransformer's $0.1482$, because TimeMixer is channel-independent and never explicitly models the 321 coupled channels. After four rungs the scoreboard is honest and split — the best temporal model owns ETTh1 and Weather but loses ECL to the only model that modeled channels, which itself lost ETTh1. The two levers, fine intra-series temporal structure and cross-variate correlation, have never been held in the same model that keeps both their virtues. The next rung has to hold both at once.

Going back to first principles on the multivariate problem with channels included: fix attention on the one channel I am predicting — the *target* — and the others are side channels that genuinely drive it (in ECL, one client's consumption is shaped by the regional load the other clients also reflect). From the target's point of view I want the side channels to *inform* its forecast and nothing else: I do not want the target's representation paying to model interactions *into* channels it never predicts, and I do not want their noise flowing back into it. That per-channel asymmetry is the whole thing, and none of the four rungs is shaped around it. The channel-independent rungs (linear, PatchTST) preserve intra-target temporal detail but make the side channels invisible — why they trailed on ECL. iTransformer crushes the target's whole series into one coarse vector, treats every channel as an equal token so self-attention is $O(C^2)$ — quadratic in exactly the largest dimension for ECL's 321 channels — and most of that compute models interactions into channels never predicted while letting their noise flow into the target. The assumption to break is that whatever granularity I pick, I apply it uniformly. The target is the thing whose internal temporal wiggle I must nail step by step; the side channels are only there to nudge it — I care *that* a side channel shifts the target, not about the micro-structure inside it. Different demands, so different granularities.

So I propose **TimeXer**: patch the target finely (PatchTST's good idea, preserving intra-target temporal detail) and variate-token the exogenous channels coarsely (iTransformer's good idea, clean cross-variate correlation), and let the two meet through a controlled asymmetric gate. The endogenous (target) path cuts the target series into $P$-length patches with the patch embedding and additionally prepends a single learnable **global token** per channel — a summary slot standing in for "the whole target series" when it talks to the exogenous side. Self-attention runs *within* the target's patch tokens plus its global token: the fine intra-target temporal modeling PatchTST had, fully preserved, with no channel noise leaking in because only target tokens participate. The exogenous path embeds each side channel's whole look-back into one variate token with `DataEmbedding_inverted` — iTransformer's coarse, physically-coherent per-channel token. The cross: the target's *global token*, and only the global token, attends to the exogenous variate tokens via cross-attention. This is the asymmetry made architecture — cross-variate information enters through one narrow gate, so the patch tokens carrying the target's fine temporal detail are never overwritten by channel mixing, side-channel noise can only reach the target through a single learned bottleneck, and the cross-attention costs $O(\text{patch\_num}\cdot C)$ rather than iTransformer's $O(C^2)$ — linear in the side-channel count, spent only on informing the one target I predict. The encoder layer therefore carries *two* attentions: self-attention over the target's patch-plus-global tokens, then cross-attention from the global token into the exogenous tokens, then the position-wise feed-forward, stacked $e_{\text{layers}}$ deep. The head is the direct multi-step flatten-and-project that has worked at every rung: flatten the target's patch features — now informed by the exogenous side through the global token — and map to the full horizon.

Why the global token is the right gate, rather than letting every patch token cross-attend to the exogenous side: if all patch tokens attended to the channels, the cost would climb back toward the quadratic blow-up, and worse, every patch token's fine temporal content would be partially rewritten by a channel mixture at every layer — precisely how iTransformer lost the intra-series detail that cost it ETTh1. A single global token per channel is the minimal sufficient interface: one slot that summarizes "what does the target need from the rest of the panel," updated by cross-attention while the patch tokens are not, then folded back among the patch tokens by the feed-forward so the prediction can use the imported channel signal without any patch having been overwritten. Stacking is deliberate: within a layer the order is self-attention (target patches refine their own temporal representation), then cross-attention (the global token pulls in channel context), then feed-forward (that context diffuses back across the patches); the imported channel signal at layer $\ell$ feeds the target's self-attention at layer $\ell+1$, so over depth the temporal modeling and the channel context co-adapt — the target attends to its own patches *conditioned on* what the side channels say — yet because the gate stays one token wide, the cost stays linear and the patches are never directly mixed. This is why I keep the two-attention layer rather than concatenating target patches and exogenous tokens into one sequence and running plain self-attention (Crossformer's move), which both pays $O((\text{patch\_num}+C)^2)$ and lets channel noise hit the patches directly. In the multivariate (`features='M'`) setting, every channel is in turn the endogenous target while all channels serve as the exogenous pool, so the endogenous patching applies to all channels and each channel's global token reads the full variate set — the symmetric form of the same idea, wrapped in the per-window instance normalization every rung needed. The edit surface fixes $d_{\text{model}}=512$, $e_{\text{layers}}=2$, $\text{batch\_size}=32$, $\text{lr}=10^{-4}$ for all three, where the method's own script uses $d_{\text{model}}=256$, $\text{batch\_size}=4$, $e_{\text{layers}}=1$ on ETTh1/Weather — so the fixed 512 over-parameterizes the small datasets and the large batch is a regime mismatch; I read `patch_len` from the config (default 16 → 6 non-overlapping patches at $L=96$) and `use_norm` (default 1). The test is whether, under the identical fixed loop, the endogenous/exogenous split is the first to hold both levers — winning ECL without losing ETTh1/Weather.

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

    def forward(self, x):                                       # x: [B, n_vars, d_model, patch_num+1]
        return self.dropout(self.linear(self.flatten(x)))


class EnEmbedding(nn.Module):
    """Target -> patch tokens (+positions) and one learnable global token per channel."""
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                       # x: [B, n_vars, T]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)   # non-overlapping patches
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                                         # append global token
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        # self-attention within the target's patch+global tokens (intra-target temporal detail)
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        # only the global token cross-attends to the exogenous variate tokens (narrow gate)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    """TimeXer: endogenous patch self-attention + exogenous cross-attention through a global token."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, "use_norm", 1)
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))    # every channel endogenous
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)                 # full variate set exogenous

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                           # [B, nvars, d_model, patch_num+1]

        dec_out = self.head(enc_out).permute(0, 2, 1)                   # [B, pred_len, nvars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
