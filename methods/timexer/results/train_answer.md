The job is long-term multivariate forecasting: given a fixed look-back window, predict the whole horizon $\hat x_{T+1:T+S}$ of every channel of a multivariate series. But the moment I fix attention on any single channel I am predicting — call it the endogenous target — the other channels stop being peers and become side information: observed series that genuinely *drive* the target and should sharpen its forecast, even though, on its own turn, the target is never helped by being lumped in with the rest. Electricity price is set by supply and demand, so wind-power and load forecasts are the causal levers for *that* channel; a building's CO₂ is driven by the surrounding meteorology. From the target's point of view I want the side channels to *inform* it and nothing more — I don't want the target's representation paying to model interactions *into* channels it never predicts, and I don't want their noise flowing back. That per-channel asymmetry is the whole problem, and nothing on the shelf is shaped around it.

The tools I have split cleanly into two camps, and each camp's virtue is exactly the other's missing piece. The linear forecasters, DLinear-style, decompose each series into a moving-average trend and a remainder and run one linear map per piece straight from the $T$-step past to the $S$-step future; they hold their own against temporal-attention Transformers, which tells me something I'll keep: the *generation* step — turning a learned representation of the past into the whole horizon — is well handled by a plain linear map, so I should not build an autoregressive decoder. But DLinear is channel-independent and linear, so a side channel can never touch the target; exogenous information is invisible to it. PatchTST is the next rung: it chops each series into subseries-level patches, embeds each patch as a token, and self-attends over the patch tokens with one shared channel-independent backbone. The patching is the good idea — a single instant is too local to be a meaningful unit, but a patch of $P$ consecutive steps carries local temporal semantics, and attention over patches reads off the intra-target temporal dependency cleanly. Yet it is channel-independent by construction; cross-variate correlation is outside its hypothesis class. iTransformer pushes the opposite lever: it inverts what a token is, embedding a whole look-back of $T$ points into one variate token via a linear $\mathbb{R}^T \to \mathbb{R}^D$ map, so there are $C$ tokens and self-attention finally runs *across* channels — the side channels can influence the target at last. But the target's whole series has been crushed into one coarse vector, losing the fine temporal detail PatchTST preserved; and it treats every channel as an equal token, so self-attention is $O(C^2)$, most of it spent modeling interactions into channels I never predict while their noise flows straight into the target. For ECL that is 320 informer channels worth of quadratic compute in exactly the largest dimension. Crossformer mixes both time and variate but only by redesigning the attention layer and patching every channel — the same noise-from-irrelevant-channels cost plus new machinery. So: PatchTST and DLinear give intra-target temporal detail and zero cross-channel influence; iTransformer and Crossformer give cross-channel influence but lose the target's detail and pay $O(C^2)$. I want both at once.

I propose TimeXer. The key realization is that I have been assuming I must pick *one* granularity and apply it to every channel — patch everything, or variate-token everything — but the target and the side channels play genuinely different roles and so deserve different granularities. For the endogenous target I keep patches: split $x \in \mathbb{R}^T$ into $N = \lfloor T/P \rfloor$ non-overlapping length-$P$ patches, embed each with a bias-free linear $\mathbb{R}^P \to \mathbb{R}^D$ plus a sinusoidal positional embedding, and self-attend over the $N$ patch tokens to capture intra-target temporal structure. Patches are non-overlapping — stride equal to patch length — because overlapping patches smear neighboring windows together, and time series carry less local redundancy than images, so that over-smoothing blurs the very dependency I want to read and costs more tokens. The patch length must sit between the extremes: a 2-point patch is barely better than a time point, a 96-point patch collapses back toward a single variate token; for a 96-step look-back $P=16$ leaves six temporal tokens, a reasonable balance. The patches need positions because attention is permutation-invariant and patch order along time is meaningful. For the side channels I take iTransformer's move and embed each whole series into *one* variate token via a linear over its window, $\mathbb{R}^{T_\text{ex}} \to \mathbb{R}^D$. That is $C$ tokens, one per side channel, carrying the "what is this channel doing overall" summary that is all an informer owes me — and because the token is series-wise rather than timestamp-wise, the representation is no longer built around step-by-step alignment, so different exogenous lengths and missing values are softened rather than being a hard per-timestep concatenation requirement. The variate token gets no positional embedding; there is no token-level order among whole-series summaries to encode.

That decision creates the problem I have to solve: the target now lives as $N$ patch tokens, each a slice of time, while the side channels live as $C$ variate tokens, each a whole series — different *kinds* of object at different granularities. Throwing all $N+C$ into one self-attention would cross-attend a fine-grained target patch against a coarse whole-series token, and that mismatch is precisely the source of information misalignment and noise. I need a meeting point: a representation of the target that is itself at variate-token granularity — a whole-series summary — so it can talk to the exogenous tokens cleanly, yet that also sits inside the target's own patch-level attention so whatever it learns can propagate back down to the patches. This is exactly the vision-Transformer class token: a single learnable $\mathbb{R}^D$ parameter, not a function of any patch, that through shared self-attention is forced to aggregate the whole sequence and becomes a node the patches can read back from. So I introduce one learnable global token $G$ per endogenous series, concatenated onto the $N$ patch tokens, as the bridge.

Running $G$ through the attentions shows it does the three jobs at once. First, fold $G$ into the endogenous self-attention over the concatenation $[P_\text{en}, G]$, all $N+1$ tokens together. One stock self-attention layer then realizes, for free, patch-to-patch (the ordinary intra-target dependency), patch-to-global (each patch can read $G$ once $G$ carries the exogenous influence), and global-to-patch ($G$ aggregates the whole series into itself, the class-token behavior). Second, the bridge to the side channels must be one-directional — exogenous information into the target, nothing back — which is exactly what cross-attention gives: queries from one stream, keys and values from the other. So I make the *global token the sole query* and the exogenous variate tokens the keys and values: $G$ attends over $V_\text{ex}$, pulls in what the side channels say, and they get nothing back and are never updated. Because only $G$ queries them — not all $N$ patches — the cross-attention per target is $O(C_\text{ex})$ in the number of exogenous tokens, not $O(N \cdot C_\text{ex})$ and not the $O(C_\text{ex}^2)$ a self-attention among informers would cost; for hundreds of informer channels that is the difference that matters, and it falls out of routing everything through the single bridge node. The full external path is therefore: exogenous tokens $\to$ (cross-attention) $\to G \to$ (the next block's self-attention over $[P_\text{en}, G]$, via global-to-patch) $\to$ patches; with a single block the flatten head still sees the exogenous-updated $G$ directly. I deliberately keep cross-attention rather than self-attending the exogenous tokens among themselves: inter-informer interaction sometimes helps, but assuming it costs $O(C^2)$ again and admits their mutual noise, so the robust default does not assume the informers help each other. Per block $l$, with patch tokens $P^l_\text{en}$, global token $G^l_\text{en}$, exogenous tokens $V_\text{ex}$, the computation is

$$[\hat P^l_\text{en}, \hat G^l_\text{en}] = \mathrm{LayerNorm}\big([P^l_\text{en}, G^l_\text{en}] + \mathrm{SelfAttn}([P^l_\text{en}, G^l_\text{en}])\big),$$
$$\hat G^l_\text{en} = \mathrm{LayerNorm}\big(\hat G^l_\text{en} + \mathrm{CrossAttn}(\hat G^l_\text{en}, V_\text{ex})\big),$$
$$[P^{l+1}_\text{en}, G^{l+1}_\text{en}] = \mathrm{LayerNorm}\big([\hat P, \hat G] + \mathrm{FFN}([\hat P, \hat G])\big),$$

stacked for $L$ blocks with a position-wise two-layer FFN (implemented as $1\times1$ convolutions) and residual-plus-LayerNorm around each sublayer. What makes this work is that not a single Transformer component is modified — self-attention, cross-attention, FFN, layer norm are all stock and proven; the only things new are the *granularity assignment* (patches for the target, variate tokens for the side channels) and the *bridge token* that lets the two granularities meet. That is deliberate: the components were never where the prior methods went wrong, the single-granularity choice was.

Two more pieces close the design. Forecasting series drift in mean and scale, so I wrap the model in a per-series normalize/de-normalize: subtract $\text{means} = x.\text{mean}$ over time (detached, since the statistics describe the input window and are not something to back-propagate through), divide by $\text{stdev} = \sqrt{\mathrm{var}(x) + 10^{-5}}$, run the model, and de-normalize each predicted channel with its own stored mean and stdev. And since the linear forecasters already showed generation is a linear map's job, there is no decoder: after $L$ blocks I flatten each target's $N+1$ output tokens into one $D\cdot(N+1)$ vector and apply a single linear map to the $S$-step horizon, trained with L2. The full multivariate task needs no separate architecture — it is this same primitive run in parallel via channel independence: permute $x_\text{enc} \in [B,T,\text{enc\_in}]$ to $[B,\text{enc\_in},T]$ so every channel becomes its own $[P_\text{en}, G]$ block with shared weights, embed the full multivariate look-back into a shared pool of variate tokens (folding time marks in as extra tokens), and let every channel's global token cross-attend that pool. The self-attention stays $O((T/P+1)^2)$ per channel and the all-channel cross-attention is $O(\text{enc\_in}\cdot C_\text{pool})$, with every channel predicted at once. The `forecast` path predicts the last channel from the earlier ones (the conceptual single-target primitive); `forecast_multi` predicts all channels against the shared full-channel pool.

```python
import math
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

    def forward(self, x):                                    # x: [B, n_vars, D, N+1]
        return self.dropout(self.linear(self.flatten(x)))


class EnEmbedding(nn.Module):
    """Target -> N patch tokens (+ positions) and one learnable global token."""
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                    # x: [B, n_vars, T]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)   # non-overlapping
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                                        # append global token
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
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)                                  # the global token
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
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)
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
                )
                for _ in range(configs.e_layers)
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

        # single-target path: last channel is endogenous, earlier channels are exogenous
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                                 # [B, n_vars, D, N+1]
        dec_out = self.head(enc_out).permute(0, 2, 1)                         # [B, S, n_vars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # every channel is endogenous (channel independence); whole look-back as variate-token pool
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                                 # [B, n_vars, D, N+1]
        dec_out = self.head(enc_out).permute(0, 2, 1)                         # [B, S, n_vars]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            if self.features == "M":
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
```
