Let me start from what actually fails in front of me. The job is long-term multivariate forecasting: predict the whole horizon of every channel of a multivariate series. But fix attention on any one channel I'm predicting — call it the target — and the others are a pile of side channels that genuinely drive it. Electricity price is set by supply and demand, so wind-power and load forecasts are not decoration, they are the causal levers for *that* channel. From the target's point of view I want the side channels to *inform* its forecast and nothing else: I don't want the target's own representation paying to model interactions *into* channels it doesn't need, and I don't want their noise flowing back into it. That per-channel asymmetry is the whole thing, and none of the tools I have on the shelf is shaped around it.

So what is on the shelf. The linear forecasters, DLinear-style, take each series, split it into a moving-average trend and a remainder, run one linear map per piece straight from the `T`-step past to the `S`-step future, and sum. Channel-independent, no attention, and they hold their own against the temporal-attention Transformers. That tells me something useful right away: the *generation* step — turning a learned representation of the past into the whole future horizon — is well handled by a plain linear map. I should not be building a decoder that autoregresses the future. But DLinear is useless for my actual problem: being channel-independent and linear, it has no way to let a side channel touch the target. Exogenous information is simply invisible to it.

PatchTST is the obvious next candidate. It chops each series into subseries-level patches, embeds each patch as a token, and runs self-attention over the patch tokens, with one shared backbone per series — channel independence again. The patching is the good idea: a single time point is too local to be a meaningful unit, but a patch of `P` consecutive steps carries local temporal semantics, and attention over patches reads off the intra-series temporal dependency cleanly. For modeling the *target's own* temporal structure this is exactly what I want. But it is channel-independent by construction. It never mixes channels. So just like DLinear, the side channels can't reach the target — the cross-variate correlation I need is outside its hypothesis class entirely.

iTransformer pushes the other lever. It inverts what a token is: take a variate's whole look-back, all `T` points, and make *that* one token via a linear `R^T → R^D` map, so I get `C` tokens, one per channel, and the stock encoder's self-attention now runs *across* channels. The score between channel `i`'s token and channel `j`'s token is a clean cross-variate correlation — finally the side channels can influence the target. But look at what it cost. The target's whole series got crushed into a single coarse vector by one linear projection; all the fine intra-series temporal detail that PatchTST's patches preserved is gone. And it treats every channel as an equal token: self-attention over `C` tokens is `O(C²)`, and most of that compute is modeling interactions *into* channels I never predict, while letting their noise flow straight into the target's representation. For ECL that's 320 informer channels, for Traffic 861 — quadratic in exactly the dimension that is largest, spent on the part I care least about. Crossformer does mix both time and variate, but only by redesigning the attention layer and patching every channel, which is the same noise-from-irrelevant-channels problem plus new machinery.

Let me line up the failure so I can see the shape of it. PatchTST and DLinear: good intra-target temporal modeling, zero cross-channel influence. iTransformer and Crossformer: cross-channel influence, but the target loses its fine temporal detail, and they pay `O(C²)` treating informer channels as if they were targets. The two virtues live in two different camps, and each camp's virtue is the other's missing piece. I want both at once.

Here is the thing I keep circling back to: I have been assuming that whatever granularity I pick, I have to pick *one* and apply it to all channels — patch everything, or variate-token everything. But why? The target and the side channels play genuinely different roles. The target is the thing whose internal temporal wiggle I must nail to the value, step by step. The side channels are only there to nudge it — I care about *that* `z^{(i)}` shifts the target, not about the precise within-`z^{(i)}` micro-structure. Those are different demands, so maybe they deserve different granularities.

Push on that. For the target, I already know what I want: patches. Split `x ∈ R^T` into `N = ⌊T/P⌋ ` non-overlapping length-`P` patches `s_1, …, s_N`, embed each with a linear `R^P → R^D`, and self-attend over those `N` patch tokens to capture the intra-target temporal dependency. That is PatchTST's strength and I see no reason to give it up.

For the side channels, what granularity? If I patch them too, I get `C·N` patch tokens, and now I either self-attend over the whole pile — `O((C·N)²)`, and worse, the cross-attention between a target patch at time `t` and an exogenous patch at some other time is comparing time-misaligned chunks of mutually delayed processes, which is mostly noise — or I do something cleverer, but either way I have paid a fortune in tokens to model fine structure inside channels whose fine structure I don't care about. That is exactly the cost I'm trying to avoid. The cheaper, cleaner move is iTransformer's: embed each exogenous series as *one* variate token, a linear `R^{T_ex} → R^D` over the available exogenous window. That's `C` tokens, one per side channel, and it carries the "what is this channel doing overall" summary that is all I need from an informer. It also has a property I really want here: because the token is series-wise rather than timestamp-wise, the model is not forced to concatenate the target and exogenous values at matching instants. Different exogenous look-back lengths can be handled by the corresponding linear projector, and missing or low-quality values are at least a data-quality problem inside a whole-series summary rather than a hard per-timestep alignment requirement. The practical tensor implementation will still need a fixed configured input length and numeric values, but the representation is no longer built around step-by-step alignment. Variate token for the side channels it is.

Now I have a problem I created. The target lives as `N` patch tokens, each a slice of time. The side channels live as `C` variate tokens, each a whole series. These are different *kinds* of objects at different granularities. If I just throw all `N + C` of them into one self-attention, I'm back to cross-attending a target patch against a whole-series exogenous token — a fine-grained thing against a coarse thing — and the mismatch is exactly what introduces information misalignment and noise. I can't fuse them directly because they don't live at the same level.

So I need a meeting point: some representation of the target that is at the *same* granularity as a variate token, i.e. a whole-series summary of the target, that can talk to the exogenous variate tokens cleanly, and that also sits inside the target's own patch-level attention so whatever it learns from the side channels can propagate back down to the patches. A single node, per target series, that is series-level on one side and patch-aware on the other.

Where have I seen a single node whose only job is to aggregate a whole token sequence into one summary? The vision Transformer's class token. It's a learnable parameter — not a function of any one patch — that you prepend to the patch sequence, and through self-attention it is forced to collect information from the whole image and become the global descriptor. The mechanism is the point: because it shares the self-attention with the patches, it both *reads* from all of them (aggregates) and, on the next layer, is something the patches can *read back* from. That is precisely the dual role I need. So introduce a learnable global token `G` for the target series: one extra `D`-dimensional learnable vector, concatenated onto the `N` patch tokens.

Running `G` through the attentions, it does three jobs at once. First, fold `G` into the target's self-attention: attend over the concatenation `[P_en, G]`, all `N + 1` tokens together. Inside that single self-attention three interactions happen at once. Patch-to-patch: the ordinary intra-target temporal dependency, unchanged from PatchTST. Patch-to-global: each patch attends to `G`, so once `G` carries the exogenous influence, every patch can read it. Global-to-patch: `G` attends to all patches, aggregating the whole target series into itself — the class-token behavior. One stock self-attention layer over `[P_en, G]` gives me all three for free; I don't have to write them separately.

Second, the bridge to the side channels. I want exogenous information to flow into the target but not the reverse — in the single-endogenous setting I never predict the side channels, and I don't want the target's patches sending noise into them or paying to model that direction. That one-directional read is exactly what cross-attention does in multi-modal fusion: queries from one stream, keys and values from the other, so the query side reads from the value side without the value side reading back. So make the *global token the query* and the exogenous variate tokens the keys and values: `G` attends over `V_ex`, pulling in whatever the side channels say, and the side channels get nothing back. And because only `G` queries them — not all `N` patches — the cross-attention for one target is `O(C_ex)` in the number of exogenous variate tokens, not `O(N·C_ex)` and not the `O(C_ex²)` self-attention among informers would cost. For 320 or 861 informer channels that is the difference that matters, and it falls out naturally from routing everything through the single bridge node instead of letting every patch talk to every channel.

The path from exogenous tokens back to patches runs entirely through `G`: exogenous variate tokens → (cross-attention) → `G` → (the next self-attention over `[patches, G]`, via global-to-patch) → patches. In the current block the global token absorbs the external information after the endogenous self-attention has already run; the FFN is position-wise, so it does not mix that information into the patch tokens by itself. The distribution to patches happens when the next block's self-attention lets patches attend to the updated global token. If there is only one block, the flattened head still sees both the patch tokens and the exogenous-updated global token directly. That's the whole information pathway, and it only works because `G` lives in both the target-token sequence and the exogenous cross-attention.

Should the exogenous tokens attend among *themselves*? I have the option: instead of cross-attending `G` against `V_ex`, I could concatenate the exogenous variate tokens with the target tokens and self-attend over everything, which would add attention *within* the exogenous set. Reason about when that helps. Sometimes the interaction *between* side channels is itself informative — two informers that jointly predict the target better than either alone — and on a dataset with rich inter-channel structure (think dense, spatially-related sensors) that extra within-exogenous attention could pay off. But it is not universally valid: on many datasets the side channels don't usefully interact, and forcing self-attention among them spends `O(C²)` again and lets their mutual noise in. The robust default, the one that doesn't assume the informers help each other, is cross-attention only: `G` reads the side channels, the side channels don't read each other. So I keep cross-attention as the design and treat within-exogenous attention as the thing I'm deliberately *not* assuming.

Now the rest of the block. After the self-attention over `[P_en, G]` and the cross-attention that updates `G`, run the position-wise feed-forward on all the tokens — patches and global alike — the ordinary two-layer MLP-per-token, with residuals and layer norm around each sublayer exactly as a stock encoder block has them. Stack `L` of these. Notice what I have *not* done: I have not modified a single component. Self-attention, cross-attention, the FFN, layer norm — all stock, all proven across many fields. The only things that are new are the *granularity assignment* (patches for the target, variate tokens for the side channels) and the *bridge token* that lets two granularities meet. Everything else is the canonical Transformer, repurposed. That's deliberate: the components are not where the prior methods went wrong, the granularity choice was.

Patches — do they overlap? PatchTST allows a stride smaller than the patch length, so patches overlap. But overlapping patches smear neighboring windows together; on time series, which carry less local redundancy than images, that over-smoothing blurs the very temporal dependency I'm trying to read, and it costs more patches. Non-overlapping — stride equal to patch length — keeps each patch a clean, distinct window. So `step = patch_len`, no overlap. And the patch length itself cannot be tiny: a 2-point patch is barely better than a time point, while a 96-point patch would collapse back toward a single variate token. For a 96-step long-term look-back, `P=16` leaves six temporal tokens before the global token, which is a reasonable balance between local semantics and attention cost; for short hourly horizons, `P=24` naturally matches a daily block. Patches also need their order encoded, since attention is permutation-invariant and patch order along time is meaningful — so add a fixed sinusoidal positional embedding to the patch embeddings. The variate token gets no positional embedding: it's a whole series, there's no token-level order among the (single) target series summary to encode at this level.

The non-stationarity. Forecasting series drift in mean and scale across the window, and I want the network to work in a stationary, scale-free space. So wrap the whole thing in the per-series normalize/de-normalize used by the non-stationary forecasting line: before anything, take `means = x.mean(1, keepdim=True).detach()`, subtract it; `stdev = sqrt(var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)`, divide by it; run the model; then de-normalize each predicted channel with its own stored mean and stdev. Detaching the mean keeps the normalization statistics out of the gradient — they describe the input window, they aren't something to back-propagate through.

The head. The linear forecasters already told me the generation step is a linear map's job. So no decoder. After `L` blocks I have the endogenous output tokens — `N` patch tokens plus the global token, `N + 1` of them, each `D`-dimensional. Flatten them into one `D·(N+1)` vector and apply a single linear map to the `S`-step horizon. One forward pass produces the whole future; nothing autoregressive. Train with L2 (MSE) over the horizon.

Now the task in front of me is the full multivariate one: the harness hands me all channels stacked as `x_enc ∈ [B, T, enc_in]` and scores me on *every* channel's horizon, not one. So I want each channel to take its turn as the target while the others inform it — and I already argued nothing here is tied to a single target. The clean way to do that is channel independence on the endogenous side: treat every one of the `enc_in` channels as an endogenous series, patch-embed *all* of them at once into `[P_en, G]` tokens with shared weights — permute `x_enc` to `[B, enc_in, T]` and run the same patch+global embedding so each channel becomes its own `N+1`-token endogenous block. For the side information each channel reads, the canonical multivariate path variate-embeds the full multivariate look-back into a shared pool of variate tokens, folding the time-feature marks in as extra tokens on the same axis when present. That pool includes the channel's own variate token as well as the others; the single-target path is the one that explicitly removes the endogenous last channel from the exogenous input. In multivariate mode every channel's global token cross-attends against the shared pool, so the cross-attention cost is `O(enc_in · C_pool)` for the whole all-channel pass even though it is still `O(C_pool)` per target. Run the encoder, take each channel's `N+1` output tokens through the flatten-head to the horizon, and shape the output as `[B, pred_len, c_out]` with `c_out == enc_in` — a forecast for every channel at once.

Let me write the embeddings first, since they carry the two granularities. The endogenous embedding patchifies the target, projects each patch with a bias-free linear, adds the sinusoidal positions (the stock `PositionalEmbedding` on the shelf, unchanged), and concatenates the learnable global token:

```python
class EnEmbedding(nn.Module):
    """Target series -> N patch tokens (+ positions) and one learnable global token."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)   # R^P -> R^D
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))  # the bridge node
        self.position_embedding = PositionalEmbedding(d_model)             # patch order matters
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                   # x: [B, n_vars, T]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        # non-overlapping patches: step == patch_len
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, n_vars, N, P]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)              # [B*n_vars, N, D]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)                                       # append global token
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [B*n_vars, N+1, D]
        return self.dropout(x), n_vars
```

The exogenous embedding is the inverted one: permute to put each channel's whole configured window on the last axis and apply a linear over that length, optionally folding in the time marks as extra variate tokens on the same axis:

```python
class DataEmbedding_inverted(nn.Module):
    """Each series in the configured window -> one variate token: R^T -> R^D."""

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):                           # x: [B, T, C]
        x = x.permute(0, 2, 1)                              # [B, C, T]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)                              # [B, C(+marks), D]
```

Now the block. `FullAttention` and its multi-head `AttentionLayer` wrapper come straight off the shelf, unchanged, for both the self-attention and the cross-attention below. The new part is the encoder layer that ties together the self-attention over `[patches, global]`, the cross-attention that updates only the global token against the exogenous tokens, and the FFN:

```python
class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)     # position-wise FFN as 1x1 conv
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # x: [B*n_vars, N+1, D] = [patches, global]; cross: [B, C, D] = exogenous variate tokens
        B, L, D = cross.shape
        # self-attention over [patches, global]: patch-to-patch, patch-to-global, global-to-patch
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # pull the global token out; let ONLY it query the exogenous tokens (one-directional fusion)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # put the exogenous-updated global token back as the last token, then FFN over all tokens
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)
```

Stack `L` of these in a stock loop-plus-final-LayerNorm `Encoder`. The head then flattens the target's `N+1` output tokens and maps to the horizon — the linear generation step:

```python
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)          # D*(N+1) -> S
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                    # x: [B, n_vars, D, N+1]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)
```

And the whole model, wiring the two embeddings into the encoder and wrapping the per-series normalization. For the full multivariate task every input channel is endogenous in parallel, so `n_vars = enc_in`:

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        ...                                                                  # standard config bookkeeping
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == "MS" else configs.enc_in     # single-target vs multivariate

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                   configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([EncoderLayer(AttentionLayer(FullAttention(...), configs.d_model, configs.n_heads),
                                              AttentionLayer(FullAttention(...), configs.d_model, configs.n_heads),
                                              configs.d_model, configs.d_ff, dropout=configs.dropout,
                                              activation=configs.activation)
                                 for _ in range(configs.e_layers)],
                                norm_layer=torch.nn.LayerNorm(configs.d_model))

        self.head_nf = configs.d_model * (self.patch_num + 1)               # +1 for the global token
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # single-target convention: the final channel is endogenous, earlier channels are exogenous
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:                                                   # per-series stationarization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # every channel is endogenous (channel independence): [B, T, C] -> [B, C, T] -> patches + global
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        # the whole multivariate look-back as the shared pool of variate tokens
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                               # [B, n_vars, D, N+1]
        dec_out = self.head(enc_out).permute(0, 2, 1)                       # [B, S, n_vars]

        if self.use_norm:                                                   # de-normalize every channel
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out
    # forward() keeps the shell's own dispatch: forecast_multi under features="M", forecast otherwise.
```

What I like about how this landed is that the multivariate task didn't need a separate architecture: it's the single-target machinery run with every channel taking its turn as target, channel-independent self-attention over the `N+1` patch-plus-global tokens, and shared cross-attention weights against the pool. `forecast` predicts one designated channel from the rest; `forecast_multi` runs every channel in parallel against the shared full-channel pool — the batched version of the same primitive, with every channel scored.
