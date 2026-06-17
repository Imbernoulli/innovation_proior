Let me start from the thing that actually hurts. I have a multivariate series — `C` channels, `L` past steps — and I want the next `H` steps for every channel. On the datasets I care about `C` is not small: a traffic network has hundreds of sensors, some of the PEMS sets push toward nine hundred. And the data drifts. The level and scale of each sensor wander across the day and the week, sensors go dead and read flat, a meter drops out. So whatever I build has to live in a world where the per-channel statistics are non-stationary and individual channels are sometimes garbage.

Two facts are sitting in front of me and they point in opposite directions. The first: channels are correlated, and the correlation is real signal. Adjacent road sensors rise and fall together; if one sensor's recent history is informative about another, a model that refuses to look across channels is throwing away information that visibly helps. The second, which the field learned the hard way: the models that *do* mix channels — the per-timestamp-token Transformers, where you jam all `C` values at one time step into a token and attend along time — turned out to be fragile. A one-layer linear model beat them. The reading of that is non-stationarity: when you fit cross-channel structure, you fit the joint distribution of the channels, and that joint distribution is exactly what shifts between train and test, so you overfit to correlations that don't hold at test time. Higher capacity over the channels bought *less* robustness, not more.

So the field's escape was channel independence — process each channel's series alone, share weights, never let channels talk. Robust, cheap, strong. But it's strictly capped: by construction it cannot use the correlation, and on dense sensor grids the correlation is genuinely there. I want past that ceiling. The honest statement of my problem is: put *some* cross-channel information back, without re-importing the fragility that channel independence was invented to escape, and do it at a cost that survives `C` in the hundreds.

Let me first get the representation right, because the per-timestamp token is part of what went wrong. If a token is a timestamp, then mixing channels happens *inside every token*, structurally, before I've done anything — the entanglement is baked in. What if I flip it: make a token a *whole channel*. Take one channel's `L` past values and embed them — a single linear map `L → d` — into one `d`-dim vector that stands for that channel. Now I have `C` tokens, one per channel. This is the cleanest unit: each channel is a robust, self-contained thing (its own normalized history), and any cross-channel operation is now an explicit, *separable* step over the set of tokens, not something fused in by accident. Before this step I want each token's level and scale already neutralized, so I normalize per instance — subtract each channel's mean over the lookback, divide by its std — and I'll undo that on the prediction at the end. That strips the drifting part that causes most of the train/test mismatch and lets the tokens carry shape, not level.

Good. So the skeleton is: RevIN → embed each channel to a token → some layers that let the `C` tokens exchange information → a linear head `d → H` per token → denormalize. Everything except the middle is forced. The middle — how the `C` channel tokens talk — is the whole game.

The obvious thing, and what the inverted-token models already do, is self-attention across the channel tokens. Let every channel attend to every other channel; the attention map is a learned `C×C` correlation matrix; weight and aggregate. It works, it recovers correlation, and it keeps each channel as a unit so it's more robust than the old per-timestamp mixers. But I keep bouncing off two problems with it, and they're the two problems I started with.

First, cost. All-pairs attention is `O(C^2)` in time and memory. At `C` in the hundreds that's the binding constraint — it's the memory curve that blows up as channels grow, the reason these models don't scale to the big sensor networks. Second, and this is the one that nags more: attention's whole content is the *per-pair* weights. Channel `i` decides how much to take from channel `j` by an extracted `i`–`j` correlation. But the robustness literature is precisely about how those per-pair correlations are untrustworthy under non-stationarity — they're what overfits, they're what shifts. And if channel `j` is anomalous — a stuck sensor reading flat — every channel that attends to `j` ingests its garbage in proportion to a now-meaningless correlation weight. So attention is `O(C^2)` *and* it leans its entire interaction on the single quantity I have the most reason to distrust. That's not a small efficiency complaint; it's the same fragility coming back in through the interaction module.

Let me sit with what "interaction" has to mean. I want each channel to benefit from the others. But maybe it doesn't need to know *which specific other channel* helped it. The pairwise picture — `i` consults `j`, then `k`, then `l` — is one way to share information, but it's a *distributed* picture: everyone talks to everyone, `C^2` conversations, and each conversation's weight is a fragile correlation. There's another topology entirely. Instead of all-pairs, route everything through a *hub*: every channel contributes to one shared summary, and every channel reads back from that one summary. Star-shaped, not mesh. In networking that's `O(C)` links instead of `O(C^2)`; the same idea has been used to sparsify dense attention down to a relay node. The conversations collapse from `C^2` pairwise to `C` aggregate-in plus `C` redistribute-out.

And the more I think about it the more the hub isn't just a cost trick — it directly fixes the robustness complaint. A single summary aggregated over *all* channels is a *statistic*, and a statistic over many channels is far more stable than any one channel or any one pair. Averaging over channels is the robust operation; it's the opposite of trusting a brittle per-pair weight. If a sensor goes anomalous, it's one contributor out of hundreds to the summary, drowned out, rather than a correlation partner that some other channel ingests at full weight. So the hub gives me linear cost *and* it makes the interaction depend on an aggregate (robust) instead of on pairs (fragile). Both of the two problems, in one move.

So the shape is: aggregate the `C` channel tokens into one global summary — call it a *core* — then redistribute that core back to every channel and let each channel fuse the global core with its own local token. Each channel ends up seeing both its own robust per-channel representation and a global context distilled from everyone. The intuition that a stable global summary sharpens noisy local members is exactly the lever I want.

Now I have to actually define "aggregate the `C` tokens into one vector." This is a function of an unordered *set* of channel tokens. So what's the right form? Let me think about what invariances it must have. The output — the global summary — should not depend on the arbitrary index I assigned to each channel. If I relabel sensor 5 as sensor 200, the summary of the whole network shouldn't change. The aggregation must be permutation-invariant over the set of channels.

The set-function theory tells me exactly the two candidate shapes. One is the general continuous-function form `ρ(Σ_m λ_m φ(x_m))`: apply an inner map `φ` to each element, but sum with a *per-coordinate* weight `λ_m`, then an outer map `ρ`. The `λ_m` make the output depend on *which slot* each element sits in — it's permutation-*variant*. The other is the set form: `ρ(Σ_{x} φ(x))` — same `φ` on every element, summed with *equal* weight, then `ρ` — which is permutation-*invariant*, and which can approximate any continuous invariant set function. The two differ only in whether the summands carry coordinate-specific weights, i.e. whether the summary cares about channel ordering.

Which do I want? My instinct from the robustness story says invariant. A per-channel weight `λ_m` is a parameter glued to channel identity — it lets the model learn "channel 5 specifically tends to look like this." That's capacity spent on the channel index, and the channel index is meaningless under distribution shift: at test time the relationship between a channel's slot and its behavior is exactly what's drifted. Channel-specific parameters increase dependence on *which* channel and decrease dependence on the *history* the channel actually shows — which is the recipe for poor robustness on unseen series. Whereas the invariant form forces the model to summarize channels by *what they contain*, treating them as an exchangeable set, which is the robust thing and the thing channel-independence taught us to value. The characteristics of each channel's series should be enough to tell channels apart; I don't need to hand the model the index. So: the permutation-invariant `ρ(Σ φ(x))` form for the core.

Let me now turn that abstract form into an actual module and prune it. `φ` is "apply the same transform to each channel token" — an MLP from the token dimension `d` to some core dimension `d'`, shared across channels: `MLP₁ : R^d → R^d'`, two layers with a GELU in the middle. The `Σ` is "pool the `C` transformed tokens into one." The outer `ρ` is another transform on the pooled result. Then I redistribute: copy the core to all channels, concatenate each channel's own token with the core, and run a second MLP to fuse — `MLP₂ : R^{d+d'} → R^d` — with a residual back to the input token so the layer *refines* each channel rather than overwriting it.

Two simplifications fall out. First, the outer `ρ` after pooling — do I need it? Right after I pool I'm going to concatenate the core onto each token and push the result through `MLP₂`. That fusion MLP can absorb any outer nonlinearity on the core; a separate `ρ` would just be a redundant transform stacked before another transform. Drop it. So the core is `pool(MLP₁(tokens))`, no outer map, and the fusion MLP does the redistribution-side nonlinearity.

Second, the pooling — the `Σ` in the set form is a plain sum, which up to scale is mean pooling. Is mean the right reduction? Let me think about what mean vs. max each do to the core. Mean pooling averages all channels into the summary: maximally stable, but it washes everything to the middle and a strong, informative channel gets diluted by the crowd. Max pooling takes, per feature, the single most-activated channel: it preserves the salient signal but it's brittle — it bets the whole summary coordinate on one channel, and if that channel is the anomalous one, the core inherits its garbage. On different datasets one wins and the other wins; neither dominates. I don't want to hard-pick.

So I want a pooling that interpolates between "average everyone" and "take the strongest," and ideally one that also regularizes, because the core is a single shared object that every channel reads from — if it overfits, everyone overfits together. Here's an operator that does exactly that. Take the activations to be pooled, `A`, of shape `C × d'` — channel by core-feature. For each core-feature `j`, normalize the activations *across channels* into a distribution with a softmax: `p_{ij} = e^{A_{ij}} / Σ_{k=1}^{C} e^{A_{kj}}`. Now treat `p_{·j}` as a multinomial over channels. During training, for each feature `j`, *sample* a channel `c ∼ Multinomial(p_{·j})` and set the core coordinate to that channel's value, `o_j = A_{cj}`. At test time, don't sample — take the expectation under that same distribution, `o_j = Σ_{i=1}^{C} p_{ij} A_{ij}`.

Look at what this is doing. The sampling probability for a channel is monotone in its activation, so high-activation channels are picked often (the max-pooling tendency) but low-activation channels are still occasionally picked (the mean-pooling tendency) — it lives between max and mean by construction. The training-time sampling is stochastic, so it injects noise into the core every step, which regularizes the shared summary instead of letting it lock onto whichever channels happened to be loud on the training set. And the test-time expectation `Σ_i p_{ij} A_{ij}` is a clean, deterministic, magnitude-weighted average — it's the mean of the very distribution we sampled from in training, so train and test are consistent, and it's exactly the "soft, weighted aggregate over channels" that the robustness argument wanted: a statistic over all channels, with weight concentrated on the salient ones but spread so no single channel dominates and no single dead channel can hijack the core. Softmax over the *channel* axis is the right axis — I'm normalizing across the things I'm pooling, the channels, for each feature independently.

With a batch dimension this indexing has to be careful. `A` is `[B, C, d']`. After the channel-softmax I permute it to `[B, d', C]` and flatten it to `[B*d', C]`, because I need exactly one sampled channel for each pair `(batch item, core feature)`. `torch.multinomial` then gives `[B*d', 1]`; reshaping to `[B, 1, d']` makes the sampled channel index line up with the channel dimension of `A`, so `gather(A, dim=1, index)` returns one core vector `[B, 1, d']`, and repeating it over `C` channels gives `[B, C, d']` for redistribution.

Let me also sanity-check the cost, because cost was half the point. `MLP₁` is a per-channel `R^d → R^d` map (taking `d' = d`), applied to `C` channels: `O(C d^2)`. The pooling is a softmax along the channel axis per feature: `O(C d)`. Concatenate and `MLP₂` on the `C` fused tokens: `O(C d^2)`. The head `d → H` per channel: `O(C d H)`. Everything is linear in `C`. No `C×C` matrix is ever formed — the only place all channels meet is the pooled core, which is `O(C)` to build and `O(C)` to broadcast back. With `d` a fixed model constant, the interaction is `O(C)` in channels, against attention's `O(C^2)`. That's the scaling I needed, and it came for free from routing through a hub instead of all-pairs.

Now assemble one interaction layer. Input is the channel tokens `S ∈ R^{C×d}`. Compute the core: `o = StochPool(MLP₁(S))`, with `o ∈ R^{d'}`. Redistribute: repeat `o` to all `C` channels and concatenate onto each token, giving `F ∈ R^{C×(d+d')}`. Fuse: `S' = MLP₂(F) + S`, the residual so the layer adds global context to each channel's own representation instead of replacing it. Then, as ordinary transformer-block hygiene around this, a position-wise feed-forward network applied per channel token (a wider GELU MLP) with its own residual, and LayerNorm after each sublayer to keep the stack stable as I add layers. Stack `N` of these. Aggregate, redistribute, through a star-shaped core — the interaction is centralized, not distributed.

Wrap it all in the skeleton I already fixed: RevIN-normalize the input, embed each channel's lookback `L → d` into a token, run the `N` star-layers, a final LayerNorm, then the linear head `d → H` per channel, transpose back to `[H, C]`, and RevIN-denormalize the prediction. Train with MSE under Adam.

Let me write it as the module. The core/fuse MLPs I can spell out as their constituent linear layers (`gen1..gen4` with GELU between) — `MLP₁ = gen1(d→d) ∘ GELU ∘ gen2(d→d')`, `MLP₂ = gen3(d+d'→d) ∘ GELU ∘ gen4(d→d)` — which is exactly the two-layer-with-GELU shape I argued for.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAR(nn.Module):
    """STar Aggregate-Redistribute: centralized cross-channel interaction.
    Aggregate all C channel tokens into one core via stochastic pooling over
    the channel axis, then redistribute the core back and fuse per channel.
    O(C), not O(C^2): no channel-by-channel comparison is ever formed."""

    def __init__(self, d_series, d_core):
        super().__init__()
        # MLP_1 : R^d -> R^{d'}  (phi, shared across channels)
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        # MLP_2 : R^{d+d'} -> R^d  (fusion on the redistribution side)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x, *args, **kwargs):      # x: [B, C, d], one token per channel
        B, C, d = x.shape

        # phi: per-channel transform to the core feature space
        a = self.gen2(F.gelu(self.gen1(x)))     # A: [B, C, d'], "activations to pool"

        # aggregate: stochastic pooling over the CHANNEL axis -> one core, broadcast back
        if self.training:
            # p_{ij} = softmax over channels of A_{ij}; sample one channel per feature
            ratio = F.softmax(a, dim=1)                 # [B, C, d']
            ratio = ratio.permute(0, 2, 1).reshape(-1, C)   # [B*d', C]
            idx = torch.multinomial(ratio, 1)               # [B*d', 1]
            idx = idx.view(B, -1, 1).permute(0, 2, 1)       # [B, 1, d']
            core = torch.gather(a, 1, idx).repeat(1, C, 1)  # o_j = A_{cj}, broadcast
        else:
            # test-time expectation: o_j = sum_i p_{ij} A_{ij}
            p = F.softmax(a, dim=1)
            core = (a * p).sum(dim=1, keepdim=True).repeat(1, C, 1)

        # redistribute: concat core onto each channel token, fuse, residual
        f = torch.cat([x, core], dim=-1)        # [B, C, d+d']
        out = self.gen4(F.gelu(self.gen3(f)))
        return out, None
```

And the full forecaster, filling the skeleton:

```python
class EncoderLayer(nn.Module):
    """One layer: STAR in the interaction slot, then a per-channel FFN, each
    with residual, dropout, and post-LayerNorm."""

    def __init__(self, interaction, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.interaction = interaction
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, _ = self.interaction(x, x, x)    # STAR ignores the extra attention-style args
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SOFTS(nn.Module):
    """Series-cOre Fused Time Series forecaster: each channel's whole lookback
    is one token; N STAR layers exchange channel information through a core;
    a linear head produces the horizon. RevIN guards against distribution shift."""

    def __init__(self, input_len, output_len, num_features,
                 hidden_size=512, core_size=128, num_layers=2, dropout=0.05):
        super().__init__()
        self.eps = 1e-5
        self.embed = nn.Linear(input_len, hidden_size)      # L -> d, per channel
        self.layers = nn.ModuleList([
            EncoderLayer(STAR(hidden_size, core_size), hidden_size, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_len)      # d -> H, per channel

    def forward(self, inputs):                  # inputs: [B, L, C]
        # RevIN: remove per-channel level/scale over the lookback
        mean = inputs.mean(dim=1, keepdim=True).detach()
        x = inputs - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x = x / stdev

        N = x.size(-1)
        h = self.embed(x.transpose(1, 2))       # [B, C, d]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.head(h).transpose(1, 2)[:, :, :N]   # [B, H, C]

        # RevIN inverse on the prediction
        return pred * stdev + mean
```

The chain, start to end: channels are correlated so I must look across them, but cross-channel structure overfits under non-stationarity and all-pairs attention costs `O(C^2)` and leans on the fragile per-pair correlation. Making each channel a token separates the interaction into its own step; routing that interaction through a single aggregated *core* instead of all pairs cuts cost to `O(C)` and replaces brittle pairwise weights with a robust statistic over all channels; the aggregation must be permutation-invariant over channels, so it's a shared per-channel map then a pool, and the pool is stochastic — sampled in training to regularize the shared core, expectation at test for a magnitude-weighted average that lives between mean and max; redistributing the core back and fusing it with each channel's own token, with residuals, lets a stable global summary sharpen each local channel without erasing it; RevIN brackets the whole thing to absorb the drift. Correlation, robustness, and linear scaling, together.
