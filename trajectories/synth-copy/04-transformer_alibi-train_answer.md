NoPE broke the wall. Deleting the positional code lifted OOD off zero on every variant — `exact_match_ood` went from sinusoidal's $0.0/0.0/0.001$ to $0.297/0.550/0.403$ and OOD token accuracy from $0.071/0.066/0.031$ to $0.433/0.757/0.434$, past the LSTM and past everything — while ID held perfect and the geometric-mean score jumped from $0.50$ to $0.706$. So absoluteness *was* the disease. But the per-variant OOD numbers expose what NoPE leaves on the table: `repeat` extrapolates best ($0.550$ exact), while `delim` and `reverse` are the weaker pair ($0.297/0.403$ exact, token accuracies suggestively symmetric around $0.43$). NoPE recovers position by *construction* — the layer-one $1/t$ count anchored at `BOS`, re-coded by the MLP into a relative score — but that recovered code is something SGD has to discover and then keep stable across 30–40 unseen positions with no inductive bias pushing it toward the right shape. It works, but it is *learned from nothing*, which is exactly the kind of thing that holds at length 25 and frays at length 40. The next move is the obvious one: stop *hoping* the model induces a good relative code and *give* it one — a relative bias whose value at an unseen distance is the obvious continuation of its value at seen distances, so there is no out-of-distribution regime to fray in.

I have to choose that relative scheme carefully, because NoPE already taught me the failure modes of the explicit family. A learned relative-bias table (T5-style) fixes a bucketing and cutoff, and its far buckets are trained only on whatever far distances appeared in $[0,20)$, so even bucketing only partly defines behavior out of range — and it adds parameters and a gather. A rotation scheme is relative-by-construction but injects per-layer rotations that cost compute, and its frequency schedule is a fixed prescription too. What I want is the property the relative family has — position lives in the *score*, depends only on $i-j$, never enters the values, reasserts at every layer — but with a bias-versus-distance function so simple it needs no parameters, no gather, and is defined *smoothly for every distance*, including ones never trained on. The disease with sinusoidal was that the signal past the training length was a novel high-dimensional pattern; the cure is a position signal whose value at distance $d$ is the obvious continuation of its values at small $d$, with no learned dial that could be miscalibrated out of range. The simplest such function of distance is a straight line.

So I propose **ALiBi — Attention with Linear Biases** (Press, Smith, Lewis 2022): add to each query–key score, before the softmax, a bias that is *linear* in the relative distance. I want a recency bias — distant keys matter less than nearby ones, all else equal — and crucially I want that preference to have the *same shape* at every distance, so distance 40 is just "distance 4 but more so," never a regime the model has not conceptually seen. So the bias is *more negative* the farther the key is from the query. For a causal query at position $i$ over keys $j \le i$, the relative distance is $i-j \ge 0$, and I subtract a penalty proportional to it:
$$\text{score}_{ij} = \frac{q_i^\top k_j}{\sqrt{\text{head\_dim}}} - m\,(i - j),$$
so the key right next to the query (distance 0) gets penalty 0, one step back gets $-m$, two steps back $-2m$, linearly, with a per-head slope $m > 0$. No embedding added anywhere, no learned bias, no bucket gather — and it is defined at any distance, because at distance 40 the penalty is just $40m$, the same line rather than a new absolute vector. This directly attacks both the rung-two failure and the rung-three softness: the thing the model interprets is a scalar that grows linearly and predictably, not a phase pattern that goes novel (sinusoidal) and not a code SGD had to induce from scratch (NoPE).

This keeps every structural property NoPE showed me matters. Position lives in the scores, not the values — the bias is added to the scores and the values are untouched, so the attention output carries no absolute-position component, exactly like the relative code NoPE synthesized in its later layers. It injects at *every* layer — every attention sublayer adds the same bias, so position reasserts itself throughout, where the sinusoidal add happened only once at the bottom. And it depends only on $i-j$, the quantity that recurs at unseen lengths. So this is the explicit, well-shaped version of the thing NoPE was groping toward — but instead of a single learned recency shape I get a *spread* of them, because the slope is per-head.

The slope needs care. One slope for all heads would force every head to the same recency preference, which is clearly wrong — some heads should look locally, some should keep a long view, and for `reverse` in particular I need at least one head that can reach far back into the input without being crushed by the penalty. So $m$ is per-head, and the $H$ slopes should span a range of recency scales: a head with large $m$ is penalized hard for distance and becomes local; a head with tiny $m$ barely notices distance and keeps a long-range view. The long-range heads are the precious ones, so I want more resolution near zero — slopes bunched toward 0, spread toward 1 — which is what a geometric sequence gives (equal ratios = equal log-spacing, dense near the small end). The clean choice is start = ratio = $2^{-8/n}$, giving $2^{-8/n}, 2^{-16/n}, \dots, 2^{-8}$; for the power-of-two case it reduces to $1/2, 1/4, \dots, 1/256$ at 8 heads. This task has 4 heads, also a power of two, so the formula applies directly: $2^{-8/4} = 2^{-2} = 1/4$, giving slopes $1/4, 1/16, 1/64, 1/256$ — four distinct recency scales from a fairly local head down to one ($1/256$) that barely penalizes distance, which is the long-range head `reverse` needs. (For a non-power-of-two head count the scaffold falls back to the closest-power-of-two set plus interleaved extras, but that path is unused here.)

Should the slopes be learned? My instinct is to let gradient descent find them, but for *extrapolation* that is the wrong instinct, and NoPE is the cautionary tale: learned slopes would be tuned to the distances seen in $[0,20)$, with no training signal about how the slope should behave at the far distances that only appear at evaluation, so a learned slope could overfit the in-range recency statistics and generalize *worse* out of range — the same "learned from nothing reliable past the training length" weakness I am trying to fix. A fixed geometric set is the same line everywhere and cannot be miscalibrated to the training length, so I fix the slopes before training; they live in a parameter-free buffer wrapped in a tiny `nn.Module` so they follow the model to the GPU, carry no gradient, and are registered via `scheme.extra_modules`.

One scaling subtlety is easy to get wrong. Standard attention divides the scores by $\sqrt{\text{head\_dim}}$ to tame the variance of a random dot product. My bias is not a random dot product — it is a deterministic geometric penalty I choose directly in score-space. If I also divided it by $\sqrt{\text{head\_dim}}$ I would just be rescaling slopes I already choose freely, coupling my slope choice to the head dimension for no reason. So the linear bias is added *after* the $\sqrt{\text{head\_dim}}$ scaling, not inside it — which is automatic in this harness, since `CausalSelfAttention` computes $q\cdot k/\sqrt{\text{head\_dim}}$ first and then adds `scheme.attn_bias(T, …)`.

Now the concrete fill, where the harness version differs from the canonical implementation. The `attn_bias` hook must return a $[H,T,T]$ (or $[1,T,T]$) tensor added to the scores *before* the causal mask, which the loop applies itself right after. The canonical ALiBi implementation exploits softmax's per-row shift-invariance to replace the honest staircase $m\cdot[-(i-1),\dots,-1,0]$ with a single broadcast row $m\cdot[0,1,\dots,T-1]$, since the two differ only by a per-row constant that softmax kills — a cheap trick that works *because* the bias is linear. This harness does **not** take that shortcut: it builds the full signed relative-distance matrix `rel = idx[None, :] − idx[:, None]` (so $\text{rel}_{ij} = j - i$, negative below the diagonal where $j < i$) and sets `bias = slope · rel`, i.e. $\text{bias}_{ij} = m(j-i) = -m(i-j)$ — the honest staircase, more negative for keys farther back, with no shift-invariance optimization. Above the diagonal `rel` is positive, but those entries are immediately overwritten by the loop's $-\infty$ causal mask before the softmax, so they never matter and I need not mask them myself. The result is shaped $[H,T,T]$ (one slope per head), the loop broadcasts across the batch and adds the causal mask on top, and the values and token embeddings stay completely position-free. `build_positional_scheme` returns the scheme with only `attn_bias` set (`token_embedding_extra` and `rotary` are `None`); `build_model` returns the plain `SeqModel(use_lstm=False)`. Neither the omitted softmax-shift construction nor the separate (rather than folded) bias-and-mask adds changes the algorithm — the attention weights are identical to canonical ALiBi, only the arithmetic path differs.

My falsifiable expectations against the NoPE numbers. In-distribution should stay perfect on all three variants — the bias is negligible at short distances. The bet is OOD, and the prediction is specific: ALiBi should beat NoPE on OOD exact match where NoPE's *learned* relative code was softest — I expect the biggest lift on `delim` (NoPE only $0.297$) and a solid lift on `repeat`, because the linear recency bias is precisely the well-shaped relative prior NoPE had to induce from scratch, defined identically at length 40 as at length 15. The risk I have to name is `reverse`: ALiBi's recency bias is a *monotone* preference for nearby tokens, but `reverse` requires attending from the end of the output all the way back to the *start* of the input — the longest possible offset — exactly the direction the recency penalty fights. So `reverse` is where the built-in prior could backfire, the expected weakest variant, possibly at or below NoPE there, with the long-range head ($\text{slope} = 1/256$) doing the heavy lifting if it works at all. The honest claim is therefore aggregate, not uniform: ALiBi should lift the geometric-mean score above NoPE's $0.706$ by winning clearly on `delim`/`repeat` even if it pays a little on `reverse` — a well-shaped, parameter-free relative prior beating a relative code learned from nothing.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- step 4: ALiBi linear distance bias
def _alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes from Press et al. 2022, generalized to non-pow2 n_heads."""
    def power_of_2_slopes(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        return [start ** (i + 1) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return torch.tensor(power_of_2_slopes(n_heads), dtype=torch.float32)
    closest = 2 ** math.floor(math.log2(n_heads))
    base = power_of_2_slopes(closest)
    extra = power_of_2_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(base + extra, dtype=torch.float32)


def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """ALiBi: additive per-head linear distance bias on attention scores."""
    slopes = _alibi_slopes(config.n_heads)
    # Cache slopes as a parameter-free buffer wrapped in a Module so it
    # follows the model device. We rebuild the bias matrix on demand.
    container = nn.Module()
    container.register_buffer("slopes", slopes, persistent=False)
    extras = nn.ModuleList([container])

    def attn_bias(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Relative-distance matrix: bias_{i,j} = -slope * (i - j) for i >= j.
        # Causal mask still applies on top of this.
        idx = torch.arange(T, device=device)
        rel = idx[None, :] - idx[:, None]  # negative below the diagonal
        rel = rel.to(dtype)
        s = container.slopes.to(device=device, dtype=dtype)
        # ALiBi only meaningfully shapes scores at positions j <= i, where
        # rel <= 0, giving bias = slope * rel (more negative for distant).
        bias = s[:, None, None] * rel[None, :, :]
        return bias  # [H, T, T]

    return PositionalScheme(
        name="alibi",
        token_embedding_extra=None,
        attn_bias=attn_bias,
        rotary=None,
        extra_modules=extras,
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Decoder-only Transformer with ALiBi (Press et al. 2022)."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```
