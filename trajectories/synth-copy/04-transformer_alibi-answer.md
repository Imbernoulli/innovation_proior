**Problem (from rung 3).** NoPE broke the OOD wall (`exact_match_ood` `0.297 / 0.550 / 0.403`, geomean
`0.706`) by removing the absolute prior, but its relative code is *learned from nothing* and weakest on
`delim`/`reverse` — the kind of thing that holds at length 25 and frays at length 40. Give the model a
relative prior with the right shape instead of hoping SGD induces one.

**Key idea.** ALiBi (Press, Smith, Lewis 2022): add to each query–key score, before softmax, a bias
*linear* in the relative distance, `−m·(i − j)` for a per-head slope `m > 0` — more negative the farther
back the key. No embedding, no learned bias, no gather. Defined at any distance (distance 40 is just
`40·m`, the same line), so there is no out-of-distribution regime to fray in.

**Why.** Position lives in the score, not the values, depends only on `i − j`, and reasserts at every
layer — the structural shape NoPE's later layers groped toward, made explicit. Per-head geometric slopes
(`2^{−8/n}`: for `n = 4` heads, `1/4, 1/16, 1/64, 1/256`) give a spread of recency scales, densest near
zero where the long-range heads live. Slopes are *fixed*, not learned, because learned slopes would
overfit the in-range distances and generalise worse out of range — the exact NoPE weakness. The bias is
added *after* `√head_dim` scaling (it is a chosen deterministic penalty, not a variance-prone dot product).
Risk: the recency bias is monotone, which fights `reverse` (attend from output end back to input start),
so `reverse` is the expected weak variant.

**Scaffold edit / differences from the canonical recipe.** Fill `build_positional_scheme` to compute the
per-head slopes, cache them as a parameter-free buffer in a tiny `nn.Module` (registered via
`scheme.extra_modules`), and return only the `attn_bias` hook. The harness builds the **full** signed
relative-distance matrix `rel = idx[None,:] − idx[:,None]`, `bias = slope·rel`, shaped `[H,T,T]` — it does
**not** use the canonical softmax-shift cheap row-broadcast, and does not fold the bias into the causal
mask. The loop applies the lower-triangular `-inf` mask *after* the bias, so the positive above-diagonal
entries are overwritten and need no manual masking; the attention weights are identical to canonical
ALiBi. `build_model` returns the plain `SeqModel(use_lstm=False)`; values and embeddings stay
position-free.

**Hyperparameters.** Per-head slopes `2^{−8/n}` (n = 4 → `1/4, 1/16, 1/64, 1/256`), fixed (no gradient).
All training settings fixed by the loop.

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
