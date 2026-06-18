**Problem (from rung 2).** Sinusoidal absolute PE is perfect in-distribution but its OOD token accuracy
(`0.071 / 0.066 / 0.031`) is *worse* than the recurrent baseline's (`0.371 / 0.207 / 0.180`), with OOD
exact match ≈ 0. The best absolute code is the worst extrapolator: the disease is absolute position
itself, because the index keyed on over `[0, 20)` does not recur over `[21, 40)`.

**Key idea.** Stop prescribing a position scheme — delete it (NoPE; Kazemnejad et al. 2023). A *causal*
decoder is not order-invariant: the query at position `t` attends to a window of size `t`, so the mask
already injects order. With no positional code the architecture can still recover absolute position in
layer 1 (uniform attention over `t` identical keys → `1/t`, anchored at `BOS`) and synthesise a relative
score in later layers (`q_t = [1, -t, …]`, `k_i = [i, 1, …]` → logit `= content − (t − i)`). SGD learns
whichever the task rewards.

**Why.** Every explicit scheme fixes a distance shape before seeing the data, and each fixed shape
mismatches some length regime — and IID accuracy cannot distinguish them. Removing the prescription
removes the absolute prior that was breaking; the mask-derived position can be *relative*, and a relative
offset recurs at unseen lengths, so nothing in `q_t^⊤ k_i` goes out of distribution. For copy/reverse the
rewarded pattern is relative and bimodal (attend to the paired source token *and* the local context),
which a monotone-recency prior could not express. Cost: strictly less work than any explicit scheme.

**Scaffold edit.** Return a scheme with all three hooks `None` and empty `extra_modules`; `build_model`
returns the plain `SeqModel(use_lstm=False)`. The harness already supplies both ingredients of the
existence proof — the layout starts with `BOS` (the anchor) and the causal mask is applied
unconditionally — so the attention score is exactly `q_t^⊤ k_i / √head_dim` under the lower-triangular
mask. There is no machinery to add or omit; the method is the absence of a positional scheme.

**Hyperparameters.** None — no learnable positional parameters, no bias, no rotation. All training
settings fixed by the loop.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- step 3: NoPE (no positional encoding)
def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """No positional encoding (Kazemnejad et al., NeurIPS 2023)."""
    return PositionalScheme(
        name="nope",
        token_embedding_extra=None,
        attn_bias=None,
        rotary=None,
        extra_modules=nn.ModuleList(),
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Decoder-only Transformer with no explicit positional encoding."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```
