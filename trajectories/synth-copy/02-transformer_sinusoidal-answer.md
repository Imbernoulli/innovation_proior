**Problem (from rung 1).** The recurrent baseline is perfect in-distribution but `exact_match_ood = 0.0`
on every variant: implicit, recurrence-carried order does not survive being stretched past the training
length, and it is slow (sequential, 400–500 s/variant). The fix is an *explicit* position signal on the
parallel decoder-only Transformer the rest of the ladder shares.

**Key idea.** Sinusoidal absolute positional encoding (Vaswani et al. 2017, §3.5). Self-attention is
permutation-equivariant, so add to each token embedding a fixed vector
`PE(t, 2i) = sin(t·ω_i)`, `PE(t, 2i+1) = cos(t·ω_i)` with `ω_i = 10000^{-2i/d}` — a multi-frequency
"continuous counter" that is bounded, shift-consistent (a fixed offset is a fixed rotation per
frequency, `p_{t+k} = M_k p_t`), and closed-form at *any* index, unlike the scaffold default's learned
table that has no entry past the training range.

**Why.** Many geometrically spaced frequencies separate a large range while each coordinate stays in
`[-1, 1]`; pairing sin with cos per frequency makes a relative shift a single position-independent linear
map. Because `sin`/`cos` are defined for every real `t`, position 30 is the next point on the same curves,
not a missing slot. The catch — which this rung exists to test — is that the model only *learned to
interpret* the phase patterns over `[0, 20)`, so out-of-range patterns are out of distribution: defined
≠ usable.

**Scaffold edit / differences from the canonical recipe.** Fill `build_positional_scheme` to build the
closed-form table, wrap it frozen (`from_pretrained(..., freeze=True)`, no gradient), register it via
`scheme.extra_modules`, and return only the `token_embedding_extra` hook; `build_model` returns the plain
`SeqModel(use_lstm=False)`. The harness adds the position term directly onto the raw token embedding, so
there is **no `√d_model` embedding scaling** and **no dropout** (the task fixes `dropout=0.0`); positions
index the whole stream `[0, T)` with a `clamp` to the table's last row (table sized `max_total_len=256`,
so the clamp never fires on this task).

**Hyperparameters.** Table length `max_total_len = 256`, `d_model = 128` (even, required), base 10000;
no learnable positional parameters. All training settings fixed by the loop.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- step 2: sinusoidal absolute PE
def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """Sinusoidal absolute positional encoding (Vaswani et al., 2017)."""
    max_len = config.max_total_len
    d_model = config.d_model
    pe = torch.zeros(max_len, d_model)
    positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    extras = nn.ModuleList()
    table = nn.Embedding.from_pretrained(pe, freeze=True)
    extras.append(table)

    def token_embedding_extra(positions: torch.Tensor) -> torch.Tensor:
        return table(positions.clamp(max=max_len - 1))

    return PositionalScheme(
        name="sinusoidal",
        token_embedding_extra=token_embedding_extra,
        attn_bias=None,
        rotary=None,
        extra_modules=extras,
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Decoder-only Transformer with sinusoidal APE."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```
