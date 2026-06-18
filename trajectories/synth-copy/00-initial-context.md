## Research question

A small causal Transformer can copy, repeat, or reverse a symbol string trivially *at the length it was
trained on* — the hard part is what it does at lengths it never saw. Train content lengths are sampled
from `[1, 20]`; evaluation runs both an in-distribution split (`id`, lengths in `[1, 20]`) and a strict
out-of-distribution split (`ood`, lengths in `[21, 40]`). The single thing being designed is the
**positional / attention mechanism** — how token order enters the model. Everything else (vocabulary,
decoder block layout, optimizer, data, metrics) is fixed. So the question is sharp: which way of
injecting position lets the same 4-layer / `d_model=128` decoder keep solving copy-style tasks past its
training length, and why.

## Prior art before the first rung (the order-encoding lineage)

The first rung reacts to a line of sequence models, each of which encodes order differently and breaks
length generalization in its own way. These precede the ladder; the fixed substrate below is the
decoder the later rungs all share.

- **Recurrent encoder-decoder (Sutskever, Vinyals, Le 2014; Cho et al. 2014).** Read the source with an
  RNN, compress it into one fixed-size vector, decode from that. Order is supplied by the recurrence
  itself — no explicit positional code. Gap: the constant-size summary is an information bottleneck, so
  it degrades on long sequences, worst beyond training lengths.
- **Additive attention seq2seq (Bahdanau, Cho, Bengio 2015).** Keep every encoder state and let the
  decoder read a content-weighted blend `c_i = Σ_j α_{ij} h_j` at each step, removing the bottleneck.
  Still recurrent, still no positional encoding — recurrence carries order. Gap: it is the classical
  non-Transformer reference here; recurrence does generalize order in principle but is slow and the
  hidden state is still a fixed-width channel under length stress.
- **Sinusoidal absolute PE (Vaswani et al. 2017, §3.5).** Self-attention is permutation-equivariant, so
  a fixed `d_model` position vector — a vector of geometrically spaced sines/cosines — is added to each
  token embedding before layer one. Closed-form, so it is *defined* at any index. Gap: the model only
  ever interprets the phase patterns over `[0, 20)`; past that the joint sinusoid configuration is out
  of distribution, so behaviour beyond training length is uncalibrated.
- **Learned absolute PE.** A trainable table, one vector per slot. Gap: there is literally no entry past
  the training range, so it cannot even represent an unseen length — ruled out for OOD by inspection.

## The fixed substrate

A self-contained character-level driver is frozen and must not be touched. Vocabulary: 16 symbols plus
`PAD/BOS/EOS/SEP` (`vocab_size = 20`). Sequence layout `[BOS] x_1 … x_T [SEP] y_1 … y_M [EOS]`, with the
loss computed only on the positions whose next token belongs to the target. Three variants, one test
command each: `delim` (`y = x`), `repeat` (`y = x` twice), `reverse` (`y = reverse(x)`). The model is a
decoder-only Transformer — token embedding, four `TransformerBlock`s (pre-LN, multi-head causal
attention, GELU MLP at 4× expansion), final LayerNorm, a linear head — with `d_model = 128`, 4 heads, 4
layers. Optimization is fixed: AdamW (lr `5e-4`, wd `1e-2`), batch 256, 6000 steps, 200-step linear
warm-up, grad-clip 1.0. The attention module already consumes a `PositionalScheme` with three optional
hooks and applies the causal mask after any bias; `build_model` may also swap the backbone to an LSTM.

## The editable interface

Exactly one region is editable — `build_positional_scheme(config)` and `build_model(config)` in
`pytorch-examples/synth_copy/custom_strategy.py` (lines 301–332). A `PositionalScheme` is a thin
container with three optional callables: `token_embedding_extra(positions) -> [B,T,D]` (an additive
token-level embedding, e.g. sinusoidal / learned APE); `attn_bias(T, device, dtype) -> [H,T,T]` or
`[1,T,T]` (an additive attention-score bias applied before the causal mask, e.g. ALiBi / T5 relative
bias); and `rotary(q, k) -> (q', k')` (applied inside attention before the dot product, e.g. RoPE). Any
learnable parameters must be registered via `scheme.extra_modules` (an `nn.ModuleList`) so AdamW picks
them up and they move to the GPU. `build_model` may return any `nn.Module` exposing
`forward(tokens) -> [B,T,V]`; passing `use_lstm=True` to `SeqModel` (or returning a custom module)
swaps the backbone away from the Transformer entirely.

The starting point is the scaffold default: a **learned absolute** positional embedding — a sensible but
length-limited fill. Each rung replaces exactly these two definitions and nothing else.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- default fill: learned absolute PE
def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """Default: a learned absolute positional embedding (length-limited)."""
    extras = nn.ModuleList()
    pos_emb = nn.Embedding(config.max_total_len, config.d_model)
    nn.init.normal_(pos_emb.weight, mean=0.0, std=0.02)
    extras.append(pos_emb)

    def token_embedding_extra(positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, T] with values in [0, max_total_len)
        return pos_emb(positions.clamp(max=config.max_total_len - 1))

    return PositionalScheme(
        name="learned_absolute",
        token_embedding_extra=token_embedding_extra,
        attn_bias=None,
        rotary=None,
        extra_modules=extras,
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Construct the model. Override to swap in a different backbone (e.g. LSTM)."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```

## Evaluation settings

One seed (42). Three variants, each tested by greedy autoregressive decoding on two splits: `id`
(content lengths `[1, 20]`) and `ood` (content lengths `[21, 40]`). Per variant the run reports
`exact_match_id`, `token_acc_id`, `exact_match_ood`, `token_acc_ood`, and a combined
`score = 0.5·exact_match_id + 0.5·exact_match_ood`. The leaderboard records the per-variant `score`; the
task-level summary is the **geometric mean of `score` across the three variants**, which penalises any
single variant where the model fails to length-generalise. Higher is better on every column.
