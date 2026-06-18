## Research question

Counting is the oldest probe of sequence-model expressivity, and the sharp question is not whether a
model can count *inside* the lengths it trained on — almost anything memorizes that — but whether it
maintains an honest *integer counter* that survives when the test input is two-to-four times longer
than anything it saw at training time. The single thing being designed here is the **encoder
architecture** — recurrent, transformer, state-space, hybrid; its layer count; its positional scheme;
its pooling. Everything else (initialization, the AdamW loop, the loss, the data generators, the
fixed scalar head, the evaluation splits) is frozen. The model must hold a count across three
environments at once: regress the number of `a`'s in a binary string, classify whether a string is
exactly `a^n b^n c^n`, and — the metric that actually separates methods — *retain* that classification
accuracy when `n` leaves the training range.

## Prior art before the first rung (counting-expressivity lineage)

The first rung reacts to a line of results on what sequence models can and cannot count. These precede
the ladder; the fixed substrate below is the harness they are all dropped into.

- **Finite-state acceptors / classical RNNs (Elman 1990; Siegelmann & Sontag 1992).** A recurrent net
  with bounded precision is, in the limit, a finite-state machine; a^n b^n needs an unbounded counter,
  which a finite automaton cannot maintain past its state budget. Gap: in principle a plain RNN cannot
  count arbitrarily far, and in practice the gradient that would teach it a counter vanishes across the
  long lag between the `a`-block and the `c`-block.
- **The gated memory cell (Hochreiter & Schmidhuber 1997; Gers et al. 2000).** A linear cell state with
  a self-loop at unit gain carries gradient across arbitrarily long lags; input/forget/output gates
  decide when to write, reset, and read. A cell that learns "+1 on `a`, hold, −1 on `c`" *is* an
  explicit integer counter, and its gradient survives the lag a plain RNN drowns. This is exactly the
  recipe the first rung uses. Gap: the counter lives in a continuous, bounded activation `tanh(c_t)`;
  far outside the training range the learned increment can drift or the read-out saturates, so the
  counter that was exact in-distribution degrades out-of-distribution.
- **Self-attention with positional encodings (Vaswani et al. 2017).** Replaces recurrence with
  any-to-any attention; order is injected by adding a positional code to the embedding. A CLS-pooled
  encoder can in principle compare block boundaries directly rather than threading a counter through
  time. Gap: attention is permutation-equivariant, so *all* of its sense of length comes from the
  positional code; a code that is sinusoidal-in-absolute-index, or learned per absolute index, is
  out-of-distribution at the longer test lengths and the count collapses there.
- **Position as the lever for length generalization (Press et al. 2022, ALiBi; Ruoss et al. 2023,
  randomized positions; McLeish et al. 2024, count-indexed embeddings).** The recurring finding is that
  length extrapolation is *governed by the positional representation*: a code that stays in-distribution
  at test length, and that exposes the latent count directly, is what lets a transformer extrapolate.
  This is the lineage the strongest rung draws on.

## The fixed substrate

A single training/evaluation driver is frozen and must not be touched. Each of the three environments
generates data on the fly (no network, no filesystem): `exact` draws binary strings over `{a, b}` and
regresses the count of `a`'s; `abc` draws balanced positive (`a^n b^n c^n`) and structured-negative
strings (wrong block counts, swapped order, interleaved middle) and classifies membership;
`length_ood` reuses the `abc` generator but scores *retention* of accuracy from the in-distribution to
the out-of-distribution length range. Every environment prepends a CLS token at position 0, trains for
6,000 steps of AdamW (`lr=3e-4`, `weight_decay=1e-2`, batch 64, grad-clip 1.0) under smooth-L1
(regression) or BCE-with-logits (classification), and evaluates on 1,024 freshly drawn examples per
split. A fixed `nn.Linear(hidden_dim, 1)` head turns the encoder's pooled summary into the scalar the
loss consumes. The config exposes `vocab_size=5` (`0=PAD, 1=a, 2=b, 3=c, 4=CLS`), `hidden_dim=128`,
`train_max_len=64`, `ood_multiplier=4`.

## The editable interface

Exactly one region is editable: the body of `build_model(config: TaskConfig) -> nn.Module` in
`pytorch-examples/synth_counting/custom_strategy.py`. The returned module is the **encoder**, and the
contract is the only hard constraint:

- **Input** — `tokens: LongTensor[B, T]` padded with `config.pad_id`; position 0 is always
  `config.cls_id`. `lengths: LongTensor[B]` gives the true length *including* the CLS token.
- **Output** — a tensor `[B, config.hidden_dim]`: the pooled sequence summary the fixed head consumes.
- **Capacity** — at most 5,000,000 parameters (asserted at startup).

Anything inside that contract is fair game: encoder type, positional scheme, layer count, attention
pattern, pooling rule. Note the CLS subtlety the scaffold flags — a causal/left-to-right encoder must
read the *last* valid position to summarise the sequence (the CLS sits *before* any content), while a
bidirectional or self-attention model may pool the CLS position instead.

The starting point is the scaffold default: a 2-layer GRU reading its final hidden state. Each rung on
the ladder replaces exactly this `build_model` body and nothing else.

```python
# EDITABLE region of pytorch-examples/synth_counting/custom_strategy.py — default fill (2-layer GRU)
def build_model(config: TaskConfig) -> nn.Module:
    """Default scaffold: a 2-layer GRU encoder reading the final hidden state."""

    class GRUEncoder(nn.Module):
        def __init__(self, cfg: TaskConfig):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_id)
            self.rnn = nn.GRU(
                input_size=cfg.hidden_dim,
                hidden_size=cfg.hidden_dim,
                num_layers=2,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(cfg.hidden_dim)

        def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            h = self.embed(tokens)
            out, _ = self.rnn(h)
            last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            last = out.gather(1, last_idx).squeeze(1)
            return self.norm(last)

    return GRUEncoder(config)
```

## Evaluation settings

Three environments, one seed (42). Each prints `FINAL_METRICS`/`TEST_METRICS` with
`in_dist_accuracy`, `ood_accuracy`, `extrapolation_gap`, and the primary `score` (already in `[0, 1]`).
For `exact` the score is the OOD accuracy on `T in [128, 256]` (rounded-count-equals-target); for `abc`
it is the in-distribution membership accuracy; for `length-ood` it is the retention
`1 - max(0, in_dist_acc - ood_acc)` on `n in (T/3, 4T/3]`. The aggregate task score is the geometric
mean across the three. Higher is better on every metric.
