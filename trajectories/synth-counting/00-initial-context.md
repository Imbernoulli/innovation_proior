## Research question

Counting probes sequence-model expressivity. The sharp question is not whether a model can count inside the lengths it trained on — memorization is easy — but whether it maintains an honest *integer counter* when the test input is two-to-four times longer than anything it saw at training time. The single design object is the **encoder architecture** — recurrent, transformer, state-space, hybrid; its layer count; its positional scheme; its pooling. Everything else is frozen: initialization, the AdamW loop, the loss, the data generators, the fixed scalar head, and the evaluation splits. The model must operate across three environments at once: regress the number of `a`'s in a binary string; classify whether a string is exactly `a^n b^n c^n`; and *retain* that classification accuracy when `n` leaves the training range.

## Prior art / Background / Baselines

These are the counting-expressivity results the rungs react to; the fixed substrate below is the harness they are all dropped into.

- **Finite-state acceptors / classical RNNs.** A recurrent net with bounded precision behaves, in the limit, like a finite-state machine; `a^n b^n` requires an unbounded counter, beyond a finite automaton's state budget.
- **The gated memory cell.** A cell state with input, forget, and output gates writes, holds, and reads information across long sequences. A cell can learn to increment on `a`, hold through `b`, and decrement or compare at `c`, implementing a tally in continuous activations.
- **Self-attention with positional encodings.** Attention replaces recurrence with any-to-any comparisons, and order is injected by adding a positional code to the embedding. A CLS-pooled encoder can compare block boundaries directly rather than threading a counter through time.
- **Position as the lever for length generalization.** ALiBi, randomized positions, and count-aware position schemes show that length extrapolation is governed by how position is represented.

## Fixed substrate / Code framework

A single training/evaluation driver is frozen and must not be touched. Each of the three environments generates data on the fly (no network, no filesystem): `exact` draws binary strings over `{a, b}` and regresses the count of `a`'s; `abc` draws balanced positives (`a^n b^n c^n`) and structured negatives (wrong block counts, swapped order, interleaved middle) and classifies membership; `length_ood` reuses the `abc` generator but scores *retention* of accuracy from the in-distribution to the out-of-distribution length range. Every environment prepends a CLS token at position 0, trains for 6,000 steps of AdamW (`lr=3e-4`, `weight_decay=1e-2`, batch 64, grad-clip 1.0) under smooth-L1 (regression) or BCE-with-logits (classification), and evaluates on 1,024 freshly drawn examples per split. A fixed `nn.Linear(hidden_dim, 1)` head turns the encoder's pooled summary into the scalar the loss consumes. Config: `vocab_size=5` (`0=PAD, 1=a, 2=b, 3=c, 4=CLS`), `hidden_dim=128`, `train_max_len=64`, `ood_multiplier=4`.

## Editable interface

Exactly one region is editable: the body of `build_model(config: TaskConfig) -> nn.Module` in `pytorch-examples/synth_counting/custom_strategy.py`. The returned module is the **encoder**, with one hard contract:

- **Input** — `tokens: LongTensor[B, T]` padded with `config.pad_id`; position 0 is always `config.cls_id`. `lengths: LongTensor[B]` gives the true length *including* the CLS token.
- **Output** — a tensor `[B, config.hidden_dim]`: the pooled sequence summary the fixed head consumes.
- **Capacity** — at most 5,000,000 parameters (asserted at startup).

Anything inside that contract is allowed: encoder type, positional scheme, layer count, attention pattern, pooling rule. Note the CLS subtlety the scaffold flags — a causal/left-to-right encoder must read the *last* valid position to summarise the sequence (the CLS sits *before* any content), while a bidirectional or self-attention model may pool the CLS position instead.

The starting point is the scaffold default: a 2-layer GRU reading its final hidden state. Each rung replaces exactly this `build_model` body and nothing else.

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

Three environments, one seed (42). Each prints `FINAL_METRICS`/`TEST_METRICS` with `in_dist_accuracy`, `ood_accuracy`, `extrapolation_gap`, and the primary `score` (already in `[0, 1]`). For `exact` the score is the OOD accuracy on `T in [128, 256]` (rounded-count-equals-target); for `abc` it is the in-distribution membership accuracy; for `length-ood` it is the retention `1 - max(0, in_dist_acc - ood_acc)` on `n in (T/3, 4T/3]`. The aggregate task score is the geometric mean across the three. Higher is better on every metric.
