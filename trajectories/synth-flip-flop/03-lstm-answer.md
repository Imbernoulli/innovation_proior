**Problem.** The selective SSM fixed the compounding glitch on dense (1.0) and long_ctx (1.0) by carrying
the bit in a linear gated state, but stalled on sparse (0.723, seq-error 0.277). The residual is the
*analog leak*: holding one write across 100+ ignores needs `Ā=exp(-Δ·A)≈1` on every ignore, and a tiny
systematic leak accumulated over a long run flips the read-out's argmax. The carry is approximate; sparse
lives exactly where "approximate" fails.

**Key idea.** A gated recurrent (LSTM) cell, whose carry is *exactly* lossless. The cell state updates
additively, `c_t = f_t⊙c_{t-1} + i_t⊙g_t`, with sigmoid gates that *saturate*: forget-gate `f_t→1` holds
the value identically (no decay, unlike the SSM's `exp(-Δ·A)`), input-gate `i_t→0` blocks an ignore
*exactly* (only multiplication by 0 fully blocks), and at a write `i_t→1` admits the candidate `g_t`. The
output gate exposes the bit at reads. Every flip-flop action — hold / overwrite / read — is a saturated
0/1 gate the sigmoid reaches cleanly, so a written bit survives any number of ignores with no drift.

**Why strictly stronger than the SSM.** (i) Additive carry with a learned forget gate gives an *exact*
hold; the SSM's `exp(-Δ·A)` is structurally a decay that only approaches 1 and leaks. (ii) A separate
input gate and candidate decouple "whether to write" from "what to write," each saturating independently,
where the SSM folds both into `Δ`/`B`. Both differences make the long-ignore hold lossless. A single-bit
memory is literally a single gated cell — the recurrent skyline.

**Grounding in this task.** The edit is minimal: `nn.Embedding(6,128)` → 1-layer `nn.LSTM(128,128)` →
bias-free `nn.Linear(128,6)`. Causal for free (LSTM consumes left-to-right; no mask). No positional
embedding ⇒ no `T=1024` extrapolation problem. ~133K params — the smallest model on the ladder.

**Expectation.** Match the SSM's 1.0 on dense/long_ctx and push **sparse to 1.0** (the exact-hold cure for
the leak), giving a perfect overall geometric mean — confirming the ladder's failure was one mechanism at
three sharpnesses: soft re-selection (Transformer) → leaky carry (SSM) → exact carry (LSTM).

**Hyperparameters.** `hidden=128`, `num_layers=1`, `batch_first=True`, bias-free head. Trained by the
fixed loop (AdamW lr=3e-4, wd=0.1, 3000 steps, batch 16).

```python
# EDITABLE region of custom_strategy.py (lines 191-241) — step 3: 1-layer LSTM (recurrent skyline)
class FlipFlopModel(nn.Module):
    """1-layer LSTM. Perfect skyline per Liu et al. (NeurIPS 2023) R2."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, max_len: int = 1024):
        super().__init__()
        hidden = 128
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        out, _ = self.rnn(h)
        return self.head(out)


def build_model(config: TaskConfig) -> nn.Module:
    """Construct the LSTM FFLM (1-layer, hidden=128, ~133K params)."""
    return FlipFlopModel(
        vocab_size=VOCAB_SIZE,
        max_len=max(config.train_len, config.eval_long_len),
    )
```
