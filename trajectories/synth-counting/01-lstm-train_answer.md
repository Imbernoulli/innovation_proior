The scaffold default is a 2-layer GRU reading its last hidden state, and the research question is squarely about *counters* — not whether a model can count inside the lengths it trained on, but whether it maintains an honest integer counter that survives when the test input is two-to-four times longer. So I want the first rung to be the model whose internal mechanism is a counter *by construction*, not by accident, because that fixes an honest recurrent floor every later method has to beat and, if it breaks, tells me precisely *where* recurrence breaks. A plain RNN is the wrong place to stop for one specific reason: trace an error backward through a recurrence $h_t = f(h_{t-1}, x_t)$ over $q$ steps and it is multiplied by a product of $q$ Jacobian factors, each a squashing-derivative times a recurrent weight — below one in magnitude, so the product decays geometrically in the lag. On `abc` the lag is exactly what matters: deciding whether a string is $a^n b^n c^n$ means comparing the `a`-block at the front against the `c`-block at the back, dozens of steps apart, and a vanilla RNN's gradient connecting them has vanished, so it learns local texture instead of a threaded count.

I propose an **LSTM encoder** — the gated memory cell as a programmable accumulator. The cure for the vanishing gradient is structural: a *linear* cell state $c_t$ carried by a self-loop at unit gain, so the backpropagated error riding that state is multiplied by $1$ per step rather than by a sub-unit factor, plus three sigmoid gates — input (when to write), forget (when to reset), output (when to read) — that decide from context what the cell stores and exposes. The forward dynamics

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

are, for counting, exactly an accumulator: the network can learn an input gate that opens on `a`, a candidate $g_t$ that contributes $+1$, a forget gate pinned near one so the running tally is held undamped, and a later contribution that checks the tally against the `b` and `c` blocks. The backward recursion makes the same point in gradient form — the state error at time $t$ inherits the next step's state error scaled by the forget gate, $\varepsilon_s^t = \cdots + f_{t+1} \odot \varepsilon_s^{t+1}$, which is unit-gain when the forget gate is open — so credit for "increment here" reaches back across the whole block instead of dying in the lag. This is why a finite-precision LSTM can in principle *implement* an integer counter rather than memorize a length-specific table: the counting recipe is a fixed point of the cell dynamics. This is the counter recipe of Weiss/Goldberg/Yahav 2018, built on the gated cell of Hochreiter & Schmidhuber 1997.

The design choices follow from the harness contract: the encoder receives `tokens: LongTensor[B, T]` with a CLS at position 0 and must return `[B, hidden_dim]`. The bottom is `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)` — five symbols (`PAD, a, b, c, CLS`) into a 128-dim space, with the padding row pinned to zero so padded positions contribute nothing to the recurrence. Then `nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)`. Two layers, not one: a single layer must both *maintain* the count and *compute* the membership decision from it, whereas stacking lets the lower layer carry the running tallies and the upper layer combine them into the $a = b = c$ comparison — a small amount of depth buys a cleaner division of labor without exploding the parameter budget. Width 128 matches `hidden_dim` so the summary lands in exactly the shape the fixed head expects, with no extra projection.

The one place I must respect a scaffold subtlety rather than copy the attention playbook is pooling. The CLS sits at position 0, *before* any content, and an LSTM is causal and left-to-right: its state at position 0 has seen only the CLS, so reading the CLS position would summarise an empty sequence. Information accumulates as the recurrence sweeps left to right, so the state that has seen the *whole* string is the one at the *last valid position*. The harness gives me `lengths` (true length including CLS), so I gather the output at index `lengths - 1`: `last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))`, then `out.gather(1, last_idx).squeeze(1)`. The `clamp(min=0)` guards the degenerate empty case; the gather picks per-row the genuinely final hidden, never a padded position. A `nn.LayerNorm(hidden_dim)` on the pooled vector then stabilizes the scale into the fixed linear head — a length-8 `exact` string and a length-256 OOD string would otherwise hand the head summaries on very different scales.

What this rung contributes is *only* the encoder. The optimizer (AdamW at 3e-4), the loss (smooth-L1 / BCE), the head, and the data are fixed by the harness; the qlib-style Adam/masked-MSE/early-stop training loop of the single-round trace is not part of this slot. So the falsifiable expectation is a *split*: in-distribution, the cell's gradient reaches across the training-range blocks and a learned increment/compare counter fits the data, so I expect high in-distribution accuracy on both `abc` (membership) and `exact` (the easiest counter — one accumulator, no comparison). Out of distribution is where the task is built to expose the weakness: the LSTM's counter is exact only to the extent its learned per-step increment is exact and its read-out $\tanh(c_t)$ has not saturated. At OOD lengths of 128–256 the accumulated state can drift past where $\tanh$ is responsive, or a per-step increment of $0.99$ instead of $1.00$ compounds over twice as many steps — so the `exact` OOD accuracy could collapse and `length-ood` retention could be poor. If that split shows up, it tells me the recurrent counter is *memorized in-range* rather than *implemented as an exact integer recurrence*, and the next rung must change the representation of length itself rather than just the way the count is carried.

```python
# EDITABLE region of pytorch-examples/synth_counting/custom_strategy.py — step 1: LSTM
def build_model(config: TaskConfig) -> nn.Module:
    """LSTM encoder following Weiss/Goldberg/Yahav 2018 counter recipe."""

    class LSTMEncoder(nn.Module):
        def __init__(self, cfg: TaskConfig):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=cfg.pad_id)
            self.rnn = nn.LSTM(
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

    return LSTMEncoder(config)
```
