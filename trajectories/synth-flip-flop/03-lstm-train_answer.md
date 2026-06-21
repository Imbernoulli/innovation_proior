The selective SSM carried the bit in a linear gated state and that move paid off almost everywhere: dense went to a perfect $1.0$ and long_ctx to $1.0$, both up from the Transformer's near-zero, confirming that a state update is the same operation at every position and that $T=1024$ is no longer out of distribution. But sparse stalled at $0.723$ (read-error $0.0657$, sequence-error $0.277$), and that residual is precise. Holding one write across the sparse tail's $100+$ ignore tokens demands that the state decay $\bar A = \exp(-\Delta\cdot A)$ sit at essentially $1$ on every ignore, and with a small state, one block, and a first-order input update, a tiny *systematic* leak in $\bar A$ accumulates over a long run and eventually drifts the held value far enough to flip the read-out's argmax. The bit is carried, but the carry is *analog* — a real-valued state slowly leaking — and "approximately held" over a hundred steps is exactly where the sparse tail lives. The direction was right; what I need is a recurrent state whose hold is *exactly* lossless rather than merely close.

I propose a gated recurrent cell — the LSTM — as the architecture, and the reason it closes the sparse gap is structural, not incidental. The heart of it is a *constant error carousel*: a cell state $c_t$ with an additive self-loop, so that when nothing is written, $c_t = c_{t-1}$ *identically* — no decay, no leak, the gradient riding the state multiplied by exactly $1$ per step. This is strictly stronger than the SSM's $\bar A \approx 1$: the SSM's hold is a decay that only approaches unity and so leaks, whereas the carousel's hold is the additive identity by construction, and over two hundred ignore tokens the carousel state does not move at all. A bare unit self-loop is not enough, though, because the cell must still *receive* writes and *block* the irrelevant ignores that follow, and a single static input weight is one number that cannot be context-sensitive. The fix is a *multiplicative* gate — and multiplication is essential, because only multiplication by $0$ can *completely* block an input. A sigmoid gate driven to saturation reaches that $0$ for practical purposes, and that practical saturation is the difference between a held bit and a leaking one.

Concretely the cell maintains a cell state $c_t$ and a hidden output $h_t$. From the input $x_t$ and previous hidden $h_{t-1}$ it computes, through learned affine maps and nonlinearities, an input gate, a forget gate, an output gate, and a candidate:

$$i_t = \sigma(W_i[x_t, h_{t-1}]), \quad f_t = \sigma(W_f[x_t, h_{t-1}]), \quad o_t = \sigma(W_o[x_t, h_{t-1}]), \quad g_t = \tanh(W_g[x_t, h_{t-1}]).$$

The cell state updates as

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t,$$

a forget-scaled carry plus an input-gated candidate — the carousel with a *learned* hold — and the output is $h_t = o_t \odot \tanh(c_t)$. Each piece earns its place against the flip-flop requirement. The forget gate $f_t$ replaces the fixed weight-$1$ self-loop with a learned $f_t \in [0,1]$ the cell can pin at $1$ to hold exactly, or drop when a write should clear the old value before overwriting. The input gate $i_t$ and the candidate $g_t$ are *separate*, which decouples "whether to write" from "what to write," so each can saturate independently. The output gate $o_t$ exposes the cell's contents at read tokens and hides them otherwise. The $\tanh$ on the candidate and the read-out keep the exposed value on a bounded scale matched to the linear head, while the gates being sigmoids let them saturate to exact $0/1$ decisions. The desired behaviour then reads straight off the token stream: at an ignore, $f_t \to 1$ and $i_t \to 0$ — hold exactly; at a write, $i_t \to 1$ and $f_t$ low — overwrite; at a read, $o_t \to 1$ — expose the bit. Every one of these is a saturated gate value a sigmoid reaches cleanly, and the value between writes is preserved by the additive recurrence with no analog leak.

This is why the LSTM is strictly stronger than the SSM rather than merely a different recurrence, and two differences target the sparse leak directly. First, the carry is purely additive with a learned forget gate, $c_t = f_t \cdot c_{t-1} + \dots$, so $f_t$ at sigmoid-saturation $1$ gives an *exact* hold; the SSM's $\exp(-\Delta\cdot A)$ is structurally a decay that only approaches $1$ and leaks. Second, the separate input gate and candidate decouple write magnitude from the decision to write, where the SSM folds both into the single $\Delta$-and-$B$ path and so couples write magnitude to the timescale. Both make the long-ignore hold lossless where the SSM's was lossy. A single-bit memory is literally a single gated cell — the recurrent skyline — and the LSTM realises it with the cleanest mechanism available.

Grounded in this task's edit surface, the implementation is about as minimal as the cell gets, and that minimality is itself the claim. The `FlipFlopModel` becomes an `nn.Embedding(6, 128)` mapping the six tokens to width $128$, a *single-layer* `nn.LSTM(128, 128)`, and a bias-free `nn.Linear(128, 6)` head on the per-step hidden outputs. The `forward` is three lines — embed, run the LSTM over the whole sequence, project — and it is causal by construction because `nn.LSTM` consumes the sequence strictly left to right, so the output at $t$ depends only on inputs $\le t$ and no causal mask is needed. It handles $T=1024$ for free, since the recurrence simply runs more steps; there is no positional embedding at all, which is exactly why long context is a non-issue — position is implicit in recurrence order, so there is nothing to extrapolate. At roughly $133\text{K}$ parameters it is the smallest model on the entire ladder, an order of magnitude below the SSM and two below the Transformer, far under the $50\text{M}$ cap; one layer with $128$ hidden units gives ample room to learn the write/hold/read gate logic with margin. The falsifiable expectation is clean: match the SSM's $1.0$ on dense and long_ctx, and push *sparse to $1.0$* — because the bit, once written, is held identically across any number of ignores and read out unambiguously — for a perfect overall geometric mean. That confirms the whole ladder's failure was a single mechanism seen at three sharpnesses: the Transformer re-selected the bit softly and compounded errors, the SSM carried the bit but leaked the carry, and the LSTM carries the bit and holds it exactly.

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
