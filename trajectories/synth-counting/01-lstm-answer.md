**Problem.** Maintain an integer counter in a sequence encoder that survives both in-distribution and
length-out-of-distribution test inputs. The first rung asks how far a recurrent counter gets on its
own, because a recurrent net is the model whose internal mechanism is a counter by construction — and
its failure mode is what the rest of the ladder must beat.

**Key idea.** Use a gated memory cell (LSTM) as the encoder. Its linear cell state, carried by a
self-loop at unit gain, can implement an explicit integer tally (open the input gate on `a`, hold with
the forget gate near one, compare at the `c`-block), and its gradient survives the long lag between the
`a`-block and the `c`-block that a vanilla RNN drowns. Two layers split "carry the counts" from
"compute the `a==b==c` decision".

**Why this and not the full paper recipe.** Only the *encoder* is editable here: embedding → 2-layer
LSTM → read the last valid hidden → LayerNorm. The optimizer (AdamW), loss (smooth-L1 / BCE), head, and
data are fixed by the harness, so the qlib-style Adam/masked-MSE/early-stop training loop of the
single-round trace is not part of this slot. Because the LSTM is causal and the CLS token sits *before*
the content (position 0), the summary must be read at `lengths - 1` (the last valid position), not at
CLS — reading CLS would summarise an empty sequence.

**Hyperparameters.** `nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)`; `nn.LSTM`
`input_size=hidden_dim=128`, `hidden_size=128`, `num_layers=2`, `batch_first=True`;
`nn.LayerNorm(hidden_dim)` on the pooled vector. Pool: gather output at `(lengths-1).clamp(min=0)`.

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
