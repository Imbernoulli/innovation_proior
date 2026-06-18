**Problem.** Looping closed the solvable environments (`memory_unit`/`grid_world` → 1.000) but lifted
the non-solvable `random_dfa` only from 0.205 to 0.309 — effective depth 6 is past the constant-depth
floor but still capped by the `O(log T)` barrier, and the geometric mean stays pinned there. Liu et al.
2022 Thm 1/4 are about *parallel* models with a fixed number of mixing stages; "more parallel depth" is
the wrong axis.

**Key idea.** Stop shortcutting the composition and *perform* it: a recurrence that reads symbols one
at a time and applies one **exact** state update per token gives `O(T)` strictly sequential stages —
one per symbol, the automaton's own definition — so the non-solvability barrier (which is about
*compressing* the composition into few stages) does not apply. The LSTM's hidden state *is* the
simulated automaton state; its forget/input/output gates are a learned transition function `δ`. The
constant-error carousel (`c_t = f_t⊙c_{t-1} + i_t⊙g_t`, backward `ε_s^t = … + f_{t+1}⊙ε_s^{t+1}`) makes
gradient survive the full 40-step lag at unit gain, so a symbol at step 1 can be credited for the loss
at step 40 — exactly the long-range credit assignment `random_dfa` needs.

**Why these pieces (grounded in this task).** Per-position output: read out *all* hidden states
`[B,T,hidden]` (not a last-step readout) so each step's prediction lines up with the per-token CE loss
under full state supervision. Single layer — one layer already gives one exact composition per token;
extra layers would re-add parallel-style depth, which is not what solves the task. No causal mask
needed: a forward recurrence is causal by construction (`h_t` depends only on `x_{1:t}`). Embedding
width 64 carries symbol identity; hidden 128 is the state register.

**Hyperparameters.** `emb_dim=64`, `hidden_dim=128`, `num_layers=1`. AdamW `lr=1e-3` (an order above the
Transformers' 3e-4 — the carousel keeps gradients well-scaled, so a larger step converges fast),
`wd=1e-9` (near-zero: decaying the recurrent/forget-gate weights would bias the cell toward forgetting,
the wrong prior for remembering 40 symbols), `beta1=0.9, beta2=0.999`. Harness clip 1.0 caps the
exploding side.

**What to watch.** `memory_unit` and `grid_world` should both be 1.000 (the gates are exactly the
noop/write logic; the recurrence computes the clamped prefix sum exactly). The decisive test is
`random_dfa`: the exact-per-token update with full-lag gradient should beat the looped Transformer's
0.309 by a clear margin and post the highest geometric mean of the three — not necessarily 1.0 (60
states, random table, 40 steps), but the strongest, because it trades parallel depth for sequential
exactness.

```python
class CustomSequenceModel(nn.Module):
    """Single-layer LSTM (Liu et al. 2022 App. B.3: emb=64, hidden=128)."""

    def __init__(self, vocab_size: int, num_states: int, seq_len: int,
                 emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, num_states)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        h, _ = self.lstm(x)
        return self.head(h)


def build_model(env_spec: EnvSpec, config: TaskConfig) -> nn.Module:
    """Single-layer LSTM, emb=64, hidden=128."""
    return CustomSequenceModel(
        vocab_size=env_spec.alphabet_size,
        num_states=env_spec.num_states,
        seq_len=env_spec.seq_len,
        emb_dim=64,
        hidden_dim=128,
        num_layers=1,
    )


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """AdamW (Liu et al. 2022 App. B.3: LSTM uses lr 1e-3, wd 1e-9)."""
    return {"lr": 1e-3, "wd": 1e-9, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    return model(input_ids)
```
