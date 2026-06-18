**Problem.** The depth-1 Transformer solved the two solvable environments (`memory_unit` 1.000,
`grid_world` 0.882) but collapsed on the non-solvable `random_dfa` (0.205), and the geometric-mean
aggregate is pinned by that third number. Simulating a length-`T` automaton run is a chain of `T`
compositions; for a non-solvable transition semigroup this chain provably does *not* collapse below
`O(log T)` attention-mixing stages (Liu et al. 2022 Thm 4). One layer is one stage — the missing
resource is **effective depth**.

**Key idea.** Buy depth *without* buying parameters by tying weights across stages: hold one shared
Transformer encoder block and apply it `n_loops` times to the embedded sequence, feeding each output
back as input. An automaton applies the *same* transition operator every step, so a shared, iterated
block is the matching hypothesis class — it learns one "advance one round" operator and repeats it,
giving `n_loops` sequential composition stages at one-layer parameter cost. This task needs only that
minimal form (no scratchpad/memory/command layout, no addressed reads — the contract is just a
`[B,T]→[B,T,Q]` module), so the loop lives inside the module's `forward`.

**Why these pieces.** Same learned token + absolute position embeddings and causal mask as the shallow
baseline (order injection; prefix-only dependence must hold at *every* loop, so the mask is threaded
through each iteration). A single closing `LayerNorm` before the head, because pre-norm leaves the
residual stream un-normalized and six residual passes through one block let the scale drift. Six loops:
`log T ≈ 5.3` for `T=40`, so six is the smallest depth comfortably past the `O(log T)` threshold while
staying in the wall-time budget (~6× the shallow compute).

**Hyperparameters.** `d_model=128`, `n_heads=4`, `n_loops=6`, dropout 0. AdamW `lr=3e-4, wd=1e-4,
beta1=0.9, beta2=0.999` (unchanged from the shallow baseline to keep the depth comparison clean).

**What to watch.** `memory_unit` should stay 1.0 (extra loops can act as near-identity); `grid_world`
should close to ≈1.0 if its 0.882 leak was a depth deficit; `random_dfa` should rise materially above
0.205 but *not* solve — six loops is only just past `O(log T)` and `S_5` is non-solvable, so the third
environment stays the bottleneck. The gap looping cannot close motivates an `O(T)`-sequential recurrence.

```python
class CustomSequenceModel(nn.Module):
    """Looped Transformer: one shared encoder layer applied n_loops times."""

    def __init__(self, vocab_size: int, num_states: int, seq_len: int,
                 d_model: int = 128, n_heads: int = 4, n_loops: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_loops = n_loops
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        mask = self.causal_mask[:T, :T]
        for _ in range(self.n_loops):
            x = self.block(x, src_mask=mask, is_causal=True)
        x = self.final_norm(x)
        return self.head(x)


def build_model(env_spec: EnvSpec, config: TaskConfig) -> nn.Module:
    """Looped Transformer: shared block, 6 loops, d_model=128, 4 heads."""
    return CustomSequenceModel(
        vocab_size=env_spec.alphabet_size,
        num_states=env_spec.num_states,
        seq_len=env_spec.seq_len,
        d_model=128,
        n_heads=4,
        n_loops=6,
        dropout=0.0,
    )


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """AdamW (Liu et al. 2022 App. B.3 GPT-2 recipe: lr 3e-4, wd 1e-4)."""
    return {"lr": 3e-4, "wd": 1e-4, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    return model(input_ids)
```
