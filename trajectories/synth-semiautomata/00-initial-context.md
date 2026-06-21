## Research question

A finite-state semiautomaton is `A = (Q, Σ, δ)`: it reads a symbol stream `σ_{1:T} ∈ Σ^T` and walks a
state sequence `q_t = δ(q_{t-1}, σ_t)` from a fixed start state. The design problem is the
**sequence-model architecture** that maps the full symbol stream `σ_{1:T}` to the full state trajectory
`q_{1:T}` under per-token cross-entropy against the true states. Everything else — the environments,
the online data stream, the loss, the optimizer, early stopping, and evaluation — is fixed.

Liu et al. 2022 prove that every semiautomaton admits an `O(log T)`-depth Transformer simulator;
that solvable semiautomata admit constant-depth simulators via Krohn–Rhodes decomposition; and that
non-solvable semiautomata — the smallest generates `A_5`, the alternating group on five letters —
provably do **not** admit a constant-depth shortcut unless `TC^0 = NC^1`. The three environments sit
on opposite sides of this complexity wall.

## Prior art / Background / Baselines

- **Recurrent encoder–decoder (Bahdanau et al. 2014).** Maintain a hidden state that is updated
  serially at each input step, turning the model into a learned state register.
- **Convolutional sequence models (ByteNet, Kalchbrenner et al. 2017).** Replace recurrence with
  stacked convolutions that process all positions in parallel; relating positions `D` apart requires
  `O(D/k)` or `O(log_k D)` layers.
- **Self-attention (Vaswani et al. 2017).** Route every position to every other in a single
  `softmax(QKᵀ/√d_k)V` operation, giving `O(1)` path length and parallel computation.

## Fixed substrate / Code framework

A single driver is frozen and exposes three synthetic semiautomaton environments plus an online stream
of `(input, state)` sequences:

1. **`memory_unit`** — a `K`-state memory cell (`K=8`), alphabet `{noop, write(0), …, write(K−1)}`:
   `noop` keeps the state, `write(j)` jumps to state `j`. Sequence length 40, 8000 steps.
2. **`grid_world`** — `Grid_9`: states `{0,…,8}` on a line, alphabet `{L, R}`, reflecting boundaries.
   Liu et al. 2022 give an `O(1)`-depth self-attention construction for this task. Sequence length 40,
   8000 steps.
3. **`random_dfa`** — a random transition table `δ : Q × Σ → Q` with `|Q|=60`, `|Σ|=8`, fixed by a seed.
   A random table on this scale typically generates a transformation semigroup containing `S_5`, so it
   lies in the non-solvable regime. Sequence length 40, 12000 steps.

Training uses per-token cross-entropy against `q_{1:T}`, online fresh batches, batch size 64,
gradient clipping at 1.0, AdamW, and early stopping after three consecutive log windows with training
accuracy ≥0.999. The optimizer is always AdamW with exactly the four hyperparameters the editable
interface returns.

## Editable interface

Exactly one region is editable — the model, the optimizer config, and the inference wrapper in
`pytorch-examples/synth_semiautomata/custom_strategy.py`. Every method must fill this contract:

- `build_model(env_spec, config) -> nn.Module` maps token ids
  `input_ids ∈ {0,…,alphabet_size−1}^{B×T}` to per-position state logits `[B, T, num_states]`.
- `get_optimizer_config(config) -> dict[str, float]` returns the AdamW hyperparameters
  `lr, wd, beta1, beta2`.
- `forward_logits(model, input_ids, env_spec) -> torch.Tensor` is the inference wrapper; use it for
  architectures that do not fit a plain `model(x)` call.

`env_spec` carries `alphabet_size`, `num_states`, and `seq_len`; the same architecture is built for
all three environments, differing only in those dimensions. The starting point is a shallow 1-layer
GPT-2-style causal Transformer.

```python
# EDITABLE region of custom_strategy.py — default fill (shallow 1-layer Transformer)
class CustomSequenceModel(nn.Module):
    """Default baseline: shallow GPT-2-style Transformer (1 layer).

    Maps token ids [B, T] to per-position state logits [B, T, Q].
    Uses learned token+position embeddings, causal multi-head attention,
    a 2-layer MLP, and a final linear head.
    """

    def __init__(self, vocab_size: int, num_states: int, seq_len: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_states)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, mask=self.causal_mask[:T, :T], is_causal=True)
        return self.head(x)


def build_model(env_spec: EnvSpec, config: TaskConfig) -> nn.Module:
    """Construct the sequence model. Output: [B, T, num_states] logits."""
    return CustomSequenceModel(
        vocab_size=env_spec.alphabet_size,
        num_states=env_spec.num_states,
        seq_len=env_spec.seq_len,
        d_model=128,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )


def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
    """AdamW hyperparameters. Defaults follow Liu et al. 2022 App. B.3."""
    return {"lr": 3e-4, "wd": 1e-4, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    """Run the model on `input_ids` and return logits [B, T, num_states]."""
    return model(input_ids)
```

## Evaluation settings

One seed (42). Three environments, each scored by `test_accuracy` (also emitted as `score`): mean
per-token state-prediction accuracy over 32 freshly sampled held-out batches, bounded in `[0, 1]`,
higher is better. The aggregate task score is the **geometric mean across the three environments**, so
all three must be strong. Per-environment wall-time is budgeted at ~30 minutes.
