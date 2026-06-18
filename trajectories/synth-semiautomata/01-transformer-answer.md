**Problem.** Simulate a finite-state semiautomaton end to end: read the symbol stream `σ_{1:T}`, emit
the full state trajectory `q_{1:T}` (one per-position prediction), trained by per-token cross-entropy.
The harness fixes the three environments, the online stream, AdamW, and the early stop; the only object
designed is the `nn.Module` from `build_model` mapping `[B, T]` token ids to `[B, T, num_states]` logits.
The three environments straddle a complexity wall (Liu et al. 2022): solvable automata admit
constant-depth attention simulators, non-solvable ones provably do not (unless `TC^0 = NC^1`).

**Key idea (the starting probe).** Begin at the shallowest credible attention model — one causal
Transformer encoder layer — because its *failure pattern* is the complexity boundary made empirical. A
depth-1 model is the cleanest constant-depth probe: whatever it cannot reach, it cannot reach because a
constant-depth shortcut does not exist, not because of overfitting (the harness trains online).

**Why these pieces.** Learned token + absolute position embeddings, *added* (attention is
permutation-equivariant; a semiautomaton's state depends on symbol *order*, so order must be injected;
length is fixed at 40, so a learned table needs no extrapolation). A **causal mask** so position `t`
attends only to the prefix `≤ t` — this is exactly the semiautomaton constraint `q_t = f(σ_{1:t})`, not
a generation requirement, and it shrinks the hypothesis space to the correct one. Pre-norm + GELU 4×
MLP for stable optimization and per-position nonlinear capacity. Linear head to state logits.

**Hyperparameters.** `d_model=128`, `n_heads=4`, `n_layers=1`, dropout 0. AdamW `lr=3e-4, wd=1e-4,
beta1=0.9, beta2=0.999` (Liu et al. 2022 App. B.3 recipe, smaller width for the wall-time budget).

**What to watch.** `memory_unit ≈ 1.0` (one head copying from the last write solves it exactly);
`grid_world` high but possibly short of 1.0 (prefix sum is constant-depth, the wall-clamping
nonlinearity may not fully close in one mixing step); `random_dfa` poor (non-solvable `S_5` regime — no
constant-depth shortcut). Because the aggregate is the geometric mean, the `random_dfa` failure
dominates the score and is what the ladder must move.

```python
class CustomSequenceModel(nn.Module):
    """Shallow 1-layer GPT-2-style Transformer (Liu et al. 2022, App. B.3)."""

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
    """Shallow Transformer: 1 layer, d_model=128, 4 heads."""
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
    """AdamW (Liu et al. 2022 App. B.3: lr 3e-4, wd 1e-4)."""
    return {"lr": 3e-4, "wd": 1e-4, "beta1": 0.9, "beta2": 0.999}


def forward_logits(model: nn.Module, input_ids: torch.Tensor,
                   env_spec: EnvSpec) -> torch.Tensor:
    return model(input_ids)
```
