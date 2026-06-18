## Research question

A finite-state semiautomaton is `A = (Q, Σ, δ)`: it reads a symbol stream `σ_{1:T} ∈ Σ^T` and walks a
state sequence `q_t = δ(q_{t-1}, σ_t)` from a fixed start state. The single thing being designed is the
**sequence-model architecture** that, given the full symbol stream `σ_{1:T}` as input, emits the whole
state trajectory `q_{1:T}` — one per-position state prediction at every step — under per-token
cross-entropy against the true states. Everything else (the environments, the online data stream, the
loss, the AdamW driver, early stopping, evaluation) is fixed.

The interesting structure is that the three environments sit on opposite sides of a complexity wall.
Liu et al. 2022 ("Transformers Learn Shortcuts to Automata", arXiv:2210.10749, ICLR 2023) prove that
every semiautomaton admits an `O(log T)`-depth Transformer simulator (Thm 1); that *solvable*
semiautomata admit constant-depth simulators via Krohn–Rhodes decomposition (Thm 2); but that
*non-solvable* semiautomata — the smallest generates `A_5`, the alternating group on five letters —
provably do **not** admit a constant-depth shortcut unless `TC^0 = NC^1` (Thm 4). So the question is
sharper than "which architecture is best": it is which architecture covers *both* the regime where a
shallow shortcut exists and the regime where one provably cannot.

## Prior art before the first rung (sequence-model lineage)

The first rung — a shallow self-attention model — is itself the endpoint of a line of sequence
architectures. These precede the ladder; the fixed substrate below is the harness they are dropped into.

- **Recurrent encoder–decoder with attention (Bahdanau et al. 2014; Sutskever et al. 2014).** A
  recurrent net computes hidden states by the recurrence `h_t = f(h_{t-1}, x_t)`: a chain of `T`
  strictly serial steps. It is the natural finite-state simulator — its hidden state *is* a learned
  state register — but its `O(T)` sequential depth is the worst possible shape for parallel hardware,
  and on very long lags the through-time gradient product over `f'·w` factors vanishes or explodes.
  Gap: serial and slow, and the credit-assignment signal degrades with the lag.
- **Convolutional sequence models (ByteNet, Kalchbrenner et al. 2017; ConvS2S, Gehring et al. 2017).**
  Drop recurrence, compute all positions in parallel with stacked convolutions. This fixes the serial
  axis but reintroduces a path-length tax: a kernel of width `k` connects positions `D` apart only
  through a stack of `O(D/k)` (or `O(log_k D)` with dilation) layers. Gap: the work to relate two
  distant positions grows with the distance again.
- **Self-attention (Vaswani et al. 2017's lineage; Parikh et al. 2016 for attention-without-recurrence).**
  `softmax(QKᵀ/√d_k)V` routes every position to every other in one hop: `O(1)` sequential depth *and*
  `O(1)` path length, at `O(T²·d)` compute, which is the cheaper layer whenever `T < d`. But the
  operation is permutation-equivariant — order must be injected by a positional code — and crucially its
  *depth* is fixed by the number of stacked layers, so by the Liu et al. complexity argument a fixed
  shallow stack cannot simulate a non-solvable automaton. Gap: parallel and any-to-any, but constant
  depth is provably insufficient on the hard regime.

## The fixed substrate

A single training/evaluation driver is frozen and must not be touched. It exposes three synthetic
semiautomaton environments and an online stream of `(input, state)` sequences:

1. **`memory_unit`** — a `K`-state memory cell (`K=8`), alphabet `{noop, write(0), …, write(K−1)}`:
   `noop` keeps the state, `write(j)` jumps to state `j`. The transformation semigroup is the
   constant-function semigroup — one of the two Krohn–Rhodes "primes" that factorize any solvable
   semiautomaton. Depth-1 attention is theoretically sufficient. Sequence length 40, 8000 steps.
2. **`grid_world`** — `Grid_9`: states `{0,…,8}` on a line, alphabet `{L, R}`, reflecting (clamping)
   boundaries. Liu et al. 2022 Thm 3 gives an `O(1)`-depth, `O(1)`-embedding self-attention
   construction (the state is the clamped prefix sum of `±1` steps). Sequence length 40, 8000 steps.
3. **`random_dfa`** — a random transition table `δ : Q × Σ → Q` with `|Q|=60`, `|Σ|=8`, fixed per
   benchmark by a seed. On 60 states a random table typically generates a transformation semigroup
   containing `S_5` (non-solvable), so this environment lives in the regime where no constant-depth
   shortcut exists — the hardest of the three. Sequence length 40, 12000 steps.

The driver trains with per-token cross-entropy against the full state sequence `q_{1:T}`, online (a
fresh batch is sampled every step, so overfitting is not the challenge), batch size 64, gradient
clipping at 1.0, AdamW, and early stops after three consecutive log windows at training accuracy
≥0.999. The optimizer is always AdamW with exactly the four hyperparameters the editable interface
returns.

## The editable interface

Exactly one region of `pytorch-examples/synth_semiautomata/custom_strategy.py` is editable — the model,
the optimizer config, and the inference wrapper. Every method on the ladder is a fill of this same
contract:

- `build_model(env_spec, config) -> nn.Module` returns a module mapping token ids
  `input_ids ∈ {0,…,alphabet_size−1}^{B×T}` to per-position state logits of shape `[B, T, num_states]`.
- `get_optimizer_config(config) -> dict[str, float]` returns the four AdamW hyperparameters
  `lr, wd, beta1, beta2`.
- `forward_logits(model, input_ids, env_spec) -> torch.Tensor` is the inference wrapper (use it to host
  looped / scratchpad-style architectures that do not fit a plain `model(x)` call).

`env_spec` carries `alphabet_size`, `num_states`, and `seq_len`; the same architecture is built for all
three environments, differing only in those dimensions. The starting point is the scaffold default: a
shallow 1-layer GPT-2-style causal Transformer.

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
all three must be strong — a model that aces the two solvable environments but collapses on the
non-solvable `random_dfa` is dragged down hard. Per-environment wall-time is budgeted at ~30 minutes.
