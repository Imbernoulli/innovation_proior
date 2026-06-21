The object I have to design is narrow: a single `nn.Module` that reads the whole symbol stream $\sigma_{1:T}$ and emits the whole state trajectory $q_{1:T}$, one per-position prediction, scored by per-token cross-entropy against the true states. Everything else — the three environments, the online stream, AdamW, the early stop — is fixed. The interesting part is that the three environments straddle a complexity wall that Liu et al. 2022 made precise: a depth-$O(\log T)$ attention stack can simulate *any* semiautomaton, a *constant* depth suffices for the *solvable* ones via Krohn–Rhodes factorization, but the *non-solvable* ones — the smallest generating $A_5$ — provably need more than constant depth unless $\mathrm{TC}^0 = \mathrm{NC}^1$. So the question is not "which architecture wins" but where each architecture sits relative to that wall. The cleanest way to learn that empirically is to start at the architecture whose *failure* is most diagnostic, not the one I expect to win.

I propose to begin with the shallowest credible attention model: a depth-1 causal Transformer, exactly the scaffold default. The point is that a one-layer model is the most constant-depth thing there is, so its failure pattern *is* the complexity boundary made visible. Because the harness trains online — a fresh batch every step, so there is no overfitting confound — whatever this model cannot reach, it cannot reach because a constant-depth shortcut does not exist for it, not because of a data or regularization artifact. That makes the depth-1 probe a clean capacity measurement, and the measurement is what tells the rest of the ladder what to do.

Let me justify each piece against the task. The input is a sequence of symbol ids; I look each up in a learned token embedding of width $d_{\text{model}} = 128$. But self-attention, $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$, is built entirely from dot products and weighted sums over the *set* of positions — it is permutation-equivariant and has no term that knows which position is which. For a semiautomaton that is fatal: the state at step $t$ is the *ordered* application of $\sigma_1,\dots,\sigma_t$, and the same multiset of symbols in a different order generally gives a different state, because group multiplication does not commute. So order must be injected. I use a learned absolute position embedding, one vector per index $0,\dots,\text{seq\_len}-1$, *added* to the token embedding. With sequence length fixed at 40 and no extrapolation required, a learned table is fully adequate and is the most flexible choice — the model learns whatever position relation it needs rather than being handed a sinusoidal prior. Adding rather than concatenating keeps the downstream matrices at width $d_{\text{model}}$; the first linear can already split content and position into separate subspaces, so concatenation would buy nothing but width.

The load-bearing structural choice is the causal mask. I register an upper-triangular boolean mask and pass `is_causal=True`, so position $t$ attends only to positions $\le t$. This is not a generation convenience — it is the semiautomaton's own definition. The true state $q_t$ is a function of $\sigma_{1:t}$ and *nothing after*; a symbol arriving at step $t+3$ cannot influence $q_t$. Letting attention see the future would hand the model information the real transition function never has, and it could fit spurious future correlations on the training stream. Masking to the causal prefix bakes in $q_t = f(\sigma_{1:t})$, which is exactly true, and strictly shrinks the hypothesis space to the correct one.

The block uses pre-norm ($x + \text{Sublayer}(\mathrm{LayerNorm}(x))$) with a GELU MLP of width $4\,d_{\text{model}}$. Pre-norm keeps a clean identity path for the gradient and trains shallow and deep stacks comparably without warmup gymnastics — which matters because the ladder will sweep *depth* and I want the optimization geometry to stay comparable as it does. The MLP is the only per-position nonlinear compute in the layer (attention mixes *across* positions but is otherwise a weighted linear combination of values), so it needs enough width to carve the per-symbol features the head reads; $4\times$ is the usual capacity knee. A final linear head maps $d_{\text{model}} \to \text{num\_states}$, giving the per-position logits the harness scores. So the forward pass is: embed token + position, one causal self-attention + MLP block, project to state logits — $[B,T] \to [B,T,\text{num\_states}]$. The optimizer is AdamW with $\text{lr}=3\mathrm{e}{-4}$, $\text{wd}=1\mathrm{e}{-4}$, $\beta_1=0.9$, $\beta_2=0.999$, the standard GPT-2-style recipe; the harness supplies gradient clipping at 1.0.

What I expect to read back is an *ordering*, and that ordering is the falsifiable claim. On `memory_unit`, the current state is entirely the most recent `write` in the prefix — a single head that attends from $t$ back to the last non-`noop` symbol and copies its target is an exact constant-depth solution, so I expect $\approx 1.0$; if even this fails, it is an optimization bug, not a wall. On `grid_world`, the state is the clamped prefix sum of $\pm 1$ steps; the prefix sum is a clean constant-depth attention computation, but the wall-clamping saturation interacts across the whole prefix in a way one mixing step may render only approximately, so I expect it high but possibly short of 1.0 — and that shortfall would be the first sign that one mixing step is too little composition. On `random_dfa`, a random $\delta$ on 60 states almost surely generates an $S_5$-containing non-solvable semigroup, for which no constant-depth shortcut exists, so I expect a depth-1 model to land poorly — in the low double digits, the analog of the depth-1 collapse in the non-solvable regime, perhaps a little above pure chance because 40 short steps and a fixed table leave some short-range structure to exploit. Because the aggregate is the geometric mean, that third number dominates the score regardless of how clean the first two are, and it is precisely the quantity the rest of the ladder must move.

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
