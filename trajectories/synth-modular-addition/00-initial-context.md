## Research question

On the modular-addition task `y = (a + b) mod p`, can the *grokking* delay — the gap between when a
network memorizes the training table and when it finally generalizes to the held-out cells — be
accelerated, or removed, by changing **only** the model architecture, the optimizer and its
hyperparameters, and an optional per-step gradient hook? Everything else is frozen: the dataset is the
full `Z_p × Z_p` table, the train/test split is a fixed `train_frac = 0.40` of the `p*p` cells, the
loss is cross-entropy, optimization is **full-batch**, and there is a hard budget of
`max_steps = 30000` per run with early stopping once held-out accuracy is sustained above `0.99`. The
single object being designed is the contents of the editable block — the model, the optimizer, and the
hook. The benchmark scores three primes (`p=59`, `p=97`, `p=113`) and the aggregate is the geometric
mean over them of `weighted_mean(score, test_accuracy)`; higher is better. Two derived diagnostics
decide which fill is better when accuracies tie: `grok_rate` (fraction of seeds that reach the
threshold inside the budget) and `mean_steps_to_grok` (lower is faster).

## Prior art before the first rung (the grokking lineage)

The first rung does not appear in a vacuum; it is the resolution of a short line of work on delayed
generalization in tiny algorithmic problems. These are the methods the ladder reacts to, each with the
gap that the next move is trying to close.

- **Delayed generalization on algorithmic tables (Power, Burns, Edwards, Babuschkin, Misra 2022,
  arXiv:2201.02177).** Train a small decoder-only transformer on an operation table — `a ∘ b` over a
  finite set — revealing a fraction of the cells and holding out the rest. Training accuracy saturates
  in `~10^3` steps; held-out accuracy stays at chance for one-to-three orders of magnitude longer, then
  climbs to near-perfect. The same paper's interventions sweep shows one lever dominates: AdamW weight
  decay more than halves the data (and time) needed to generalize, far more than any other knob. Gap:
  the *mechanism* and the *delay* are left as phenomena — generalization eventually happens, but slowly,
  and the paper does not give a cheap way to force it sooner.
- **Reverse-engineering the modular-addition circuit (Nanda, Chan, Lieberum, Smith, Steinhardt 2023,
  arXiv:2301.05217).** Takes a *one-layer* transformer on `a + b mod p` and shows the grokked network
  implements a discrete-Fourier algorithm: it embeds each residue on a circle, rotates by the operand
  angles in attention and the MLP, and reads off the angle of the sum. This fixes a concrete, minimal
  architecture (`d_model=128`, 4 heads, `d_mlp=512`, an explicit `=` token, full-batch AdamW with a
  short warmup) as the canonical grokking testbed. Gap: it explains *what* is learned but inherits the
  same long delay — the Fourier circuit only crystallizes late in training.
- **Plain MLPs also grok (Gromov 2023, arXiv:2301.02679).** Shows the transformer is not necessary: a
  two-layer fully-connected network with **no biases**, a **quadratic** activation, and **one-hot**
  inputs groks on the same modular task, and the grokked first-layer weights are clean sinusoids in the
  input index — the same Fourier structure, reached by a much simpler model. Gap: it is an architecture
  result, not yet a recipe for *speed* under a fixed budget.

## The fixed substrate

The training and evaluation driver is frozen and must not be touched. It builds the full `p*p` table of
`(a, b) → (a+b) mod p`, splits it once per top-level seed into a `0.40` train fraction and the rest as
test, and runs **full-batch** gradient descent: every optimizer step uses *all* training pairs. Each
step does `optimizer.zero_grad()`, a forward pass, `F.cross_entropy(logits, y)`, `loss.backward()`,
then `hook.post_grad(step)`, then `optimizer.step()`. Held-out accuracy is evaluated every 500 steps;
a run is declared "grokked" the first step its test accuracy is `≥ 0.99` over two consecutive eval
windows, at which point it early-stops. A run that never reaches the threshold records
`steps_to_grok = max_steps` and its latest metrics. A wall-clock safeguard stops a run gracefully
before the per-seed timeout. The model receives a `LongTensor[B, 2]` of `(a, b)` pairs in `[0, p)` and
must return `FloatTensor[B, p]` logits; `build_model` must not depend on the split.

## The editable interface

Exactly one region is editable — the block holding three things, and every rung on the ladder is a fill
of this same contract:

1. `build_model(p, config) -> nn.Module` — the architecture, `forward(x: Long[B,2]) -> Float[B,p]`,
   plus any helper `nn.Module` classes it needs;
2. `make_optimizer(model, config) -> torch.optim.Optimizer` — a standard optimizer;
3. `class TrainHook` with `post_grad(self, step)` — called after `loss.backward()` and before
   `optimizer.step()`, free to modify each parameter's `.grad`; the default returns `None` (no-op).

The starting point is the scaffold default: the Nanda one-layer transformer trained with full-batch
AdamW (`wd=1.0`, a 10-step linear warmup to `lr=1e-3`, `betas=(0.9, 0.98)`) and a no-op hook. Each rung
replaces exactly these definitions and nothing else.

```python
# EDITABLE region of custom_strategy.py — default fill (Nanda one-layer transformer, AdamW wd=1.0)
class GrokTransformer(nn.Module):
    """Nanda 2023 one-layer decoder-only transformer for modular addition."""

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.p = p
        self.vocab_size = p + 1
        self.eq_token = p
        self.tok_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=False),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=False),
        )
        self.unembed = nn.Linear(d_model, p, bias=False)
        self._init_paper_weights(d_model)

    def _init_paper_weights(self, d_model: int) -> None:
        hidden_std = 1.0 / math.sqrt(d_model)
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.attn.in_proj_weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.attn.out_proj.weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.mlp[0].weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=hidden_std)
        nn.init.normal_(self.unembed.weight, mean=0.0, std=1.0 / math.sqrt(self.p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long, device=x.device)
        seq = torch.cat([x, eq], dim=1)
        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, 3)
        h = self.tok_embed(seq) + self.pos_embed(pos)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = h + attn_out
        h = h + self.mlp(h)
        return self.unembed(h[:, -1, :])


def build_model(p: int, config: TaskConfig) -> nn.Module:
    return GrokTransformer(p=p, d_model=128, n_heads=4, d_mlp=512)


class WarmupAdamW(torch.optim.AdamW):
    """AdamW with Nanda's 10-step linear warmup to lr=1e-3."""

    def __init__(self, params, *, warmup_steps: int = 10, **kwargs):
        super().__init__(params, **kwargs)
        self.warmup_steps = warmup_steps
        self._step_count_for_warmup = 0
        self._base_lrs = [group["lr"] for group in self.param_groups]

    def step(self, closure: Callable | None = None):
        self._step_count_for_warmup += 1
        scale = min(self._step_count_for_warmup / self.warmup_steps, 1.0)
        for group, base_lr in zip(self.param_groups, self._base_lrs):
            group["lr"] = base_lr * scale
        return super().step(closure)


def make_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    return WarmupAdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)


class TrainHook:
    """No-op hook."""

    def __init__(self, model: nn.Module, config: TaskConfig):
        self.model = model
        self.config = config

    def post_grad(self, step: int) -> None:
        return None
```

## Evaluation settings

Three primes spanning a range of table sizes — `p=59`, `p=97`, `p=113` — each over the three seeds
`{42, 123, 456}` (one fixed train/test split per seed). The recorded per-prime metrics are
`test_accuracy` (held-out accuracy at the last step), `score` (best held-out accuracy at any eval
checkpoint), `grok_rate` (fraction of seeds reaching `test_acc ≥ 0.99` inside the budget), and
`mean_steps_to_grok` (mean step of first sustained grok, `= max_steps` if a seed never groks). The
aggregate task score is the geometric mean over the three primes of
`weighted_mean(score, test_accuracy)`. Higher is better; among fills that reach perfect accuracy, the
faster grok (lower `mean_steps_to_grok`, higher `grok_rate`) wins.
