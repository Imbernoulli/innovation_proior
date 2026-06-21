## Research question

On the modular-addition task `y = (a + b) mod p`, can the *grokking* delay — the gap between when a network memorizes the training table and when it generalizes to the held-out cells — be shortened or removed by changing **only** the model architecture, the optimizer and its hyperparameters, and an optional per-step gradient hook? Everything else is frozen: the full `Z_p × Z_p` table, a fixed `train_frac = 0.40` split, cross-entropy loss, **full-batch** training, and a hard budget of `max_steps = 30000` with early stopping once held-out accuracy sustains `≥ 0.99`. The design object is the contents of the editable block. The benchmark scores three primes (`p = 59`, `p = 97`, `p = 113`); the aggregate is the geometric mean over them of `weighted_mean(score, test_accuracy)`. Higher is better. If accuracies tie, prefer higher `grok_rate` and lower `mean_steps_to_grok`.

## Prior art / Background / Baselines

- **Power et al. (2022).** Train a small decoder-only transformer on held-out algorithmic operation tables; it first memorizes, then after a long plateau suddenly generalizes. Their intervention sweep finds AdamW weight decay is the strongest known knob for reducing the delay. Gap: generalization still takes a long time, and no cheap intervention reliably forces it inside a fixed step budget.
- **Nanda et al. (2023).** Reverse-engineer a one-layer transformer on modular addition and identify the grokked solution as a discrete-Fourier circuit; they fix a minimal canonical architecture (`d_model = 128`, 4 heads, `d_mlp = 512`, an explicit `=` token, full-batch AdamW with warmup). Gap: the analysis describes the final circuit but does not remove the long training delay.
- **Gromov (2023).** Shows a two-layer MLP with no biases, quadratic activation, and one-hot inputs also groks on modular addition, and its learned weights encode the same Fourier structure. Gap: this is an architectural existence result, not a recipe for fast, reliable generalization under a fixed budget.

## Fixed substrate / Code framework

The training and evaluation driver is frozen. It builds the full `p*p` table of `(a, b) → (a+b) mod p`, splits it once per seed into a `0.40` train fraction and the rest as test, and runs **full-batch** gradient descent: every step does `optimizer.zero_grad()`, forward pass, `F.cross_entropy(logits, y)`, `loss.backward()`, `hook.post_grad(step)`, then `optimizer.step()`. Held-out accuracy is evaluated every 500 steps; a run is declared "grokked" the first step its test accuracy is `≥ 0.99` over two consecutive windows, then early-stops. A run that never reaches the threshold records `steps_to_grok = max_steps`. The model receives `LongTensor[B, 2]` of `(a, b)` pairs in `[0, p)` and must return `FloatTensor[B, p]` logits; `build_model` must not depend on the split.

## Editable interface

Exactly one region is editable — the block holding three things:

1. `build_model(p, config) -> nn.Module` — the architecture, `forward(x: Long[B,2]) -> Float[B,p]`, plus any helper `nn.Module` classes it needs;
2. `make_optimizer(model, config) -> torch.optim.Optimizer` — a standard optimizer;
3. `class TrainHook` with `post_grad(self, step)` — called after `loss.backward()` and before `optimizer.step()`, free to modify each parameter's `.grad`; the default returns `None` (no-op).

The starting point is the scaffold default: the Nanda one-layer transformer trained with full-batch AdamW (`wd=1.0`, a 10-step linear warmup to `lr=1e-3`, `betas=(0.9, 0.98)`) and a no-op hook.

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

Three primes — `p = 59`, `p = 97`, `p = 113` — each over seeds `{42, 123, 456}` with one fixed train/test split per seed. The per-prime metrics are `test_accuracy` at the last step, `score` (best held-out accuracy at any checkpoint), `grok_rate` (fraction of seeds reaching `test_acc ≥ 0.99` inside the budget), and `mean_steps_to_grok` (mean step of first sustained grok, `= max_steps` if a seed never groks). The aggregate task score is the geometric mean over the three primes of `weighted_mean(score, test_accuracy)`. Higher is better; among fills that reach perfect accuracy, lower `mean_steps_to_grok` and higher `grok_rate` win.
