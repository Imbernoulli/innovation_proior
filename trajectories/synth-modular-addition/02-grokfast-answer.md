**Problem.** The default recipe groks, but slowly and fragilely: on `p=59` one seed never grokked
inside the 30000-step budget (`grok_rate 0.667`, `mean_steps_to_grok 15167`), and even on the primes
that always grok the late jump costs thousands of full-batch steps. The delay itself is the enemy, and
it must be cut without touching the architecture, the loss, or the optimizer — only the gradient hook.

**Key idea (amplify slow gradients).** Read each parameter's gradient sequence `g(t)` over steps as a
discrete-time signal whose motion mixes a *fast* component (memorization/overfitting) and a *slow*
component (the delayed generalization). "Slow" is low frequency. Take a low-pass-filtered copy of the
gradient and add it back: `ĝ(t) = g(t) + h(t) * g(t)`, i.e. `Ĝ(ω) = (1 + H(ω)) G(ω)` — a high-boost of
the slow part (gain large at low `ω`, ≈1 at high `ω`, so the fast part is *kept*, not denoised away).
Use a one-pole EMA for `h` so the state is one buffer per parameter: `μ ← α μ + (1−α) g`, then
`ĝ = g + λ μ`. Its transfer function `λ(1−α)/(1−α e^{-iω})` gives low-frequency gain `1+λ` and near-unit
high-frequency gain, with `α` the cutoff and `λ` the gain.

**Why it lands in the hook.** For any linear first-order optimizer, filtering the gradient by `h` is
provably equivalent to filtering the *update* by `h` (the optimizer's transfer function cancels), so the
boost on the parameter motion can be applied to `p.grad` after `backward()` and before `step()` —
exactly `TrainHook.post_grad`. The architecture and `WarmupAdamW(wd=1.0)` are left untouched so the rung
isolates the EMA hook; weight decay's pull is still present, now with its slow trajectory amplified.

**Hyperparameters.** Same model and optimizer as step 1. Hook: `alpha=0.98` (≈50-step memory),
`lamb=2.0` (low-frequency gain `1+λ=3`); buffer seeded from the first gradient seen.

**What to watch.** On `p=59`, the budget-failed seed should now grok (`grok_rate → 1.0`,
`test_accuracy → 1.0`) and `mean_steps_to_grok` should fall far below 15167; on `p=97`/`p=113`,
accuracies stay 1.0 while steps-to-grok drop below 2833/2333. If steps-to-grok do not improve the
slow-gradient hypothesis is wrong; if accuracy falls, `lamb` is drowning the fast component.

```python
# EDITABLE region of custom_strategy.py — step 2: default model/optimizer + Grokfast-EMA hook
class GrokTransformer(nn.Module):
    """One-layer decoder-only transformer (same as Nanda 2023)."""

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
        self._init_baseline_weights(d_model)

    def _init_baseline_weights(self, d_model: int) -> None:
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
    """AdamW lr=1e-3 wd=1.0 (Grokfast keeps the baseline optimizer untouched)."""
    return WarmupAdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=1.0,
    )


class TrainHook:
    """Grokfast-EMA gradient filter.

    Maintains an EMA of each parameter's gradient and adds `lamb * EMA` to
    the raw gradient before opt.step().
    """

    def __init__(self, model: nn.Module, config: TaskConfig):
        self.model = model
        self.config = config
        self.alpha = 0.98
        self.lamb = 2.0
        self.grads: dict[str, torch.Tensor] = {}

    def post_grad(self, step: int) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            if name not in self.grads:
                self.grads[name] = param.grad.detach().clone()
            else:
                self.grads[name].mul_(self.alpha).add_(param.grad.detach(), alpha=1.0 - self.alpha)
            param.grad.add_(self.grads[name], alpha=self.lamb)
```
