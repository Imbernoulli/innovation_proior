**Problem.** On `y = (a + b) mod p`, a small transformer memorizes the revealed cells in `~10^3` steps
but only generalizes to the held-out cells far later — the *grokking* delay. The budget is fixed at
30000 full-batch steps. Before improving anything I need the canonical recipe's behavior as the floor.

**Key idea (the canonical fill).** Use the minimal model known to grok *and* to be interpretable: a
one-layer decoder-only transformer over the sequence `[a, b, =]` with `d_model=128`, 4 heads,
`d_mlp=512`, biases off, logits read from the `=` position. Train it with the single most reliable
grokking lever — decoupled weight decay — using AdamW `wd=1.0`, `betas=(0.9, 0.98)`, `lr=1e-3` with a
10-step linear warmup. Weight decay's steady pull toward low-norm/flat solutions is what moves the
trajectory off the sharp memorizing solution onto the generalizing Fourier circuit.

**Why this is the floor.** It is the exact configuration the phenomenon was characterized in, so it
should grok — but slowly. The empty `TrainHook` is deliberate: this rung adds no gradient-level
machinery, so its slow, budget-fragile grok is the weakness the next rung must fix.

**Hyperparameters.** `d_model=128`, `n_heads=4`, `d_mlp=512`; AdamW `lr=1e-3`, `betas=(0.9, 0.98)`,
`weight_decay=1.0`, 10-step linear warmup; no-op hook.

**What to watch.** Expect `train_acc=1.0` very early and a late, *variable* grok: large
`mean_steps_to_grok` and a `grok_rate < 1` on at least the smallest-table prime (a seed running out of
budget), which drags that prime's `score`/`test_accuracy` below a clean 1.0. That delay is what step 2
attacks with the gradient hook.

```python
# EDITABLE region of custom_strategy.py — step 1: Nanda one-layer transformer + AdamW(wd=1.0)
class GrokTransformer(nn.Module):
    """Nanda 2023 one-layer decoder-only transformer."""

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
    """Nanda 2023: one-layer transformer d_model=128 n_heads=4 d_mlp=512."""
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
    """Power 2022 default setting: AdamW lr=1e-3 wd=1.0 betas=(0.9, 0.98)."""
    return WarmupAdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=1.0,
    )


class TrainHook:
    """No-op hook."""

    def __init__(self, model: nn.Module, config: TaskConfig):
        self.model = model
        self.config = config

    def post_grad(self, step: int) -> None:
        return None
```
