The thing I get to design here is small — an architecture, an optimizer, and an optional gradient hook — but the question is sharp: on `y = (a + b) mod p`, can I make the *grokking* delay short, or kill it, under a fixed 30000-step full-batch budget. A small transformer fits the revealed cells in about `10^3` steps and only generalizes to the held-out cells far later, and the aggregate is dragged down by whichever seed tips over latest. Before I try to be clever I need a reference point, because the canonical setup is the one the whole phenomenon was characterized in, and its failure mode is what every later move has to react to. So the first rung is not a guess; it is the floor I read the numbers off.

What I propose for that floor is the **Nanda one-layer transformer trained with full-batch AdamW at `weight_decay = 1.0`** — the minimal model known both to grok and, once it does, to be interpretable. Each of the `p` residues is a token, plus one extra `=` token, so the vocabulary is `p + 1`. The model sees the three-token sequence `[a, b, =]`, embeds each token, adds a learned positional embedding for the three slots, and runs a single multi-head self-attention block and a single MLP block (`d_model → d_mlp → d_model` with a ReLU), each as a residual addition. The logits over the `p` classes are read off the *last* position — the `=` slot. I take `d_model = 128`, `4` heads, `d_mlp = 512`, biases off throughout, and a plain linear unembedding to `p` classes.

Reading off the `=` slot is deliberate: it is the only position whose attention can look back over both operand tokens and combine them, so reading the logits there mirrors how a decoder predicts the token after the prompt — one-step prediction of the result from the query `[a, b, =]`. Pinning *one* layer at exactly these widths, rather than something bigger or deeper, is what keeps the memorize-then-generalize transition clean: a one-layer model is the smallest thing that still grokks, so both phases stay visible and the grokked solution is simple enough to read out. And it is worth being concrete about what that grokked solution is, because it is the target every later rung will try to reach faster. Once this model generalizes it does not implement a lookup table; it implements a discrete-Fourier circuit. It embeds each residue $k$ at an angle $\theta_k = 2\pi k/p$ on a circle in embedding space, the attention and MLP rotate those angles so the operand angles add, and the unembedding reads off which residue's angle matches $\theta_a + \theta_b$ — a closed-form trigonometric computation of $(a+b)\bmod p$ rather than a memorized association. That the generalizing solution is this Fourier circuit is exactly why one layer suffices: assembling it needs only to compose two operand embeddings and rotate, which a single attention-plus-MLP block can do. The initialization matters too — every hidden-width matrix is normal with standard deviation $1/\sqrt{d_{\text{model}}}$ and the unembedding $1/\sqrt{p}$, matching the released reference network, so I am reproducing the regime in which grokking was characterized and not some accidentally different basin where the timing could differ.

The optimizer is where the literature is loudest, and the choice is not free. The single most reliable lever on this task is decoupled weight decay: in the original intervention sweep (Power et al. 2022), adding weight decay more than halves the amount of data — and equivalently here, the amount of optimization time — needed to cross from memorization to generalization, far more than any other knob. The mechanistic reading is consistent: the memorizing, table-lookup solution that fits the revealed cells is a high-norm, sharp interpolation, while the rule-like solution that also answers the held-out cells is lower-norm and flatter. Weight decay is a steady pressure toward low norm, so over a long run it pushes the trajectory off the memorizing solution and onto the generalizing one — that pressure is the *reason* the late jump happens at all under a tight budget. So I use AdamW — decoupled decay, not L2 folded into the gradient — with `weight_decay = 1.0` (the strong setting the phenomenon was characterized at), `betas = (0.9, 0.98)`, and `lr = 1e-3`. One more reference detail is a 10-step linear warmup of the learning rate from zero to `1e-3`: with full-batch updates on a fresh model the first few gradients can be large, and the warmup keeps the very first steps from kicking the weights somewhere bad. I implement it as a thin `AdamW` subclass that scales each group's learning rate by $\min(\text{step}/10,\,1)$ before each `step()`.

The hook stays empty. There is nothing to filter or amplify yet — the whole purpose of this rung is to establish how the *plain* canonical recipe behaves before any gradient-level machinery, so `post_grad` returns `None`. What I expect from this floor is also its weakness: it will grok on all three primes within the budget, but late and with variable timing. Training accuracy will hit `1.0` in well under a thousand steps; then for a long stretch the held-out accuracy sits far below threshold while weight decay slowly grinds toward the flat Fourier solution. Because the tipping step depends on the random split and init, the seeds will grok at noticeably different steps, and on the smallest-table prime — where the late jump sits closest to the 30000-step ceiling — I expect at least one seed to run out of budget and not grok at all, pulling that prime's `score` and `test_accuracy` below a clean `1.0` and its `grok_rate` below one. That slow, budget-fragile grok is precisely the delay the next rung must attack, and the empty hook is the place to attack it.

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
