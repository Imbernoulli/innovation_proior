**Problem.** The Grokfast hook made grokking *reliable* (`grok_rate 1.0` on every prime) and cut the
worst-case grok on `p=59` from "never" to 5500 steps, but every prime still spends 3000–6000 full-batch
steps memorizing before the held-out curve moves. Amplifying the slow part of a slow process shortens
the delay; it does not remove the memorize-first regime the transformer creates. The next move changes
the architecture so its natural minimum *is* the generalizing circuit.

**Key idea (a quadratic MLP whose minimum is the Fourier rule).** `(a+b) mod p` is a trigonometric
identity: placing residue `k` at angle `2πk/p`, `cos(θ_a+θ_b)` is *bilinear* in the sines/cosines of
`θ_a` and `θ_b`. A linear layer cannot form products of two inputs, but a **quadratic** activation can:
square a linear projection and the cross terms are exactly the `a`-feature × `b`-feature products the
addition formula needs. So use a two-layer fully-connected net, **no biases**: one-hot inputs
`one_hot(a) ‖ one_hot(b)` of dim `2p`, `Linear(2p, 256)`, `x → x²`, `Linear(256, p)`. A *sinusoidal*
first layer is a global optimum whose value depends on `(a,b)` only through `a+b mod p`, so it
generalizes to the held-out cells automatically — the architecture cannot easily memorize a per-cell
lookup, so the rule is the path of least resistance, not a late discovery.

**Why faster than the transformer.** For the transformer, memorization is easy and the Fourier circuit
is a late discovery that weight decay drags the trajectory toward (what Grokfast amplifies). For the
quadratic MLP, the Fourier solution is the low-loss region from the start, so there is almost no
memorize-then-grok delay to amplify in the first place.

**Hyperparameters / choices.** Hidden width `N=256`; both layers normal-init with
`std = sqrt(0.25)/(2N)^{1/3}` (mean-field scaling keeping post-square pre-activations `O(1)`); biases
off everywhere. Optimizer unchanged: AdamW `lr=1e-3`, `betas=(0.9, 0.98)`, `weight_decay=1.0`; no-op
hook. Note: the reference construction used MSE; the fixed loop keeps cross-entropy (outside the
editable block), so this runs the Gromov architecture under cross-entropy.

**What to watch.** Expect `mean_steps_to_grok` to drop well below Grokfast's 5500/3500/3167 toward the
low thousands the 500-step eval grid can resolve, `grok_rate 1.0` everywhere, `test_accuracy`/`score` at
a clean 1.0 including the hard `p=59` split, and a large wall-clock drop (the MLP is far cheaper than the
transformer). If steps-to-grok do not beat Grokfast, the architecture is not the bottleneck; if accuracy
falls below 1.0, suspect the cross-entropy-vs-MSE mismatch or the quadratic-activation conditioning.

```python
# EDITABLE region of custom_strategy.py — step 3: Gromov quadratic MLP + unchanged AdamW
class GromovMLP(nn.Module):
    """Two-layer MLP with quadratic activation, one-hot inputs.

    Following Gromov 2023 (arXiv:2301.02679):
      - Input: one-hot(a) || one-hot(b), shape [B, 2p]
      - Hidden: Linear(2p, N) -> x**2, with N=256
      - Output: Linear(N, p)
    """

    def __init__(self, p: int, hidden_width: int = 256):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, hidden_width, bias=False)
        self.fc2 = nn.Linear(hidden_width, p, bias=False)
        # Gromov-style mean-field init (matches d-doshi/Grokking utils/models.py)
        import math
        scale1 = (0.25 ** 0.5) / ((2 * hidden_width) ** (1 / 3))
        scale2 = (0.25 ** 0.5) / ((2 * hidden_width) ** (1 / 3))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=scale1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=scale2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2] long -> one-hot concat [B, 2p]
        a_oh = F.one_hot(x[:, 0], num_classes=self.p).float()
        b_oh = F.one_hot(x[:, 1], num_classes=self.p).float()
        h = torch.cat([a_oh, b_oh], dim=1)
        h = self.fc1(h)
        h = h * h  # quadratic activation (Gromov 2023)
        return self.fc2(h)


def build_model(p: int, config: TaskConfig) -> nn.Module:
    """Gromov 2023 MLP: 2-layer fcn, quadratic activation, hidden width N=256."""
    return GromovMLP(p=p, hidden_width=256)


def make_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """AdamW lr=1e-3 wd=1.0 (matches the rest of the suite)."""
    return torch.optim.AdamW(
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
