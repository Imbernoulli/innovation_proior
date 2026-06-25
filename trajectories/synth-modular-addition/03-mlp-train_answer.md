The Grokfast hook did exactly what the slow-gradient hypothesis predicted. On `p=59` the seed that had run out of budget under the default — 456, stuck at `test_accuracy 0.982` with `grok_rate 0` — now groks at 5500 steps, and the prime's grok rate is a full 1.0 with mean steps-to-grok down from 15167 to 5500. On `p=97` the mean went from 2833 to 3500 and on `p=113` from 2333 to 3167 — essentially flat, every seed still grokking. So amplifying the slow component bought the thing that mattered most, reliability on the hard split, and pulled the worst-case grok in by nearly 3x. But it did *not* make grokking dramatically faster across the board: I am still spending three-to-six thousand full-batch steps memorizing before the held-out curve moves at all. Grokfast accelerates the trajectory; it does not remove the regime that produces it. To *kill* the delay rather than shorten it, I should ask whether the process has to be slow in the first place.

Why is it slow? The whole grokking story rests on the model first finding a sharp, high-norm memorizing solution and only later, under weight decay's pull, drifting onto a low-norm rule-like solution. The transformer makes this two-phase story almost inevitable: with an embedding table, attention, an MLP, and an unembedding it has enormous freedom to fit the revealed cells by interpolation before any of them assembles the actual circuit, so it memorizes first and discovers the rule late. Grokfast leans on the late discovery; it does not change the fact that the model is *built* to memorize first. The grokked one-layer transformer is known to implement a discrete-Fourier circuit — embed each residue on a circle, rotate by the operand angles, read off the angle of the sum. If that Fourier structure is the only thing that generalizes here, a far simpler model whose natural solution *is* that structure should find it much sooner. That is the move: keep the optimizer and weight decay that are working, and change the architecture to one whose inductive bias points straight at the modular circuit.

I propose the **quadratic MLP**: a two-layer fully-connected network with a *quadratic* activation, no biases, and one-hot inputs. The reason it has the right shape comes straight from the cosine addition formula,

$$\cos(\theta_a + \theta_b) = \cos\theta_a\cos\theta_b - \sin\theta_a\sin\theta_b.$$

Place residue $k$ at angle $\theta_k = 2\pi k/p$; then $(a+b)\bmod p$ is the residue whose angle is $\theta_a+\theta_b$, and that angle's cosine and sine are *bilinear* in the cosines and sines of $\theta_a$ and $\theta_b$ — products of one feature of $a$ with one feature of $b$. A linear layer cannot form a product of two inputs; it can only add them. A quadratic nonlinearity can: square a linear projection $z = \sum w\cdot(\text{features of }a,b)$ and $z^2$ contains cross terms, products of an $a$-feature with a $b$-feature, which is precisely the bilinear ingredient the addition formula needs. So a two-layer net with a quadratic activation expresses the modular-addition circuit in closed form, without any of the transformer's machinery and without having to discover attention patterns first.

Concretely as a fill of the scaffold: the model takes `x: Long[B, 2]`, represents each operand as a one-hot vector over the `p` residues, and concatenates them into `one_hot(a) ‖ one_hot(b)` of dimension $2p$. One-hot is the right encoding precisely because I do *not* want to leak numeric structure — the model should learn the circle embedding itself — and with a one-hot input the first linear layer's columns *are* a learnable per-residue embedding. Then `Linear(2p, N)` with no bias and hidden width $N=256$, the activation $x\mapsto x^2$ applied elementwise, and `Linear(N, p)` with no bias for the logits. No biases anywhere: biases would let the network fit constant per-class offsets that have nothing to do with the additive structure, and the circuit I am aiming for is purely multiplicative-then-additive, so biases are pure memorization capacity I do not want. Two matmuls, one square, no attention, no positional embeddings, no `=` token.

Why this network generalizes almost as soon as it fits is in what the quadratic layer represents. The $j$-th hidden pre-activation on the one-hot input is $z_j = u_j[a] + v_j[b]$, where $u_j[a]$ is the learned scalar hidden unit $j$ assigns to operand value $a$ and $v_j[b]$ likewise for $b$. Squaring gives

$$z_j^2 = u_j[a]^2 + 2\,u_j[a]\,v_j[b] + v_j[b]^2,$$

and the cross term $2\,u_j[a]\,v_j[b]$ is exactly the $a$-feature × $b$-feature product. Now suppose the learned per-value features are sinusoidal, $u_j[a] = \cos(2\pi f_j a/p + \varphi_j)$ for some frequency $f_j$. By the product-to-sum identity, $u_j[a]\,v_j[b]$ becomes a sum of $\cos(2\pi f_j(a+b)/p + \cdots)$ and $\cos(2\pi f_j(a-b)/p + \cdots)$ — terms that depend on $a$ and $b$ only through $a+b$ (and $a-b$). A handful of such frequency channels, linearly combined by the second layer, can reconstruct an indicator that peaks at the residue $c=(a+b)\bmod p$. So a *sinusoidal* setting of the first-layer weights is a global optimum of this architecture, and its value at any $(a,b)$ depends only on $a+b \bmod p$ — which is to say it generalizes to the held-out cells *automatically*, since those cells share the same $a+b$ structure as the training cells. The architecture's natural minimum is the generalizing solution.

That is the crucial difference from the transformer. For the transformer the memorizing solution is easy to reach and the Fourier solution is a late discovery that weight decay (amplified by Grokfast) drags the trajectory toward over thousands of steps. For the quadratic MLP the Fourier solution *is* the low-loss region the optimizer is heading toward from the start, because the architecture cannot easily express a per-cell lookup — squaring a low-dimensional linear projection has very limited capacity to memorize 40% of a $p\times p$ table cell-by-cell, but ample capacity to express a few sinusoidal frequency channels. Memorization is not the path of least resistance; the rule is, which should compress the memorize-then-grok delay to almost nothing.

The initialization matters more than usual, because a quadratic activation is sensitive to the scale of its input — square a too-large pre-activation and the gradients explode, square a too-small one and the signal vanishes. I want the first-layer outputs to start at order one so the square is well-conditioned, so I set both layers to normal weights with standard deviation $\sqrt{0.25}\,/\,(2N)^{1/3}$, a mean-field scaling that accounts for the cubic interaction the quadratic activation induces between the two layers and the width $N$, keeping pre-activations $O(1)$ at init; this is the standard init for this exact architecture, the regime in which it is observed to grok cleanly. For the optimizer I keep what is already working — AdamW with `lr=1e-3`, `betas=(0.9, 0.98)`, `weight_decay=1.0` — and deliberately do not change it, so the only thing that moves from the Grokfast rung is the architecture. I also drop the gradient hook, since the whole premise now is that the architecture, not a gradient trick, removes the delay. Weight decay still helps as hygiene: it keeps the second-layer combination sparse in frequency and cleans up spurious channels, which is why the learned weights come out as clean sinusoids — but I am no longer relying on it to *move* the trajectory across a memorize-to-generalize gap.

One honest mismatch with the architecture's most natural regime: the quadratic-MLP circuit is cleanest under a mean-squared-error objective on one-hot targets, where the cleanest sinusoids appear, while the benchmark's loop is fixed to cross-entropy, which lives outside the editable block. So I am running this architecture under cross-entropy rather than its native MSE. This should be fine and arguably easier for classification accuracy — cross-entropy with the same bilinear logits still has the Fourier solution as a minimizer, since the softmax of an indicator-shaped logit peaks at the right class — but it is a departure from the MSE regime, and if anything looks off (e.g. less clean frequency structure) the loss mismatch is the first place to look.

The falsifiable expectations against the Grokfast numbers are sharp. Grokfast reached `grok_rate 1.0` on every prime but still at 5500 / 3500 / 3167 mean steps-to-grok. If the architectural story is right — that this network's natural minimum *is* the generalizing circuit, so there is almost no memorize-first detour — then the quadratic MLP should grok far sooner on every prime, with mean steps-to-grok dropping to roughly the smallest the 500-step eval grid can resolve, `grok_rate 1.0` everywhere, and `test_accuracy`/`score` at a clean 1.0 including the hard `p=59` split. I would also expect the wall-clock per run to collapse, since the MLP is far cheaper than the transformer per step. If steps-to-grok do not beat Grokfast's, the claim that the architecture removes the delay is wrong and the transformer's late-discovery dynamics were not the bottleneck; if accuracy comes in below 1.0, the cross-entropy-vs-MSE mismatch or the quadratic-activation conditioning is the suspect.

```python
# EDITABLE region of custom_strategy.py — step 3: quadratic MLP + unchanged AdamW
class QuadraticMLP(nn.Module):
    """Two-layer MLP with quadratic activation, one-hot inputs.

      - Input: one-hot(a) || one-hot(b), shape [B, 2p]
      - Hidden: Linear(2p, N) -> x**2, with N=256
      - Output: Linear(N, p)
    """

    def __init__(self, p: int, hidden_width: int = 256):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, hidden_width, bias=False)
        self.fc2 = nn.Linear(hidden_width, p, bias=False)
        # mean-field init keeping pre-activations O(1) for the quadratic activation
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
        h = h * h  # quadratic activation
        return self.fc2(h)


def build_model(p: int, config: TaskConfig) -> nn.Module:
    """Quadratic MLP: 2-layer fcn, quadratic activation, hidden width N=256."""
    return QuadraticMLP(p=p, hidden_width=256)


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
