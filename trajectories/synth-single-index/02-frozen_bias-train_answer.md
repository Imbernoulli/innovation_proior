Vanilla SGD gave me the split in numbers, exactly as the gradient analysis predicted. On `relu-d100` all three seeds pinned at `direction_recovery` $\approx 0.998$ with `test_mse` $\approx 0.0017$ — the easy $k=1$ regime where ordinary SGD already finds the direction. The two hard links are where the floor showed: `hermite-d100` ($k=3$) averaged $0.656$ recovery with `test_mse` stuck at $0.16$, and `sign-d100` ($k=1$, non-smooth) collapsed to a mean of $0.210$ with per-seed values $\{0.049, 0.156, 0.426\}$ — barely above chance on two of three seeds. That seed-to-seed spread is the tell: the outcome depends on whether a run's random rows happened to carry enough early overlap, which is what you see when the landscape has no benign structure pulling every run to the same place. This is not a learning-rate problem — the same SGD that nails ReLU cannot move the rows on the hard links — and the diagnosis is two coupled failures: the first layer and the head are trained jointly, so the high-dimensional direction search and the one-dimensional link fit interfere. The fix has to be structural.

I propose the **frozen-bias shallow network**: split the two jobs by *what is trainable*. The structural observation is that a hidden neuron $\mathrm{ReLU}(\langle w_j, x\rangle + b_j)$ is parameterised by two things with two different roles in a single-index model. The non-parametric job — approximating the univariate link $g(u)$ for $u = \langle\theta^*, x\rangle$ — is exactly the job of a *spread of thresholds*: a bank $\{\mathrm{ReLU}(u - b_j)\}$ with varied $b_j$ is a one-dimensional spline basis that can fit any reasonable $g$. That is a kernel / random-feature job, and the biases are its sampling. The high-dimensional job — finding $\theta^*$ — is the job of the *directions* $w_j$. The lazy-versus-rich lesson (Chizat, Oyallon & Bach 2019) is that the part that stays at init and acts like a fixed kernel is the lazy part, while the part that moves and learns features is the rich part; the link fit should be lazy and the direction search rich. So the one essential move is to **freeze the biases at a wide random init and train only the directions and the head**.

What makes this collapse the landscape — and this is the justification, not an analogy — is the following. Write the population loss $L = \mathbb{E}[(G(x) - y)^2]$ for $G(x) = \sum_j a_j\,\mathrm{ReLU}(\langle w_j, x\rangle + b_j)$ and decompose it against the Gaussian in Hermite polynomials. The key identity is that a degree-$p$ feature along direction $w$ overlaps the target's degree-$p$ component by exactly $\langle w, \theta^*\rangle^p$. If the biases $b_j$ are frozen, they do not depend on the directions, so the *only* place a row $w_j$ enters the loss is through the scalar overlap $m_j = \langle w_j, \theta^*\rangle$. The $d$-dimensional dependence on each row collapses onto a single number, and $\nabla_{w_j} L$ becomes colinear with $\theta^*$ — it can only push $m_j$ up or down, i.e. rotate the row toward or away from the truth. The high-dimensional search becomes a scalar flow in the overlaps. In the ideal limit (infinitely many unregularised random thresholds) the projected loss is strictly decreasing in $|m|$ with only two kinds of critical point — the equator $m=0$ and the poles $m=\pm 1$, no spurious minima, so gradient flow slides from a random start up to a pole. With finitely many *frozen* random biases this picture survives as long as the random-feature approximation of $g$ is good, which needs the bias spread to be $O(1)$ — wide enough to span the range of $u = \langle\theta^*, x\rangle \sim N(0,1)$. Under vanilla SGD the biases were *also* trained, which dragged the non-parametric problem back into the high-dimensional dynamics and re-entangled the two jobs; freezing them is exactly what undoes that.

I have to be honest about how much of the clean theory the fixed harness lets me keep. The cleanest analysis uses a *tied* architecture (one shared inner direction, neurons differing only in bias and sign), a time-scale separation (search the direction first with a small/sparse readout, then turn the head on), and a final fresh-sample ridge refit. The harness exposes none of that: the model is a fixed *wide* $\text{Linear}(d, 256)$ with all rows independent, I get mini-batches rather than a controllable two-phase schedule, and the closed-form refit is a separate move I am deliberately holding for the next rung. So I keep only the one essential move — freeze the biases — and apply it to the standard wide MLP, accepting that many independent rows are a noisier realisation of the collapsed landscape than the tied net would be. Concretely: initialise the first-layer rows on the unit sphere (rather than Kaiming) so every row is a clean, comparable direction-only probe with overlap $\sim 1/\sqrt{d}$ — since the optimiser now controls only the *direction* of each row, Kaiming's $\sqrt{2/d}$ scale would let row norms drift and muddy the direction-only reading; draw the biases $\sim \mathrm{Uniform}(-1, 1)$ for the $O(1)$ threshold spread and freeze them with `requires_grad_(False)`; keep the small uniform readout; and build the optimiser from `[p for p in net.parameters() if p.requires_grad]` so the frozen biases are excluded automatically. The same SGD-with-momentum mean-squared-error mini-batch loop runs unchanged, and `finalize` stays a no-op.

The expectations are asymmetric, and that asymmetry is itself the test of whether the collapse is real. On `relu-d100` I expect no change — the easy $k=1$ regime saturates at $\approx 0.998$ regardless of bias treatment; a drop here would mean the unit-sphere init or the freeze is hurting the easy case. On `sign-d100` I expect the clearest gain: the link is $k=1$ so the direction signal is present at first order, and the failure under vanilla SGD was the rough entangled landscape, so collapsing it should raise recovery well above $0.210$ and, importantly, *tighten* the seed-to-seed spread as the benign landscape removes the dependence on lucky init. On `hermite-d100` I expect roughly flat: the freeze cleans the landscape but does not aggregate the weak $\sim d^{-1}$ third-order signal, which still sits below the $256$-sample noise. If hermite stays stuck near $0.656$, that is consistent with the diagnosis — the missing ingredient is not landscape shape but signal aggregation, which points straight at the next rung: stop crawling with noisy $256$-sample steps and take one giant full-batch step that sums the signal over all $n_{\text{train}}$ samples, paired with the closed-form readout refit the `finalize` hook is waiting for.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — step 2: frozen-bias shallow net
class Strategy:
    """Frozen-bias shallow network: biases sampled once and never trained."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Random first-layer rows on the unit sphere; biases sampled uniformly
        # in [-1, 1] and FROZEN -- the key move.
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            net.fc1.bias.uniform_(-1.0, 1.0)
        net.fc1.bias.requires_grad_(False)

        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        params = [p for p in net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def training_step(
        self,
        net: TwoLayerMLP,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        config: TaskConfig,
    ) -> StepMetrics:
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return StepMetrics(loss=float(loss.item()), extra={})

    def finalize(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        config: TaskConfig,
    ) -> None:
        return


def build_strategy(config: TaskConfig) -> Strategy:
    return Strategy(config)
```
