The task is to fill the four empty callbacks of a fixed two-layer ReLU MLP so that the network recovers the hidden direction $\theta^*$ of a single-index target $y = g(\langle\theta^*, x\rangle) + \text{noise}$, across links of growing difficulty. Before reaching for anything clever I want the honest floor: what does plain gradient descent, left to its own devices, actually do here? Every later method has to beat this control, so I make the most ordinary fill of all four slots and study exactly what that ordinary choice can and cannot do. The prior art frames the stakes — random-feature and NTK models freeze the first layer and so can never move their features toward $\theta^*$; teacher-student analyses assume the link $g$ is already known; the information-exponent thresholds of Ben Arous, Gheissari & Jagannath (2021) say that for an online single neuron, recovering $\theta^*$ costs $n = \Theta(d^{k-1})$ samples, set by the link's information exponent $k$ alone — and none of these is a network trained end to end on the fixed wide MLP I am handed. So the control is "run plain SGD on the whole net and hope the first layer moves."

The method is the textbook recipe, deliberately containing no single-index structure. The first layer $\text{Linear}(d, W)$ with $d=100$, $W=256$ gets Kaiming-normal init: each weight drawn $\sim N(0, 2/d)$, the factor $2$ compensating for ReLU killing half its pre-activations, biases at zero. The readout $\text{Linear}(W, 1)$ gets a small uniform init in $[-1/\sqrt{W}, 1/\sqrt{W}]$, the usual $1/\sqrt{\text{fan\_in}}$ scale that keeps the output at $O(1)$ variance. Nothing here knows about $\theta^*$: the $W$ rows of $\text{fc1}$ point in independent random directions, each correlated with $\theta^*$ by only $\sim 1/\sqrt{d} \approx 0.1$ — the concentration fact that a random unit vector is nearly orthogonal to any fixed one. The entire burden of finding the direction therefore falls on the optimiser, which is plain SGD with momentum $0.9$, learning rate $10^{-2}$, no weight decay. The per-step update is the canonical supervised loop — zero grads, forward the mini-batch, mean-squared error against the noisy targets, backward, step — over batches of $256$ for $8000$ steps with a fresh i.i.d. index each step. The `finalize` callback does nothing; vanilla SGD has no closed-form refit.

What decides the outcome is the structure of the first-layer gradient. For one row $w_j$ under squared loss the gradient is schematically $\nabla_{w_j} L = \mathbb{E}[\,2(f(x) - y)\,a_j\,x\,\mathbf{1}\{\langle w_j, x\rangle + b_j \ge 0\}\,]$, and the piece carrying directional information is the correlation $\mathbb{E}[\,g(\langle\theta^*,x\rangle)\,x\,\mathbf{1}\{\cdots\}\,]$. Expanded against the Gaussian in Hermite polynomials, the leading term pointing along $\theta^*$ is governed by the first non-vanishing Hermite coefficient of $g$ — the information exponent $k$. For $k=1$ (ReLU, sign) the first Hermite moment $\mathbb{E}[y\,x] = \mu_1\,\theta^*$ is already nonzero, so every gradient carries an $O(1)$-detectable component along $\theta^*$ from the very first step. For $k=3$ (the $\mathrm{He}_3$ link) the first two moments vanish; the direction does not appear until the third-order term, which after contracting against a row that overlaps $\theta^*$ by only $\sim 1/\sqrt{d}$ has size $\sim d^{-(k-1)/2} = d^{-1} \approx 0.01$ — buried under the per-mini-batch sampling noise $\sim \sqrt{d/B} = \sqrt{100/256} \approx 0.6$. The signal-to-noise ratio per step on the hard link is thus about $0.01/0.6 \approx 0.02$: the direction signal is essentially invisible inside the mini-batch gradient, and momentum cannot manufacture a signal that is not there — it only averages noise.

This predicts a clean $k$-split before I even run it. On `relu-d100` ($k=1$) the first moment is large, the signal is present every step, and SGD should drive `direction_recovery` near $1$ — the regime where the textbook recipe genuinely works and later methods can at best tie. On `sign-d100` ($k=1$ but discontinuous) the first moment is nonzero but the readout must approximate a step from ReLU features over a rougher landscape, so I expect partial, high-variance recovery. On `hermite-d100` ($k=3$) I expect the worst: the direction signal sits below the mini-batch noise, the rows barely rotate toward $\theta^*$, and recovery lands well short of $1$ — the $n = \Theta(d^{k-1})$ wall seen from the per-step side. There is a second, structural reason the hard links suffer even when SGD finds some direction: every row is free and trained jointly with the head, so the high-dimensional direction search and the one-dimensional link fit happen at once and interfere — the head chases whatever the rows currently encode while the rows are still near the equator. Nothing decouples those two jobs and nothing aggregates the weak per-step signal across the dataset; each step sees only its $256$-sample slice.

It is worth being explicit that this is the control's honest ceiling, not an under-tuned strawman. Cranking `base_lr` amplifies noise as much as the absent signal, so the rows diffuse faster around their random init rather than converging; running more steps does not accumulate a sub-threshold signal either, because each fresh i.i.d. batch is an independent draw of essentially the same noise, making the long run a random walk on the sphere rather than a descent toward a pole. The only ways to surface the signal are to change the *estimator* — aggregate over the whole dataset — and the *landscape* — decouple direction from link. Those are precisely the structural moves the later rungs make: the first will freeze part of the net to collapse the landscape, and the next will replace the noisy mini-batch crawl with one full-batch step that sums the signal over all $n_{\text{train}} = 32768$ samples. At this rung my edit is the trivial one — leave the scaffold at its default — and the harness's own estimator $\hat\theta = \mathrm{normalize}(\sum_j |a_j|\,w_j)$ reads off whatever alignment those rows happen to acquire.

```python
# EDITABLE region of custom_strategy.py (lines 176-239) — step 1: vanilla SGD MLP
class Strategy:
    """Vanilla SGD on a two-layer ReLU MLP (reference baseline)."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Kaiming-normal first layer; small uniform second layer.
        nn.init.kaiming_normal_(net.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(net.fc1.bias)
        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            net.parameters(),
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
