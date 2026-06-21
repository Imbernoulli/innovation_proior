The task is to recover a latent sparse target $f^*(x) = h^*(x_I)$ on $\{+1,-1\}^{100}$ with a two-layer network under one-pass batch-SGD, across three targets that span the hierarchy: the leap-1 staircase $h_1 = z_1 + z_1z_2 + z_1z_2z_3$, the leap-2 non-MSP chain $h_2 = z_1z_2 + z_2z_3 + z_3z_4$, and the leap-3 monomial $h_3 = z_1z_2z_3$. The whole point of the task is *feature learning*, and the prior art tells me sharply which functions a plain SGD-trained net can find and which a fixed feature map cannot. Before I argue for any adaptive method, I want the honest floor: the strategy that learns no features at all. If I can name precisely what a frozen feature map can and cannot do here, I have the yardstick every later rung is measured against.

I propose the lazy / NTK baseline: freeze the first layer at its random initialization and train only the readout. Concretely, the frozen first layer is a bank of $M$ random features $\phi_j(x) = \mathrm{relu}(\langle w_j, x\rangle + c_j)$ with $w_j, c_j$ drawn once and never updated, and the output is the linear readout
$$\hat f(x) = \frac{1}{\sqrt M}\sum_{j=1}^{M} a_j\,\phi_j(x),$$
with only $a$ trainable. Square-loss SGD on $a$ is then exactly convex online least-squares onto a fixed $M$-dimensional feature map — a random-features approximation to kernel ridge regression, with nothing nonlinear left in the optimization. This is the regime a two-layer net *falls into* if you parametrize it the wrong way, so it is worth seeing exactly how badly it does before I argue for the feature-learning parametrization.

The choices inside this frozen fill are not arbitrary even though nothing adapts. The weight variance is $w \sim N(0, 1/d)$: because $x \in \{+1,-1\}^d$, the pre-activation $\langle w, x\rangle$ is a sum of $d$ independent terms, so unit-variance weights would give it standard deviation $\sqrt d \approx 10$ and the ReLU would be essentially linear-or-dead over the data — an ill-conditioned basis. Scaling to $N(0,1/d)$ keeps $\langle w,x\rangle$ at $O(1)$, the standard He/NTK scaling that makes the features an isotropic sketch of the input. The bias $c \sim N(0,1)$ shifts each ReLU kink off the origin so the features tile the input distribution rather than all firing on the same half-space. The output normalization is $1/\sqrt M$ rather than the mean-field $1/M$: this is the load-bearing difference from the next rung. With $1/\sqrt M$ and unit-scale readout, each feature contributes $O(1/\sqrt M)$ and the readout can move an $O(1)$ amount while the first-layer weights, were they trainable, would barely move — that is exactly the linearization-around-init that *defines* the lazy regime, and choosing it is the whole point of the baseline. The readout starts at zero; since the features are fixed and the loss is convex in $a$, the init only affects the transient, and starting at zero means the network outputs $0$ before training and grows toward the least-squares solution. The step size follows the same logic: the readout sees a fixed feature Gram with scale set by the feature variance, so with $1/\sqrt M$ normalization and $M=100$ a learning rate of $0.5$ would make the iterate oscillate or diverge; I shrink it by the feature dimension to $\mathrm{lr} = 0.5/d$, plain SGD with no momentum, a faithful online gradient step on the convex readout problem. The optimizer filters to the `requires_grad` parameters, i.e. just the readout, since the first layer is frozen.

The part that actually matters is reasoning about what this floor can and cannot reach, because that is what the baseline is for. A fixed feature map $\phi$ defines a kernel $K(x,x') = \mathbb{E}_{w,c}[\phi_w(x)\phi_w(x')]$, and online least-squares onto $M$ random features approximates ridgeless regression in its RKHS. The fixed-feature lower bounds (Ghorbani–Mei–Misiakiewicz–Montanari 2021; Hsu–Sanford–Servedio–Vlatakis-Gkaragkounis 2021) say a degree-$k$ sparse parity over an *unknown* subset $I$ needs $\min(n,q) = \Omega(d^k)$ for any feature map of effective dimension $q$. Here $q$ is bounded by both the feature count $M=100$ and the sample budget $n = b\cdot T = 150\cdot 4000 = 6\cdot10^5$, with $d=100$, so the thresholds are $d^1 = 100$, $d^2 = 10^4$, $d^3 = 10^6$. The prediction is concrete and per-degree. The degree-1 component (the lone $z_1$ in $h_1$) is reachable — a linear function of a single coordinate sits in the span of even a modest random basis with $O(d)$ samples. The degree-2 pieces (the $z_1z_2$ of $h_1$, all three terms of $h_2$) clear the $d^2$ sample threshold but cannot be *resolved* with only $M=100$ features, because a generic random-feature map needs $\Omega(d^2)$ features to carry a degree-2 product over an unknown pair in its effective span. The degree-3 pieces (the $z_1z_2z_3$ of $h_1$, all of $h_3$) need $d^3 = 10^6$ — above both my budget and my feature count, flatly unreachable. So the lazy machine grabs at most the degree-1 piece of $h_1$ and otherwise sits at the trivial predictor.

What is the trivial predictor's score? Each target has unit-magnitude monomials and zero mean, so the constant predictor $0$ has $\text{test\_mse} = \mathbb{E}[h^{*2}] = (\text{number of monomials})$: that is $3$ for $h_1$, $3$ for $h_2$, $1$ for $h_3$. With $\text{score} = \exp(-\text{test\_mse})$ I expect $h_1$ near $\exp(-2) \approx 0.13$ if it removes the degree-1 unit, $h_2$ near $\exp(-3) \approx 0.05$, and — the trap the geometric-mean aggregate exists to expose — $h_3$ near $\exp(-1) \approx 0.37$ purely because $h_3$'s target is the *smallest* in magnitude, not because anything was learned. The Fourier-recovery metric is the honest tell: it measures $|\hat S(\text{model}) - \hat S(h^*)|$ directly, so a lazy machine that recovers nothing shows recovery near its worst possible value regardless of how flattering the score looks. The diagnosis this rung delivers points straight at the next: the failure is not a learning-rate or optimizer problem, since the readout regression is convex and converges; it is that the *features cannot move toward the latent subset $I$* because they were frozen before the network knew which coordinates were latent. The fix is to unfreeze the features and leave the lazy regime — which is the next rung.

```python
def build_model(config: TaskConfig) -> nn.Module:
    """Random-feature / NTK-style network: freeze first layer, train head."""

    class RandomFeaturesNTK(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=True)
            self.fc2 = nn.Linear(M, 1, bias=False)
            # He / NTK-style init for ReLU: w ~ N(0, 1/d), b ~ N(0, 1).
            nn.init.normal_(self.fc1.weight, mean=0.0, std=(1.0 / d) ** 0.5)
            nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
            nn.init.zeros_(self.fc2.weight)
            # Freeze first-layer weights (the "lazy" / NTK regime where features
            # are fixed at initialization).
            self.fc1.weight.requires_grad_(False)
            self.fc1.bias.requires_grad_(False)
            self.M = M

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            # 1/sqrt(M) normalization for NTK-style output.
            return self.fc2(h).view(-1) / (self.M ** 0.5)

    return RandomFeaturesNTK(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """SGD on the trainable parameters (just the readout in lazy training)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=0.5 / config.d, momentum=0.0)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```
