The lazy baseline told me exactly what is missing, and it told me in the Fourier-recovery column. NTK landed at $\text{test\_mse}_{h_1} = 2.78$, $h_2 = 2.99$, $h_3 = 1.00$ — all three essentially at their trivial-predictor variances $(3,3,1)$, with recovery $0.96, 1.00, 1.00$, i.e. recovering almost nothing. Even on $h_1$, where the degree-1 piece should have been reachable, $\text{test\_mse}$ only fell from $3$ to $2.78$: a frozen $M=100$ random-feature basis barely resolved the lone $z_1$ and got none of $z_1z_2$ or $z_1z_2z_3$. The failure is sharp and structural — the features were frozen *before* the network knew which coordinates were latent, and a non-adaptive kernel machine cannot put its limited capacity where the signal is. The fix is not a better optimizer on those features; it is to let the features *move*, and the parametrization that lets them move is the whole game now.

I propose mean-field two-layer SGD. The single decisive change from NTK is the output normalization: I take the mean-field scaling
$$\hat f(x) = \frac{1}{M}\sum_{j=1}^{M} a_j\,\sigma(\langle w_j, x\rangle),$$
unfreeze the first layer, and train by square-loss one-pass batch-SGD with a large plain step. With the $1/M$ out front (rather than the lazy $1/\sqrt M$), each neuron contributes $O(1/M)$ to the output but the weights are free to travel an $O(1)$ distance, so the dynamics stays genuinely nonlinear and the features can rotate toward the latent subset. The $M$ neurons are exchangeable — the output only sees them through their empirical measure $\rho = \frac1M\sum_j \delta_{\theta_j}$, $\theta_j = (a_j, w_j)$ — and the population risk $R(\rho) = \mathbb{E}_x[(f^* - \hat f(\cdot;\rho))^2]$ is a functional of $\rho$ alone (a constant, a linear term, and a quadratic neuron-interaction term). In the wide, small-step limit one-pass SGD on this object is a Wasserstein gradient flow $\partial_t\rho_t = \nabla\!\cdot(\rho_t H\,\nabla\psi(\theta;\rho_t))$ — a gas of neuron-particles descending the risk (Mei–Montanari–Nguyen 2018; Chizat–Bach 2018).

What makes this tractable, and what separates the three targets, is the sparsity. Write $x = (z, r)$ for signal and noise coordinates and $w = (u, v)$ correspondingly. At initialization the noise block is sign-symmetric, so $\hat f(x;\rho_0)$ does not depend on $r$; and because the flow couples weights only through $\langle w,x\rangle$, a noise-sign flip pushes $\rho_t$ to a measure solving the same PDE with the same initial condition, so by uniqueness the network stays independent of the irrelevant directions throughout training. The noise then enters only through $\langle v, r\rangle$, approximately Gaussian $\|v\|_2 G$ for large $d$, so the whole $(d-P)$-dimensional noise weight collapses to a single scalar $s = \|v\|_2$ acting as a smoothing width. The flow reduces to a dimension-free gradient flow on $(a, u, s)\in\mathbb{R}^{P+2}$. The crucial fact: as $d\to\infty$ the signal block starts at $u^0 = O(1/\sqrt d)\to 0$, so the dynamics *starts with the signal weights at the origin*, and whether a target is learned is precisely whether the flow can push $u$ off the origin. The first-layer evolution is
$$\frac{d}{dt}u_i = a\,\mathbb{E}_z\big[(h^*(z) - \hat f(z))\,\sigma'(\langle u,z\rangle + sG)\,z_i\big] - \text{reg},$$
and a coordinate $i$ leaves the origin only if something correlates $z_i$ with the residual-weighted gradient there. This is the lens that decides $h_1$ vs $h_2$ vs $h_3$.

For $h_3 = z_1z_2z_3$, symmetry forces $u_1 = u_2 = u_3$ for all time, and the driver $\mathbb{E}_z[z_1z_2z_3\,\sigma'(u(z_1+z_2+z_3))\,z_1]$ is homogeneous in $u$ and vanishes to high order at $u=0$: there is no first-order push, since the only thing correlating with $z_1$ is the full triple, which contributes nothing at the origin. The coordinates never budge and $R \ge \hat h_3(\{1,2,3\})^2 = 1$. For $h_2 = z_1z_2 + z_2z_3 + z_3z_4$, the first support $\{1,2\}$ would have to lift *two* coordinates at once: $u_1$'s push through $z_1z_2$ is gated by $u_2$ and symmetrically $u_2$'s by $u_1$, a homogeneous linear system from the origin. The general statement is that for a non-MSP target there is a leftover index set $\Omega$ whose new coordinates never appear "alone"; bounding the relevant correlations gives $|\frac{d}{dt}u_i| \le K(1+t)^2 \max_{j\in\Omega}|u_j|$, so by Grönwall every coordinate in $\Omega$ stays at zero. Both $h_2$ and $h_3$ are stuck, at strictly positive risk, *independent of step size or activation*. But $h_1 = z_1 + z_1z_2 + z_1z_2z_3$ is the vanilla staircase, supports $\{1\}\to\{1,2\}\to\{1,2,3\}$ each adding one new coordinate (MSP / leap-1), and here the recipe comes alive in a cascade from the origin: $z_1$ drives $u_1$ at first order so $u_1 \sim t$; once $u_1\neq0$ the term $z_1z_2$ drives $u_2 \sim t^2$; then $u_1,u_2\neq0$ light $u_3$ through $z_1z_2z_3$, $u_3 \sim t^4$. The weights light up sequentially, lower degree first, each new stair an entire order slower, $|u_k| = \Theta(t^{2^{k-1}})$ — climbing the staircase that $h_2$ and $h_3$ have no entry point to start (Abbe–Boix-Adsera–Misiakiewicz 2022, 2023).

Two design choices keep that cascade alive, which is why the fill looks as it does. The activation must have $\sigma^{(r)}(0)\neq0$ for $r = 0,\dots,P$, because the cascade $\frac{d}{dt}u_k = a\,\alpha_{1..k}\,m_k\prod_{j<k}u_j$ (with $m_r = \sigma^{(r)}(0)$) dies the instant any $m_k = 0$ — and a symmetric activation (odd $\tanh$ zeroes the even derivatives, even zeroes the odd) breaks half the chain. The shifted sigmoid $\sigma(x) = (1 + e^{-x+0.5})^{-1} = \mathrm{sigmoid}(x - 0.5)$ moves evaluation off the logistic's symmetric center so every low-order derivative at the origin is nonzero. The readout signs must be diverse: the second, linear phase is kernel regression with $K(z,z') = \mathbb{E}_a[\sigma(\langle u(a),z\rangle)\sigma(\langle u(a),z'\rangle)]$, whose Fourier-basis Gram $M_a = (\mathbb{E}_a[a^{\beta(S)+\beta(S')}])$, $\beta(S) = \sum_{k\in S}2^{k-1}$, is full rank precisely because $a$ is random — random $\pm1$ signs give the neuron diversity that makes the kernel positive definite, while fixing all $a$ equal would collapse it. So: $w \sim N(0, I_d)$ (signal weights near the origin), $a \sim \mathrm{Unif}\{+1,-1\}$ (diversity), shifted sigmoid (live cascade), $1/M$ normalization (feature-learning regime), plain SGD at $\mathrm{lr} = 0.5$ (the bare gradient flow, nothing adaptive). The falsifiable prediction splits by leap: $h_1$ should beat NTK's $2.78$ as it climbs the degree-1 and degree-2 stairs, while $h_2$ and $h_3$ should sit at the same trivial $\text{test\_mse}\approx 3$ and $\approx 1$ — feature learning helps, but plain SGD climbs only one new coordinate at a time, so any leap greater than one stalls it.

```python
def build_model(config: TaskConfig) -> nn.Module:
    """Two-layer mean-field network (Abbe et al. Fig. 1 default)."""

    class TwoLayerMeanField(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=False)
            self.fc2 = nn.Linear(M, 1, bias=False)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
            # No trainable bias in the Abbe et al. mean-field setup.
            with torch.no_grad():
                signs = torch.randint(0, 2, (M, 1), dtype=torch.float32) * 2.0 - 1.0
                self.fc2.weight.copy_(signs.t())
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u - 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            return self.fc2(h).view(-1) / self.M

    return TwoLayerMeanField(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """Plain SGD with eta_k = 1/2."""
    return torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)


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
