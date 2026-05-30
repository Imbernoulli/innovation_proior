# DeepONet

**Problem.** Learn a nonlinear *operator* G: u ↦ G(u) — a map from an input function to an output function (e.g. the solution map of an ODE/PDE, an integral operator) — from data, accurately and with small generalization error, covering both non-local and differential operators without restricting inputs/outputs to a grid.

**Key idea.** Encode the input function u by its values at m fixed "sensors" [u(x₁), …, u(x_m)], and take the output query location y as a second input; the target is the scalar G(u)(y). The operator universal-approximation theorem (Chen & Chen 1995) gives an explicit factorized form — a sum of p products of (a network of the sensor values) and (a function of y). DeepONet realizes this with two sub-networks: a **branch** net that maps the sensor values to coefficients [b₁, …, b_p], and a **trunk** net that maps y to basis functions [t₁, …, t_p], merged by a dot product:

  G(u)(y) ≈ Σ_{k=1}^{p} b_k(u) · t_k(y) + b₀.

The branch handles *which function*, the trunk handles *where*; this branch/trunk split is the inductive bias that makes the model generalize far better than a fully-connected network on the concatenated input [u(x₁), …, u(x_m), y].

**Universal approximation theorem for operators (the seed).** For σ continuous non-polynomial, V a compact set of input functions, G a nonlinear continuous operator, and any ε > 0, there exist n, p, m and constants such that

|G(u)(y) − Σ_{k=1}^{p} [Σ_{i=1}^{n} c_iᵏ σ(Σ_{j=1}^{m} ξ_{ij}ᵏ u(x_j) + θ_iᵏ)] · [σ(w_k·y + ζ_k)]| < ε

for all u ∈ V and y in the output domain. The first bracket is the branch (a net of the sensor values → scalar), the second is the trunk (a neuron of y).

**Architecture choices.**
- *Two sub-nets, dot-product merge.* Directly the theorem's sum-of-products; the trunk's t_k are a learned basis in y, the branch's b_k are input-dependent coefficients.
- *Deep sub-nets.* The theorem only needs shallow nets; deepening both adds expressivity (lower approximation error).
- *Trunk activates its last layer*, so the t_k are basis-function outputs (matching σ(w_k·y + ζ_k)).
- *Bias b₀ (and branch last-layer bias).* Not required by the theorem; reduces generalization error and training variance.
- *Stacked vs unstacked.* Stacked = the literal theorem: one trunk + p separate branch nets (one scalar b_k each). Unstacked = one trunk + a single branch net emitting all of [b₁, …, b_p]. Since p ≳ 10, the unstacked form has far fewer parameters, trains faster, uses less memory, and — despite a slightly larger training error — has *smaller* generalization/test error (tighter, near-linear train–test correlation). Use unstacked with bias.
- *Sub-net type.* FNN by default (no grid needed); CNN if sensors lie on a grid; attention for general settings.

**Number of sensors.** A compactness argument (V compact ⇒ the reconstructed-function sets and their unions are compact ⇒ G of them is compact) shows a finite sensor count m suffices to reach any accuracy ε; the required m grows with the richness of the input-function space, with error falling fast as sensors are added and then plateauing once u is resolved.

**Code (PyTorch, unstacked with bias).**

```python
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(), last_activation=False):
        super().__init__()
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        )
        self.activation = activation
        self.last_activation = last_activation

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1 or self.last_activation:
                x = self.activation(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, m, p, branch_layers, trunk_layers, activation=nn.Tanh()):
        super().__init__()
        self.branch = FNN([m] + branch_layers + [p], activation)               # [u(x_1..m)] -> [b_1..b_p]
        self.trunk = FNN(trunk_layers + [p], activation, last_activation=True)  # y -> [t_1..t_p]
        self.b0 = nn.Parameter(torch.zeros(1))                                  # overall bias

    def forward(self, u_sensors, y):
        b = self.branch(u_sensors)                 # [batch, p]
        t = self.trunk(y)                          # [batch, p]
        out = torch.einsum("bi,bi->b", b, t)       # sum_k b_k t_k
        return out + self.b0                        # [batch]

    def forward_grid(self, u_sensors, y):
        # one set of input functions evaluated at all query points y
        b = self.branch(u_sensors)                 # [batch, p]
        t = self.trunk(y)                          # [n_y, p]
        return torch.einsum("bi,ni->bn", b, t) + self.b0   # [batch, n_y]
```
