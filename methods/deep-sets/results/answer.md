# Deep Sets

## Problem

Many learning tasks take a **set** as input — an unordered collection $X=\{x_1,\dots,x_M\}$ of variable size: a sample set from a distribution (estimate its entropy/mutual information), a point cloud (classify the object), a galaxy cluster (regress red-shift), a query set of words (expand it). A model on sets must respect the symmetry of its input:

- **Permutation invariance** (set → label): $f(\{x_1,\dots,x_M\}) = f(\{x_{\pi(1)},\dots,x_{\pi(M)}\})$ for every permutation $\pi$.
- **Permutation equivariance** (set → per-element output): $\mathbf f(\pi\mathbf x)=\pi\,\mathbf f(\mathbf x)$.

Treating the set as a fixed-order vector or a sequence (RNN) destroys this symmetry and generalizes poorly across set sizes.

## Key idea

There is one canonical functional form for permutation-invariant set functions, and one canonical form for permutation-equivariant linear layers — and both are *forced*, not merely convenient.

### Structure theorem (sum-decomposition / pooling)

A function $f$ on sets drawn from a **countable** universe is permutation-invariant **iff** it decomposes as

$$f(X)=\rho\!\Big(\sum_{x\in X}\phi(x)\Big)$$

for suitable $\phi$ (per-element embedding) and $\rho$ (readout).

- **Sufficiency**: the sum over the set is order-independent, so any $f$ of this form is invariant.
- **Necessity**: countability gives an injection $c:\mathfrak X\to\mathbb N$. Take $\phi(x)=4^{-c(x)}$. Then $\sum_{x\in X}\phi(x)$ is a base-4 number whose nonzero digits mark exactly the elements present, so the encoding $X\mapsto\sum\phi(x)$ is injective on subsets; define $\rho$ as $f$ composed with its inverse.

For an uncountable domain with fixed size $M$ (e.g. $[0,1]^M$), the same form holds with $\phi(x)=[1,x,x^2,\dots,x^M]$: the sum-of-powers map is injective (Newton–Girard ⇒ elementary symmetric polynomials ⇒ same roots) with continuous inverse (roots depend continuously on coefficients), so $E(X)=\sum\phi(x)$ is a homeomorphism onto its image and $\rho=f\circ E^{-1}$ is continuous. By Stone–Weierstrass plus the Fundamental Theorem of Symmetric Functions, this form is also a universal approximator.

### Equivariant linear layer (tied two-parameter form)

A standard layer $\mathbf f_\Theta(\mathbf x)=\sigma(\Theta\mathbf x)$, $\Theta\in\mathbb R^{M\times M}$, is permutation-equivariant **iff** $\Theta$ commutes with every permutation matrix, which holds **iff**

$$\Theta=\lambda\mathbf I+\gamma\,(\mathbf 1\mathbf 1^{\mathsf T}),\qquad \lambda,\gamma\in\mathbb R,$$

i.e. all diagonal entries equal and all off-diagonal entries equal. The layer is then

$$\mathbf f(\mathbf x)=\sigma\!\big(\lambda\,\mathbf x+\gamma\,(\textstyle\sum_m x_m)\,\mathbf 1\big),$$

an element-wise term plus a broadcast pool. Replacing sum-pool by max-pool ($\sigma(\lambda\mathbf x+\gamma\,\mathrm{maxpool}(\mathbf x)\mathbf 1)$) stays equivariant and works better in some applications. For $D\to D'$ channels: $\mathbf f(X)=\sigma(X\Gamma-\mathrm{pool}(X)\,\Lambda)$ (element-wise map $\Gamma$ minus the broadcast pooled map $\Lambda$), or the reduced $\sigma(\beta+(X-\mathbf 1\,\mathrm{maxpool}(X))\Gamma)$. Stacking such layers stays equivariant; a final commutative pool makes the network invariant.

## Algorithm

**Invariant model.** Embed each element with a shared network $\phi$, pool (sum or mean) over the set, read out with $\rho$:
$$\;x_m\xrightarrow{\phi}\phi(x_m)\;\Rightarrow\;s=\sum_m\phi(x_m)\;\Rightarrow\;\rho(s).$$

**Equivariant model.** Stack permutation-equivariant layers $X\mapsto\sigma(\Gamma X-\Lambda\,\mathrm{pool}(X))$, then pool and read out.

## Code

Invariant model (sum-pooling), as used for the digit-sum task:

```python
import torch
import torch.nn as nn

class InvariantDeepSet(nn.Module):
    """f(X) = rho( sum_x phi(x) ) — the sum-decomposition structure theorem."""
    def __init__(self, in_dim, phi_dim=30, hidden=100, out_dim=1):
        super().__init__()
        # phi: shared per-element embedding (applied to every element identically)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, phi_dim),
        )
        # rho: readout on the pooled code
        self.rho = nn.Linear(phi_dim, out_dim)

    def forward(self, x, mask=None):
        # x: (batch, M, in_dim)
        h = self.phi(x)                       # (batch, M, phi_dim) — phi(x_m)
        if mask is not None:                  # mask out padding for variable M
            h = h * mask.unsqueeze(-1)
        s = h.sum(dim=1)                      # sum_x phi(x)  — order-independent pool
        return self.rho(s)                    # rho(.)
```

Permutation-equivariant layer and stacked equivariant model, as used for point-cloud classification:

```python
class PermEquivariant(nn.Module):
    """Theta = lambda*I + gamma*11^T  =>  f(X) = sigma( Gamma X - Lambda pool(X) ).
    Two-parameter tied form is the *only* permutation-equivariant linear layer."""
    def __init__(self, in_dim, out_dim, pool='max'):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)              # element-wise term
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False) # pooled term
        self.pool = pool

    def forward(self, x):
        # x: (batch, M, in_dim)
        if self.pool == 'max':
            xm, _ = x.max(dim=1, keepdim=True)   # commutative pool over the set
        else:
            xm = x.mean(dim=1, keepdim=True)
        return self.Gamma(x) - self.Lambda(xm)   # broadcast: equivariant by construction


class EquivariantDeepSet(nn.Module):
    def __init__(self, in_dim, d=256, num_classes=40, pool='max'):
        super().__init__()
        self.phi = nn.Sequential(
            PermEquivariant(in_dim, d, pool), nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
            PermEquivariant(d, d, pool),       nn.ELU(inplace=True),
        )
        self.rho = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(d, d), nn.ELU(inplace=True),
            nn.Dropout(0.5), nn.Linear(d, num_classes),
        )

    def forward(self, x):
        h = self.phi(x)            # stacked equivariant layers stay equivariant
        s, _ = h.max(dim=1)        # final commutative pool => invariant
        return self.rho(s)
```
