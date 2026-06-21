The Johnson-Lindenstrauss lemma promises that a finite set of high-dimensional points can be mapped into $k = \Theta(\epsilon^{-2}\log(1/\delta))$ coordinates while every pairwise distance is preserved up to a $(1\pm\epsilon)$ factor. A dense Gaussian or Rademacher matrix delivers this routinely: for any fixed vector the projected squared norm concentrates with failure probability $\delta$, and a union bound over the differences of a point set finishes the argument. The target dimension is therefore already correct and is not where the difficulty lives. The difficulty is the cost of applying the map. A dense $k \times d$ matrix writes into essentially all $k$ output coordinates for every nonzero input coordinate, so even on a sparse input the multiplication costs $O(k\,\|x\|_0)$ and the output is immediately dense. The discrete-sign and database-friendly variants reduce constants and random bits but leave a constant fraction of nonzero entries per column, which is still dense at the level of sparse-vector multiplication. Fast-transform constructions precondition by spreading mass with a random sign flip and a Hadamard or Fourier matrix before sampling, but that preconditioning is a global $\Theta(d\log d)$ pass over the ambient dimension, which defeats the purpose in streaming or sparse-update settings. Signed hashing updates only a few counters per nonzero and is genuinely fast, but the strong estimators built on it — CountSketch with several repetitions and a median — are nonlinear, while the geometric goal demands one linear map followed by an ordinary Euclidean norm. What we need is a single linear map that keeps the dense target dimension, applies to a vector in time proportional to its input sparsity, uses manageable randomness, and preserves the norm of an arbitrary fixed vector rather than only a well-spread one.

I propose a fixed-column-sparsity Johnson-Lindenstrauss distribution: a sparse sign matrix in which every column carries exactly $s$ nonzero entries, each of magnitude $1/\sqrt{s}$ with an independent random sign. The first design decision is the load-bearing one. The naive idea of zeroing matrix entries independently fails on the simplest possible input. A basis vector then lands in a random number of rows, so its projected norm fluctuates by itself, before it ever interacts with another coordinate — diagonal noise that no sign cancellation can repair, because there is no second coordinate to cancel against. Fixing the number of active entries per column to exactly $s$, each weighted by $1/\sqrt{s}$, makes a basis vector preserved *exactly*: the diagonal part of $\|Sx\|_2^2$ is deterministically equal to the squared norm and contributes no error at all. The support of a column may be chosen either as $s$ distinct rows or as one row inside each of $s$ equal blocks; the block version is what the reference implementation uses because it makes the per-block sampling trivial and keeps the rows partitioned. Spending the sparsity budget this way — removing self-error first — is what lets a concentration proof handle the only error that remains.

What remains is entirely cross-collision error between two distinct input coordinates. Writing $\eta_{r,i}$ for the indicator that column $i$ activates row $r$ and $\sigma_{r,i}$ for its sign, the error on a unit vector $x$ is

$$Z = \|Sx\|_2^2 - 1 = \frac{1}{s}\sum_r \sum_{i \neq j} \eta_{r,i}\,\eta_{r,j}\,\sigma_{r,i}\,\sigma_{r,j}\,x_i\,x_j .$$

Every surviving term is a signed collision: two different coordinates $i$ and $j$ both chose the same row $r$. The whole problem reduces to showing this signed collision polynomial is small. There is a clean but ultimately too-expensive route: if the column supports were arranged so that every pair of columns collided in only about $s^2/k$ rows, then with locations fixed, $Z$ would be a quadratic form in the signs and Hanson-Wright would control it through its Frobenius norm (giving $k$ on the dense JL scale) and its spectral norm (giving $s$ on the $\epsilon^{-1}\log(1/\delta)$ scale). That route names the right answer but demands a worst-case collision guarantee over every pair of columns in a large ambient dimension, which costs extra logarithms or dimension dependence — strictly more than a single fixed vector actually needs.

So I prove it directly with a high moment rather than conditioning on a globally perfect collision code. Choose an even moment $\ell = \Theta(\log(1/\delta))$, so that Markov's inequality converts an $\ell$-th moment bound into the JL tail. Expanding $\mathbb{E}[Z^\ell]$, each monomial is a product of collision factors that can be read as a directed multigraph whose vertices are input coordinates and whose edges are collision events. The signs do the first filtering: each edge carries two signs, so the expectation over signs vanishes unless every vertex has even degree. This is exactly why full independence is unnecessary — the expansion only reaches degree $\ell$, so the signs need only be independent on the coordinates that can appear inside that expansion; bounded independence of order matched to $\ell$ suffices, and full independence is a convenience rather than the mechanism. The support variables do the second filtering: a column choosing $s$ distinct rows (or one row per block) gives the collision indicators the right upper bounds, and sampling without replacement contributes helpful negative dependence. Each surviving even-degree graph pays collision probabilities tied to powers of $s/k$, and the even-degree constraint limits how many graphs must be counted. Bounding the contribution of a single row or block and then assembling the full $\ell$-th moment, the surviving terms are controlled well enough that the choice

$$\ell = \Theta(\log(1/\delta)), \qquad s = \Theta(\epsilon^{-1}\ell) = \Theta(\epsilon^{-1}\log(1/\delta)), \qquad k = \Theta\!\left(\frac{s^2}{\ell}\right) = \Theta(\epsilon^{-2}\ell) = \Theta(\epsilon^{-2}\log(1/\delta))$$

makes $\mathbb{E}[|Z|^\ell]$ small on the scale needed for failure probability $\delta$. The target dimension $k$ is the dense JL order, and the per-column sparsity is only $\Theta(\epsilon^{-1}\log(1/\delta))$, so a vector is embedded in

$$O(s\,\|x\|_0) = O(\epsilon^{-1}\log(1/\delta)\,\|x\|_0)$$

time. This is decisively more than JL stapled to a hash table: hashing alone supplies fast counter updates but its strongest estimators aggregate nonlinearly, whereas here a single linear image has a concentrating Euclidean norm. The parameters are also not proof artifacts — a two-coordinate vector forces $s = \Omega(\epsilon^{-1}\log(1/\delta))$ for these graph and block schemes, and a with-replacement variant would let even a basis vector suffer repeated choices inside one column. The construction works precisely because exact per-column sparsity removes the self-collision weakness and then pays exactly the sparsity needed to control cross-collisions.

```python
"""Reference block sparse JL transform for the reconstruction artifact.

This is an executable sketch of the support pattern, not a replacement for the
paper's bounded-independence construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseJLBlock:
    input_dim: int
    epsilon: float
    delta: float
    seed: int = 0
    row_constant: float = 12.0
    sparsity_constant: float = 12.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")

        ell = max(1, math.ceil(math.log(1.0 / self.delta)))
        s = max(1, math.ceil(self.sparsity_constant * ell / self.epsilon))
        rows = max(s, math.ceil(self.row_constant * ell / (self.epsilon * self.epsilon)))
        block_size = math.ceil(rows / s)
        rows = s * block_size

        object.__setattr__(self, "moment_order", ell)
        object.__setattr__(self, "sparsity", s)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "block_size", block_size)

    def matrix(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        mat = np.zeros((self.rows, self.input_dim), dtype=float)
        scale = 1.0 / math.sqrt(self.sparsity)

        for col in range(self.input_dim):
            for block in range(self.sparsity):
                start = block * self.block_size
                row = start + int(rng.integers(self.block_size))
                sign = 1.0 if rng.integers(2) else -1.0
                mat[row, col] = sign * scale

        return mat

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(f"x must have shape ({self.input_dim},)")
        return self.matrix() @ x
```
