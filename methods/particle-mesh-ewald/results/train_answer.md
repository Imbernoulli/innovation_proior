I have a solvated protein in a periodic box — twenty or thirty thousand atoms — and at every one of millions of time steps I need the electrostatic energy and the force on every atom. The electrostatics is the wall. Simply truncating the Coulomb interaction at a cutoff looks fine for a while and then quietly goes wrong: water orders artificially at the cutoff shell, the protein drifts, the ions sit in the wrong places. The $1/r$ tail genuinely matters, and for a periodic system the honest object is the Ewald lattice sum, which converges absolutely and respects the boundary conditions exactly. The trouble is purely cost. Writing the pair potential as $\psi(r;\beta) = \Phi_{\rm dir}(r;\beta) + \Phi_{\rm rec}(r;\beta)$, the total Ewald energy of a neutral cell of $N$ charges $q_i$ at $r_i$ with volume $V$ is

$$E = \sum_{i<j} q_i q_j \frac{\operatorname{erfc}(\beta r_{ij})}{r_{ij}} \;-\; \frac{\beta}{\sqrt{\pi}}\sum_i q_i^2 \;+\; \tfrac12 \sum_{i,j} q_i q_j\,\Phi_{\rm rec}(r_j-r_i;\beta) \;+\; J(D),$$

with the reciprocal pair potential, over reciprocal vectors $m = m_1 a_1^* + m_2 a_2^* + m_3 a_3^*$,

$$\Phi_{\rm rec}(r;\beta) = \frac{1}{\pi V}\sum_{m\neq 0} \frac{\exp(-\pi^2 m^2/\beta^2)}{m^2}\,\exp(2\pi i\, m\cdot r).$$

The four terms are the direct sum of *screened* charges, the self-energy that removes each charge's interaction with its own screening Gaussian, the reciprocal sum, and the de Leeuw–Perram–Smith surface/dipole term $J(D)$. The key observation is that $\psi$ depends on $\beta$ only through an additive constant, so for a neutral cell the total $E$ is independent of $\beta$; the parameter merely moves work between the direct and reciprocal sums. Crank $\beta$ up and $\operatorname{erfc}(\beta r)$ collapses, so the direct sum is just the minimum-image pairs inside a $9$ Å cutoff — $O(N)$ with the Verlet list I already have. Self and surface are $O(N)$ too. Everything is cheap except the reciprocal sum $E_{\rm rec} = \tfrac12 \sum_{i,j} q_i q_j \Phi_{\rm rec}(r_j-r_i;\beta)$, a sum over all $N^2$ pairs at every step. That is the bottleneck, and pushing $\beta$ up to cheapen the direct sum only shovels more weight into it; tuning $\beta$ to balance the two gets to $O(N^{3/2})$ at best, which for thirty thousand atoms is still hopeless.

Rewriting $E_{\rm rec}$ with the structure factor $S(m) = \sum_j q_j \exp(2\pi i\, m\cdot r_j)$ gives $E_{\rm rec} = (1/2\pi V)\sum_{m\neq 0} [\exp(-\pi^2 m^2/\beta^2)/m^2]\,S(m)S(-m)$. The framing improves — the Gaussian kernel decays fast, so only a modest shell of $m$ matters — but it does not escape the scaling: each $S(m)$ is itself a sum over all $N$ charges, needed for $O(N)$ vectors, so we are back at $O(N^2)$. The sum $\sum_j q_j \exp(2\pi i\, m\cdot r_j)$ has the exact shape of a Fourier transform of a charge density; if only the $r_j$ sat on a regular grid, all the $S(m)$ would be one FFT at $O(M\log M)$. But the atoms are at arbitrary positions. This is the gap the plasma and cosmology people answered with particle–mesh: assign off-grid charges to a regular mesh with a smooth window, solve the long-range Poisson problem on the grid by FFT, interpolate back — all $O(N\log N)$. Their assignment windows, though, are low order, built for roughly uniform particles where moderate accuracy suffices; for the lumpy charge distribution of a protein high accuracy is hard to get, and the low-order windows give forces of limited smoothness, which is bad for integrating the equations of motion. I want a knob that dials accuracy arbitrarily high and forces that are continuous in position. So I keep the particle–mesh skeleton — charges to a grid, long-range part by FFT — and redesign the assignment.

I propose Particle Mesh Ewald. The pivot is that $\Phi_{\rm rec}$ depends only on the *difference* of fractional coordinates $f_k = a_k^*\cdot r$ and is the same fixed function at every step and for every pair. In fractional coordinates,

$$\Phi_{\rm rec}(f_1,f_2,f_3;\beta) = \frac{1}{\pi V}\sum_{m\neq 0} \frac{\exp(-\pi^2 m^2/\beta^2)}{m^2}\,\exp\!\big[2\pi i (m_1 f_1 + m_2 f_2 + m_3 f_3)\big],$$

a product of three one-dimensional complex exponentials. So I do not recompute it. Once at the start I evaluate $\Phi_{\rm rec}$ (and its gradient) on the regular grid of fractional coordinates $(l_1/K_1, l_2/K_2, l_3/K_3)$ for chosen grid sizes $K_1,K_2,K_3$ and store the arrays. The only remaining job is to get from an atom's actual fractional position to a value of $\Phi_{\rm rec}$, and the bridge is interpolation. Since I rejected low-order windows, I use high-order Lagrange interpolation of the smooth exponentials and let the *order* be the accuracy knob — the error of order-$q$ interpolation of a smooth function on a grid of spacing $h$ shrinks like $h^q$, geometrically in the order. The naive trap is that high-order *global* polynomial interpolation is ill-conditioned (Runge's phenomenon, oscillations near the endpoints). The escape is to always keep the evaluation point in the *centre* of the stencil: for a point $x$ take the $(2p{-}1)$th-order Lagrange interpolation on the local window of $2p$ grid points straddling $x$, with $x$ in the central interval — centred, local, fixed-width, no endpoint pathology.

Concretely, write the weights in barycentric form. For real $x$ with $[x]=\operatorname{floor}(x)$, define the integer that locates the left edge of the centred stencil, $k_{p,K}(x) = [Kx] - p + 1$. On the canonical window $0\le k\le 2p{-}1$,

$$\varphi_{p,K}(x,k) = \frac{(-1)^k \binom{2p-1}{k}\,/\,(x - k/K)}{\sum_{l=0}^{2p-1} (-1)^l \binom{2p-1}{l}\,/\,(x - l/K)},$$

zero otherwise, the barycentric weight of $(2p{-}1)$th-order Lagrange interpolation through the points $k/K$; then shift the window to centre on $x$ via $\theta_{p,K}(x,k) = \varphi_{p,K}[\,x - k_{p,K}(x),\; k - k_{p,K}(x)\,]$. Three properties carry the method. The weights are nonzero for only $2p$ values of $k$, so the stencil is local and cheap. They satisfy $\sum_k \theta_{p,K}(x,k) = 1$, a partition of unity, which reproduces constants and conserves charge. And $\theta_{p,K}(k/K,k)=1$ with $\theta_{p,K}(l/K,k)=0$ for $l\neq k$, so $\theta$ is continuous in $x$ — and because $x$ is pinned to the central interval there is no high-order ill-conditioning. That continuity is exactly what makes the interpolated energy and forces continuous functions of the atomic positions, the property the cell-multipole methods cannot offer because an atom crossing a hierarchy boundary makes them jump.

Now interpolate $\Phi_{\rm rec}$ at a pair. Because the exponential factorizes over the three axes the interpolation factorizes too, and using $\Phi_{\rm rec}(-r)=\Phi_{\rm rec}(r)$ the interpolated value couples atom $i$'s weights and atom $j$'s weights only through $\Phi_{\rm rec}$ evaluated at the *difference* of their grid indices — the signature of a convolution. Make it explicit by spreading every charge onto a single gridded charge array through its weight cloud,

$$Q(l_1/K_1, l_2/K_2, l_3/K_3) = \sum_{j=1}^{N} q_j\, \theta_{p,K_1}(f_{j1},l_1)\,\theta_{p,K_2}(f_{j2},l_2)\,\theta_{p,K_3}(f_{j3},l_3),$$

which costs $O((2p)^2 N)$ — a few hundred grid touches per atom, linear in $N$. The interpolated reciprocal energy then collapses into

$$E_{\rm rec} \approx \tfrac12 \sum_{\text{grid}} Q\cdot(\Phi_{\rm rec}*Q),$$

a discrete convolution of the precomputed $\Phi_{\rm rec}$ grid with $Q$: spread the charges to make $Q$, convolve with the stored $\Phi_{\rm rec}$ to get a potential on the grid, then gather that potential back to each atom through the *same* weights, weighted by $q_i$. A convolution on a periodic grid is a pointwise product in Fourier space, so the FFT does it for free at $O(M\log M)$ with $M=K_1K_2K_3$. There is a clean simplification by Parseval: $\tfrac12\sum_{\text{grid}} Q\cdot(\Phi_{\rm rec}*Q)$ equals $(1/2\pi V)\sum_{m\neq 0}[\exp(-\pi^2 m^2/\beta^2)/m^2]\,|\mathcal{F}(Q)(m)|^2$, where $\mathcal{F}(Q)(m)$ is precisely the *approximate* structure factor $S(m)$. So I need not even FFT a precomputed $\Phi_{\rm rec}$ grid; I apply the reciprocal kernel directly to $\mathcal{F}(Q)$. The energy is one FFT of $Q$, a multiply by the kernel, and a sum. For the forces I want the gradient field on the grid, so I form $\mathcal{F}(Q)$ times the kernel, multiply by $i\,2\pi\, m$ to differentiate, inverse-FFT to get the three Cartesian gradient components, and gather them through the same weights, with the force as $-\partial E_{\rm rec}/\partial r_i$. Because the only position dependence is through the continuous $\theta$ weights, the forces are continuous; and interpolating energy and forces symmetrically makes the reciprocal forces sum to zero to machine precision, so Newton's third law holds exactly even though the scheme is an approximation.

The accuracy and the scaling come out of one error bound. The object interpolated per axis is the complex exponential $\exp(2\pi i\, m x)$, and the standard remainder of $(2p{-}1)$th-order Lagrange interpolation on the grid $\{k/K\}$ gives, for all $x$,

$$\Big|\exp(2\pi i\, m x) - \sum_{k\in\mathbb{Z}} \exp(2\pi i\, m k/K)\,\theta_{p,K}(x,k)\Big| < 2\,\binom{2p}{p}\,\Big(\frac{m}{4K}\Big)^{2p}, \qquad \binom{2p}{p}=\frac{(2p)!}{(p!)^2}.$$

Propagating this through $\Phi_{\rm rec}$ and bounding the reciprocal sum by an integral — changing variables to $x_k = a^*_{1k}m_1 + a^*_{2k}m_2 + a^*_{3k}m_3$, going to spherical coordinates, applying Cauchy–Schwarz on the components $m_k$, and using the Gaussian moments — yields

$$\big|\hat\Phi_{\rm rec,p} - \Phi_{\rm rec,p}\big| \le 8\,\binom{2p}{p}\frac{(2p)!}{p!}\,\frac{\beta}{\sqrt{\pi}}\Big(\frac{\beta}{8\pi}\Big)^{2p}\Big[\Big(\frac{a_1}{K_1}\Big)^{2p} + \Big(\frac{a_2}{K_2}\Big)^{2p} + \Big(\frac{a_3}{K_3}\Big)^{2p}\Big].$$

The error is governed by the *grid spacing* $a_k/K_k$ raised to twice the order. Fix $a_k/K_k$ below about one Ångström and choose $p$: the error goes to zero geometrically for any tolerance I name. Holding the spacing fixed means $K_1K_2K_3$ grows in proportion to the cell volume, which grows in proportion to $N$, so $M$ is $O(N)$; the convolution is $O(N\log N)$ and the spread/gather is $O((2p)^2 N) = O(N)$. For any fixed accuracy the whole reciprocal evaluation is $O(N\log N)$, and this is a proof, not a hope — accuracy is set by $p$, cost by $N$, and the two are decoupled. A few choices pin the method down: $\beta\approx 0.386$ Å$^{-1}$ makes $\operatorname{erfc}(\beta r)/r$ negligible beyond the $9$ Å cutoff (and the total energy does not care about its exact value, being $\beta$-invariant); a grid spacing $a_k/K_k$ of about $0.5$–$1$ Å keeps the $(a_k/K_k)^{2p}$ factor small; and the order $p$ trades accuracy against a spread/gather cost that grows as $p^3$, with a modest order on a sub-Ångström grid enough for MD-grade $\sim 10^{-4}$ force errors and a couple of extra orders available to push toward near-exact. The scheme is fully general for non-orthogonal cells and slots into a Verlet-list MD code; it costs more memory than a plain neighbor-list method, but memory is cheap. The reciprocal sum has gone from $O(N^2)$ to $O(N\log N)$ at any accuracy, with smooth forces. That is the method.

Here is the reciprocal-space evaluator — the spread → FFT → influence → inverse-FFT → gather pipeline; the direct, self, and surface terms are the standard $O(N)$ Ewald bookkeeping.

```python
import numpy as np
from math import comb

def lagrange_weights(u, p):
    """Centred (2p-1)th-order Lagrange weights theta_{p,K} at scaled coord u=K*f,
    on the 2p-point window. Returns (left_index, weights[2p])."""
    floor_u = int(np.floor(u))
    left = floor_u - p + 1                       # k_{p,K}(x) = [Ku] - p + 1
    nodes = np.arange(2 * p)
    x = u - left
    bary = np.array([(-1) ** k * comb(2 * p - 1, k) for k in nodes], float)
    diff = x - nodes
    on_node = np.isclose(diff, 0.0)
    if on_node.any():
        w = np.where(on_node, 1.0, 0.0)
    else:
        terms = bary / diff
        w = terms / terms.sum()                  # partition of unity
    return left, w

class PME:
    def __init__(self, cell, beta, grid_dims, order):
        self.cell = np.asarray(cell, float)      # 3x3, lattice vectors as columns
        self.V = abs(np.linalg.det(self.cell))
        self.a_star = np.linalg.inv(self.cell).T # rows a_k*: f = a_star @ r
        self.beta = beta
        self.K = tuple(grid_dims)
        self.p = order
        self.M = self.K[0] * self.K[1] * self.K[2]
        self._setup()

    def _setup(self):
        K1, K2, K3 = self.K
        m1 = np.fft.fftfreq(K1, d=1.0 / K1)
        m2 = np.fft.fftfreq(K2, d=1.0 / K2)
        m3 = np.fft.fftfreq(K3, d=1.0 / K3)
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing="ij")
        A = self.a_star
        mx = M1 * A[0, 0] + M2 * A[1, 0] + M3 * A[2, 0]
        my = M1 * A[0, 1] + M2 * A[1, 1] + M3 * A[2, 1]
        mz = M1 * A[0, 2] + M2 * A[1, 2] + M3 * A[2, 2]
        m2sq = mx ** 2 + my ** 2 + mz ** 2
        kern = np.zeros_like(m2sq)
        nz = m2sq > 0
        kern[nz] = np.exp(-(np.pi ** 2) * m2sq[nz] / self.beta ** 2) \
                   / (np.pi * self.V * m2sq[nz])
        self.kern = kern                         # reciprocal Ewald kernel in k-space
        self.m_vec = (mx, my, mz)

    def _spread(self, positions, charges):
        K1, K2, K3 = self.K
        Q = np.zeros((K1, K2, K3), dtype=complex)
        info = []
        for r, q in zip(positions, charges):
            f = self.a_star @ r
            f = f - np.floor(f)
            l1, w1 = lagrange_weights(f[0] * K1, self.p)
            l2, w2 = lagrange_weights(f[1] * K2, self.p)
            l3, w3 = lagrange_weights(f[2] * K3, self.p)
            i1 = (l1 + np.arange(2 * self.p)) % K1
            i2 = (l2 + np.arange(2 * self.p)) % K2
            i3 = (l3 + np.arange(2 * self.p)) % K3
            Q[np.ix_(i1, i2, i3)] += q * w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            info.append((i1, i2, i3, w1, w2, w3, q))
        return Q, info

    def energy_and_forces(self, positions, charges):
        Q, info = self._spread(positions, charges)
        Qhat = np.fft.fftn(Q)                     # F(Q)(m) = approximate structure factor S(m)
        # E_rec = 1/2 sum Q.(Phi_rec*Q) = 1/(2 pi V) sum kernel |S(m)|^2  (Parseval)
        E_rec = 0.5 * np.sum(self.kern * np.abs(Qhat) ** 2)
        pot_hat = self.kern * Qhat               # F(Phi_rec * Q)
        mx, my, mz = self.m_vec                   # *M restores numpy ifft normalization
        gx = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mx)).real * self.M
        gy = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * my)).real * self.M
        gz = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mz)).real * self.M
        F = np.zeros_like(positions)
        for k, (i1, i2, i3, w1, w2, w3, q) in enumerate(info):
            wc = w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            b = np.ix_(i1, i2, i3)
            F[k] = -q * np.array([np.sum(wc * gx[b]),
                                  np.sum(wc * gy[b]),
                                  np.sum(wc * gz[b])])
        return E_rec, F
```
