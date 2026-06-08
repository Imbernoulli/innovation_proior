# The Restricted Isometry Property and exact sparse recovery by ℓ₁ minimization

## Problem

Recover an unknown signal $x \in \mathbb{R}^n$ from $m \ll n$ linear measurements
$y = \Phi x$, knowing only that $x$ is **sparse** — at most $s \ll n$ nonzero entries
(possibly after a fixed transform). The system is underdetermined, so sparsity is the side
information that makes recovery well posed. The exact-but-NP-hard formulation is
$\min_{\tilde x}\|\tilde x\|_{\ell_0}$ s.t. $\Phi\tilde x = y$; the tractable surrogate is
**basis pursuit**, $\min_{\tilde x}\|\tilde x\|_{\ell_1}$ s.t. $\Phi\tilde x = y$, a linear
program. The questions: when does $\ell_1$ return the $\ell_0$ answer exactly, and how few
measurements suffice?

## Key idea

Impose a single, deterministic, uniform property on $\Phi$: it must act as a
**near-isometry on sparse vectors**. Then the cross-polytope geometry of the $\ell_1$ ball
makes its minimizer coincide with the true sparse signal, and random matrices satisfy the
property at $m = O(s\log(n/s))$ — far fewer measurements than the dimension.

## The Restricted Isometry Property

For each integer $s$, the **restricted isometry constant** $\delta_s$ of $\Phi$ is the
smallest $\delta\ge 0$ such that
$$ (1-\delta_s)\,\|x\|_{\ell_2}^2 \ \le\ \|\Phi x\|_{\ell_2}^2 \ \le\ (1+\delta_s)\,\|x\|_{\ell_2}^2
\qquad\text{for all $s$-sparse } x. $$
Equivalently, every $m\times s$ submatrix $\Phi_T$ ($|T|\le s$) has all singular values in
$[\sqrt{1-\delta_s},\sqrt{1+\delta_s}]$: every set of $s$ columns is nearly orthonormal.
Orthonormal columns give $\delta_s=0$; $\delta_s$ measures the worst-case departure over all
$s$-subsets.

**Identifiability.** $\delta_{2s}<1$ implies the null space of $\Phi$ contains no nonzero
$2s$-sparse vector, hence every $s$-sparse signal is the unique sparsest solution. Bare
identifiability only needs the lower restricted singular value of every $2s$-column block
to be positive; RIP also controls the upper singular values. If a $2s$-column block is
rank-deficient, it yields $s$-sparse $x\neq x'$ with $\Phi x = \Phi x'$ — indistinguishable
by any method.

## Theorem (exact recovery)

**If $\delta_{2s} < \sqrt2 - 1$, then for every $s$-sparse $x$, the solution $x^\star$ of**
$$ \min_{\tilde x}\ \|\tilde x\|_{\ell_1}\quad\text{s.t.}\quad \Phi\tilde x = y=\Phi x $$
**is exactly $x$.** More generally, for arbitrary $x$ and noisy data $y=\Phi x+z$ with
$\|z\|_2\le\varepsilon$, the constrained program $\min\|\tilde x\|_1$ s.t.
$\|\Phi\tilde x - y\|_2\le\varepsilon$ obeys
$$ \|x^\star - x\|_{\ell_2}\ \le\ C_0\,\frac{\|x - x_s\|_{\ell_1}}{\sqrt s}\ +\ C_1\,\varepsilon, $$
where $x_s$ is the best $s$-term approximation and $C_0,C_1$ are explicit constants (e.g.
$\le 4.2$ and $8.5$ at $\delta_{2s}=0.2$). If $x$ is $s$-sparse and noiseless, both terms
vanish and recovery is exact.

### Proof (cone / null-space argument)

Write $x^\star = x + h$; feasibility forces $h\in\ker\Phi$ (noiseless) and the goal is
$h=0$.

1. **Disjoint-support inner products.** For nonzero $u,u'$ on disjoint supports of sizes
   $\le s,s'$, first normalize $a=u/\|u\|_2$ and $b=u'/\|u'\|_2$. Applying RIP to $a\pm b$
   (which are $(s{+}s')$-sparse with $\|a\pm b\|^2=2$) and the parallelogram identity gives
   $|\langle\Phi a,\Phi b\rangle|\le\delta_{s+s'}$; rescaling gives
   $|\langle\Phi u,\Phi u'\rangle|\le\delta_{s+s'}\|u\|_{\ell_2}\|u'\|_{\ell_2}.$

2. **Shell decomposition.** Let $T_0=\mathrm{supp}(x_s)$ ($|T_0|=s$). Sort $h$ outside $T_0$
   by decreasing magnitude into blocks $T_1,T_2,\dots$ of size $s$. For $j\ge2$, magnitude
   averaging gives
   $\|h_{T_j}\|_{\ell_2}\le s^{-1/2}\|h_{T_{j-1}}\|_{\ell_1}$, hence
   $\sum_{j\ge2}\|h_{T_j}\|_{\ell_2}\le s^{-1/2}\|h_{T_0^c}\|_{\ell_1}.$

3. **Cone constraint from $\ell_1$-minimality.** $\|x\|_1\ge\|x+h\|_1$ and the reverse
   triangle inequality give $\|h_{T_0^c}\|_{\ell_1}\le\|h_{T_0}\|_{\ell_1}+2\|x_{T_0^c}\|_{\ell_1}$
   (the last term is $0$ when $x$ is $s$-sparse).

4. **RIP on the head.** Since $\Phi h_{T_0\cup T_1}=-\sum_{j\ge2}\Phi h_{T_j}$, the lower
   bound $\|\Phi h_{T_0\cup T_1}\|^2\ge(1-\delta_{2s})\|h_{T_0\cup T_1}\|^2$ and step 1 yield
   $$ \|h_{T_0\cup T_1}\|_{\ell_2}\le\frac{\sqrt2\,\delta_{2s}}{1-\delta_{2s}}
      \sum_{j\ge2}\|h_{T_j}\|_{\ell_2}\le\frac{\sqrt2\,\delta_{2s}}{1-\delta_{2s}}\,
      s^{-1/2}\|h_{T_0^c}\|_{\ell_1}. $$

5. **Null-space property.** With $\|h_{T_0}\|_1\le\sqrt s\,\|h_{T_0\cup T_1}\|_2$,
   $$ \|h_{T_0}\|_{\ell_1}\ \le\ \rho\,\|h_{T_0^c}\|_{\ell_1},\qquad
      \rho=\frac{\sqrt2\,\delta_{2s}}{1-\delta_{2s}}. $$
   Combined with step 3, $\rho<1$ forces $h=0$ in the noiseless $s$-sparse case. And
   $$ \rho<1\iff \sqrt2\,\delta_{2s}<1-\delta_{2s}\iff \delta_{2s}<\tfrac{1}{1+\sqrt2}=\sqrt2-1. $$

For noisy/compressible $x$ the same chain with the feasibility bound
$\|\Phi(x^\star-x)\|_2\le2\varepsilon$ produces the stated $C_0,C_1$ error bound. $\square$

(An alternative, slightly weaker route via a **dual certificate** $w$ with
$\langle\varphi_j,w\rangle=\mathrm{sgn}(x_j)$ on $T$ and $|\langle\varphi_j,w\rangle|<1$ off
$T$ — built as $w=\Phi_T(\Phi_T^\top\Phi_T)^{-1}\mathrm{sgn}(x)$ and refined by iterating
away an exceptional set — gives exact recovery under
$\delta_s+\theta_{s,s}+\theta_{s,2s}<1$, where $\theta_{s,s'}$ is the restricted-orthogonality
constant; this is implied by $\delta_s+\delta_{2s}+\delta_{3s}<1$ via
$\theta_{s,s'}\le\delta_{s+s'}\le\theta_{s,s'}+\max(\delta_s,\delta_{s'})$.)

## Random matrices: $m = O(s\log(n/s))$

Let $\Phi$ have i.i.d. Gaussian (or subgaussian) entries with variance $1/m$.

- **Per support.** For fixed $|T|=k$, $\Phi_T$ is $m\times k$ Gaussian; its extreme singular
  values concentrate at $1\pm\sqrt{k/m}$ (Marchenko–Pastur), and by Gaussian
  concentration-of-measure (singular values are $1$-Lipschitz),
  $\mathbb{P}(\sigma_{\max}>1+\sqrt{k/m}+t)\le e^{-mt^2/2}$ and symmetrically for
  $\sigma_{\min}$ — so $\delta(\Phi_T)\approx 2\sqrt{k/m}$ except with exponentially small
  probability.
- **Uniform over supports.** Union-bounding over all $\binom{n}{k}\le(en/k)^k$ supports,
  $$ \mathbb{P}(\exists\,\text{bad }T)\le\exp\!\Big(k\log\tfrac{en}{k}-\tfrac{mt^2}{2}\Big)\to0
     \quad\text{when}\quad m\gtrsim \tfrac1{t^2}\,k\log\tfrac{n}{k}. $$
  Setting $k=2s$ and choosing $t$ to enforce $\delta_{2s}<\sqrt2-1$ gives RIP, hence exact recovery, once
  $$ \boxed{\,m = O\!\big(s\log(n/s)\big)\,}. $$
- **Why this count.** RIP is a Johnson–Lindenstrauss bound made *uniform over sparse
  supports*: JL preserves norms of finitely many fixed points; here the target is all
  $2s$-sparse vectors $=$ a union of $\binom{n}{2s}$ subspaces of dimension $2s$, each handled
  by a covering net ($e^{O(s)}$ points) plus continuity, then union-bounded. The two logs add
  to $s\log(n/s)$. This matches the Gelfand-width / Kashin–Garnaev–Gluskin lower bound, so no
  method recovers from fundamentally fewer measurements — the bound is optimal up to constants.

## Final artifact (worked example)

Basis-pursuit decoder and the random-matrix construction for the
$m\sim s\log(n/s) \ll n$ regime.

```python
import numpy as np
from scipy.optimize import linprog

def sample_gaussian(m, n, rng):
    # variance 1/m so columns ~unit-norm; RIP at level ~2s once m = O(s log(n/s))
    return rng.standard_normal((m, n)) / np.sqrt(m)

def make_sparse(n, s, rng):
    x = np.zeros(n); T = rng.choice(n, size=s, replace=False)
    x[T] = rng.standard_normal(s); return x

def recover_l1(Phi, y):
    # min ||x||_1 s.t. Phi x = y, as LP via x = u - v, u,v >= 0
    m, n = Phi.shape
    res = linprog(np.ones(2 * n), A_eq=np.hstack([Phi, -Phi]), b_eq=y,
                  bounds=[(0, None)] * (2 * n), method="highs")
    if not res.success:
        raise RuntimeError(f"basis pursuit LP failed: {res.message}")
    return res.x[:n] - res.x[n:]

def measurements_needed(n, s, C=4.0):
    return int(np.ceil(C * s * np.log(max(n / max(s, 1), np.e))))   # m = O(s log(n/s))

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, s = 256, 5
    m   = measurements_needed(n, s)
    Phi = sample_gaussian(m, n, rng)
    x   = make_sparse(n, s, rng)
    xhat = recover_l1(Phi, Phi @ x)
    print(m, np.linalg.norm(xhat - x))
```
