# Fast and Sparse Johnson–Lindenstrauss Transforms

## Problem

The Johnson–Lindenstrauss lemma reduces dimension from $d$ to $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ with distortion $1\pm\varepsilon$, but the standard *dense* matrix $S$ costs $\Theta(k\cdot\|x\|_0)=\Theta(kd)$ per vector — the bottleneck of the search/regression pipelines it is meant to accelerate. Goal: keep the optimal $k$ and a genuine linear $\ell_2$ image, but apply the embedding far faster — in $O(d\log d)$ for dense inputs, and in time scaling with $\|x\|_0$ (with few random bits) for sparse/streaming inputs.

## Key idea

Two complementary constructions, dictated by *why* a naïve sparse matrix fails: it distorts a spiky vector ($x=e_i$ has all the embedding's information in one mostly-empty column).

- **Fast-JL (FJLT), $\Phi=PHD$.** Precondition with a randomized Walsh–Hadamard transform $HD$, which is orthogonal (zero added distortion) and — by the uncertainty principle — *smooths* any input to small $\|HDx\|_\infty$ with high probability; then apply a sparse Gaussian projection $P$, which is safe precisely because its input is now smooth. FFT makes $HD$ cost $O(d\log d)$.
- **Sparse-JL (graph / block).** For sparse/streaming inputs a preconditioner would destroy sparsity, so keep a *genuinely sparse* matrix with exactly $s$ nonzeros per column and hash **without replacement** — killing the within-column self-collisions that force the prior $\log^2$ factor — reaching column sparsity $s=\Theta(\varepsilon^{-1}\log(1/\delta))$, with matching lower bounds for these graph/block schemes.

## Fast Johnson–Lindenstrauss Transform

$\Phi=PHD$, a product of three $d$-related matrices (with $\Phi:\mathbb{R}^d\to\mathbb{R}^k$, $k=c\varepsilon^{-2}\log n$):

- $D=\mathrm{diag}(\pm1,\ldots,\pm1)$ with i.i.d. Rademacher diagonal.
- $H\in\mathbb{R}^{d\times d}$ normalized Walsh–Hadamard, $H_{ij}=d^{-1/2}(-1)^{\langle i-1,\,j-1\rangle}$ ($\langle\cdot,\cdot\rangle$ = dot product mod 2 of the bit expansions); $HD$ orthogonal, computable in $O(d\log d)$ via the GF$(2)^d$ FFT.
- $P\in\mathbb{R}^{k\times d}$ with entries i.i.d. $0$ w.p. $1-q$, else $N(0,q^{-1})$, sparsity $q=\min\{\Theta(\varepsilon^{p-2}\log^p n/d),\,1\}$ (for $\ell_2$: $q=\min\{\Theta(\log^2 n/d),1\}$).

**Smoothing lemma.** For unit $x$, with $u=HDx$ and $d\,a_i=\pm\sqrt d$, the MGF bound $\mathbb{E}\,e^{t d\,u_1}=\prod_i\cosh(t\sqrt d\,x_i)\le e^{t^2 d/2}$ gives, with $t=s$, $\Pr[|u_1|\ge s]\le 2e^{-s^2 d/2}$; at $s=\Theta(d^{-1/2}\sqrt{\log n})$ this is $\le 1/(20nd)$, so by a union bound $\max_{x\in X}\|HDx\|_\infty=O(d^{-1/2}\sqrt{\log n})$ w.p. $\ge 1-1/20$. Since $HD$ is orthogonal, $\|HDx\|_2=\|x\|_2$ exactly.

**FJLT lemma.** For a set $X$ of $n$ vectors, $\varepsilon<1$, $p\in\{1,2\}$, with probability $\ge 2/3$: (i) for all $x\in X$, $(1-\varepsilon)\alpha_p\|x\|_2^p\le\|\Phi x\|_p^p\le(1+\varepsilon)\alpha_p\|x\|_2^p$ with $\alpha_2=k$, $\alpha_1=k\sqrt{2\pi^{-1}}$; (ii) $\Phi$ runs in
$$O\!\big(d\log d+\min\{\,d\varepsilon^{-2}\log n,\ \varepsilon^{p-4}\log^{p+1}n\,\}\big)$$
operations. Amplify to $1-\delta$ by $O(\log(1/\delta))$ independent repetitions.

*Proof sketch ($\ell_2$).* Condition on the smoothness event $\|u\|_\infty\le s$, $m=s^{-2}$. Each output $y_1=\sum_j r_j b_j u_j$ with $(y_1\mid Z)\sim N(0,q^{-1}Z)$, $Z=\sum_j b_j u_j^2$, $\mathbb{E}Z=q$ (2-stability of the Gaussian). $\mathbb{E}[Z^t]$ is convex in $u^2\in\mathcal P=\{a:0\le a_j\le1/m,\sum a_j=1\}$, so it is maximized at a vertex $u^*$, where $Z^*\sim B(m,q)/m$ with $\mathrm{var}=q(1-q)/m$. Standard binomial tail bounds give, with the smoothness premise, $q/2\le Z_i\le 2q$ for all $k$ rows and $\sum_i y_i^2\in(1\pm\varepsilon)k$ w.p. $1-e^{-\Omega(\varepsilon^2k)}$; $k=c\varepsilon^{-2}\log n$ makes this $1-1/\mathrm{poly}(n)$. *Runtime:* $Dx$ is $O(d)$, $H(Dx)$ is $O(d\log d)$, $P(\cdot)$ is $O(|P|)$ with $|P|\sim B(dk,q)$, $\mathbb{E}|P|=O(\varepsilon^{p-4}\log^{p+1}n)$. $\square$

## Sparse Johnson–Lindenstrauss Transform

$S_{i,j}=\eta_{i,j}\sigma_{i,j}/\sqrt s$, with $\sigma$ Rademacher ($O(\log(1/\delta))$-wise independent) and $\eta_{i,j}$ the nonzero-indicator, **exactly $s$ per column**, hashed without replacement:
- **graph construction** — $s$ targets per column without replacement (bipartite graph, left-degree $s$);
- **block construction** — split $[k]$ into $s$ blocks of size $k/s$, one random target per block (= CountSketch with higher independence, but kept as a linear $\ell_2$ estimator). Seed length $O(\log(1/\delta)\log d)$.

For unit $x$, with $Z=\|Sx\|_2^2-1=\frac1s\sum_{r}\sum_{i\ne j}\eta_{r,i}\eta_{r,j}\sigma_{r,i}\sigma_{r,j}x_i x_j$ (error = collisions), it suffices that $\Pr[|Z|>2\varepsilon-\varepsilon^2]<\delta$.

**Theorem (preserved distortion).** With $k=\Theta(\varepsilon^{-2}\log(1/\delta))$ and $s=\Theta(\varepsilon^{-1}\log(1/\delta))$, both constructions satisfy the JL lemma: $\Pr[|\,\|Sx\|_2^2-1\,|>2\varepsilon-\varepsilon^2]<\delta$. The same hard inputs give $s=\Omega(\varepsilon^{-1}\log(1/\delta))$ lower bounds for these graph/block schemes.

*Proof sketch — analysis 1 (Hanson–Wright).* $Z=\sigma^T T\sigma$, $T$ block-diagonal, $\mathrm{tr}\,T=0$. Under the collision/code condition $\sum_r\eta_{r,i}\eta_{r,j}=O(s^2/k)$ for all $i\ne j$ (columns form an error-correcting code of relative distance $1-O(s/k)$, which is necessary and sufficient), one gets $\|T\|_F^2=O(1/k)$ and $\|T\|_2\le1/s$. Hanson–Wright, $\mathbb{E}\,|\sigma^TT\sigma|^\ell\le C^\ell\max\{\sqrt\ell\,\|T\|_F,\ \ell\,\|T\|_2\}^\ell$ with $\ell=\log(1/\delta)$, plus Markov, gives the bound $\le\delta$.

*Proof sketch — analysis 2 (combinatorial moment).* Expand $\mathbb{E}\,Z_r^t$, map monomials to even-degree labeled multigraphs, use $\mathbb{E}_\eta\prod\eta=(s/k)^v$ and a Cauchy–Schwarz induction over edge-additions to get $\mathbb{E}\,Z_r^t\le t(2e^2)^t\{(s/k)^2\text{ or }(t/\ln(k/s))^t\}\le t(2e^3)^t(s/k)^2t^t$. Across rows the $\eta$ are negatively correlated, so $\mathbb{E}\prod_iZ_{r_i}^{\ell_i}\le\prod_i\mathbb{E}\,Z_{r_i}^{\ell_i}$; assembling $\mathbb{E}\,Z^\ell\le(8e^3(\ell+1)/s)^\ell(\ell+1)\sum_q(s^2/(qk))^q$ and choosing $\ell=\Theta(\log(1/\delta))$, $s=\Theta(\varepsilon^{-1}\log(1/\delta))$, $k=2s^2/(e\ell)=\Theta(\varepsilon^{-2}\log(1/\delta))$ yields $\mathbb{E}\,Z^\ell\le(2\varepsilon-\varepsilon^2)^\ell\delta$; Markov closes it.

*Tightness and DKS lower bound.* For graph/block, if $s\le1/(2\varepsilon)$, a vector with $t=\lfloor1/(s\varepsilon)\rfloor$ equal entries incurs error $\ge2\varepsilon$ from one collision w.p. $\Omega(1/\log(1/\delta))\gg\delta$; if $s>1/(2\varepsilon)$, $(1/\sqrt2,1/\sqrt2,0,\dots)$ incurs error $2\varepsilon$ w.p. $>\delta$ unless $s=\Omega(\varepsilon^{-1}\log(1/\delta))$. For the DKS with-replacement scheme, the self-collision hard input $(1,0,\dots)$ gives $s=\Omega(\varepsilon^{-1}\lceil\log^2(1/\delta)/\log^2(1/\varepsilon)\rceil)=\tilde\Omega(\varepsilon^{-1}\log^2(1/\delta))$.

## Code

```python
import numpy as np

def fwht(a):
    """Normalized Walsh-Hadamard transform, O(d log d); len(a) a power of two."""
    a = a.astype(float).copy()
    n = len(a); h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    return a / np.sqrt(n)            # H orthogonal

class FastJL:
    """Phi = P H D, normalized from the p-th-power theorem."""
    def __init__(self, d, k, n, eps=1.0, p=2, seed=0):
        assert d & (d - 1) == 0
        assert p in (1, 2)
        rng = np.random.default_rng(seed)
        self.k = k
        self.D = rng.choice([-1.0, 1.0], size=d)
        logn = np.log(max(n, 2))
        q = min((eps ** (p - 2)) * (logn ** p) / d, 1.0)
        alpha = k if p == 2 else k * np.sqrt(2.0 / np.pi)
        self.output_scale = alpha ** (-1.0 / p)
        mask = rng.random((k, d)) < q
        self.rows, self.cols = np.nonzero(mask)
        self.vals = rng.standard_normal(len(self.rows)) / np.sqrt(q)
    def apply(self, x):
        u = fwht(self.D * x)                            # smooths ||u||_inf
        y = np.zeros(self.k)
        np.add.at(y, self.rows, self.vals * u[self.cols])
        return self.output_scale * y

class SparseJLBlock:
    """Exactly s nonzeros/column, one per block, no within-column collision."""
    def __init__(self, d, k, s, seed=0):
        assert k % s == 0
        rng = np.random.default_rng(seed)
        self.k, self.s, self.block = k, s, k // s
        self.rows = np.empty((d, s), dtype=int)
        self.signs = rng.choice([-1.0, 1.0], size=(d, s))
        for j in range(d):
            for b in range(s):
                self.rows[j, b] = b * self.block + rng.integers(self.block)
    def apply(self, x):
        y = np.zeros(self.k)
        for j in np.nonzero(x)[0]:                      # cost ~ nnz(x): streaming-friendly
            y[self.rows[j]] += self.signs[j] * x[j] / np.sqrt(self.s)
        return y
```

The two constructions split the original wound: `FastJL` removes the dense $O(kd)$ multiply for dense vectors via an $O(d\log d)$ FFT plus a touch of sparse arithmetic; `SparseJLBlock` makes the cost scale with $\|x\|_0$ (and the seed with $O(\log(1/\delta)\log d)$ bits) for sparse and streaming vectors — both at the optimal target dimension $k=\Theta(\varepsilon^{-2}\log(1/\delta))$.
