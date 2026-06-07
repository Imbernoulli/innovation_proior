# Synthesis — Extremal constants for autocorrelation inequalities (the principled method)

## The task / object
Find the best constant $C_{opt}(w)$ such that for all $f \in L^1(\R)\cap L^2(\R)$,
$$\iint f(x)f(y)\,w(x-y)\,dx\,dy \le C_{opt}(w)\,\|f\|_1\|f\|_2,$$
with $w$ a symmetric, decreasing weight, $\|w\|_1=\|w\|_\infty=1$. Two cases of interest:
- $w=\chi_{[-1/2,1/2]}$ → this is the **average** (Barnard–Steinerberger) autocorrelation inequality: $\int_{-1/2}^{1/2}\int_\R f(x)f(x+t)\,dx\,dt$. Known: B–S proved $\le 0.91$; Madrid–Ramos improved to $0.864$; lower bound (example) $\ge 0.8$.
- $w=e^{-\pi x^2}$ → Gaussian-mean version.

Goal of the principled method: pin $C_{opt}$ between rigorous **upper and lower** bounds that nearly coincide (the paper gets $0.805580\!\pm\!9\cdot10^{-6}$ for the indicator; $0.71524\!\pm\!1.2\cdot10^{-5}$ for the Gaussian), **with a certificate**, rather than just producing a good test function (a lower bound) by search.

## Anti-pattern (context only, NOT the method)
AlphaEvolve (arXiv 2506.13131) and follow-ups (ThetaEvolve 2511.23473) attack the closely-related "second autocorrelation inequality" by **evolutionary search over step functions**: a 50,000-step highly irregular extremizer pushing the constant from $\ge 0.8892$ → $0.8962$ → $0.9610$. That is a one-sided, search-improved numeric bound (a better *test function* = better lower bound on a sup). It gives no matching upper bound, no certificate, and no proof of optimality. The principled discovery is the opposite move: reduce the variational problem to a structured finite-dimensional one with a **two-sided** rigorous bracket.

## The methodological leap (what to reconstruct)
A chain of reductions turning a hard nonconvex sup over an infinite-dim function space into a tractable eigenvalue computation with a rigorous discretization error bound:

1. **Sign + symmetry reduction (Riesz rearrangement).** The sup over all $f$ equals the sup over $f\ge0$, equals the sup over $f$ symmetric-decreasing. Because $w$ is symmetric-decreasing, Riesz's rearrangement inequality only increases the bilinear form $\iint f f\, w$ under symmetric-decreasing rearrangement while preserving $\|f\|_1,\|f\|_2$. → can restrict to nice $f$.

2. **Existence + compact support (Euler–Lagrange).** Restrict support to $[-R,R]$, maximize $\mathcal F(g)=\log\langle g|K_w|g\rangle - \log\|g\|_1 - \tfrac12\log\|g\|_2^2$. E–L equation gives, on support $[-a,a]$,
   $$\frac{2\, f^\star * w}{\langle f^\star|K_w|f^\star\rangle} = \frac1{\|f^\star\|_1} + \frac{f^\star}{\|f^\star\|_2^2}.$$
   Integrating + using $\int_{[-a,a]\times\R} f^\star w = \|w\|_1\|f^\star\|_1$ gives a bound on the support length: $a \le 2\|w\|_1^2/C_{opt,R}^2$. So extremizers are compactly supported with explicit bound; $C_{opt}=\lim_{R\to\infty}C_{opt,R}$ is attained.

3. **The $L^{1:2}$ obstruction.** $\|f\|_{L^{1:2}}^2:=\|f\|_1\|f\|_2$ is NOT a Hilbert norm; the denominator $\|f\|_1\|f\|_2$ mixes an $L^1$ and an $L^2$ norm → the Rayleigh quotient is not an eigenvalue problem. This is the wall.

4. **Linearization via a parameter $\lambda$ (AM–GM trick).** Introduce, for $\lambda>0$, the quadratic form
   $$\|f\|_{B_\lambda}^2 = \lambda\|f\|_2^2 + \lambda^{-1}\Big(\int|f|\Big)^2,\qquad \|f\|_{H_\lambda}^2 = \lambda\|f\|_2^2 + \lambda^{-1}\Big(\int f\Big)^2.$$
   AM–GM: $\min_\lambda(\lambda a^2 + \lambda^{-1}b^2)=2ab$, so $\|f\|_{L^{1:2}}^2 = \tfrac12\inf_\lambda \|f\|_{B_\lambda}^2$. And $\|f\|_{B_\lambda}=\|f\|_{H_\lambda}$ iff $f\ge0$. $H_\lambda$ IS a Hilbert space (parallelogram law). Define
   $$c_\lambda := \max_f \frac{2\langle f|K_w|f\rangle}{\lambda\langle f|f\rangle + \lambda^{-1}\langle f|1\rangle\langle 1|f\rangle},\qquad C_{opt} = \max_{\lambda>0} c_\lambda.$$
   For fixed $\lambda$ (and the $f\ge0$ constraint), $c_\lambda$ is a **generalized eigenvalue / Rayleigh quotient in a Hilbert space** = principled tractable relaxation. Bound $c_\lambda \le \min\{2\lambda, 2/\lambda\}$, max over $\lambda$ near $\lambda^*=\|f^\star\|_1/\|f^\star\|_2$.

5. **Discretization to step functions + a-priori error bound.** $V_\delta$ = step functions constant on $[n\delta,(n+1)\delta)$; $[f]_\delta$ = block-average projection. Key regularity from E–L: extremizer derivative bounded, $\|\hat f'\|_2 \le 4/(c_\lambda\lambda^{3/2})$. Combine with optimal Poincaré $\|\{f\}_\delta\|_2 \le \frac\delta\pi\|f'\|_2$ and Young's convolution inequality to get the **rigorous discretization error**:
   $$0 \le c_\lambda - c_{\lambda,\delta} \le \frac{16\,\delta^2}{\pi^2 c_\lambda \lambda^2}.$$
   This $\delta^2$ rate is what makes a *certified* upper bound possible — distinguishes this from the maximum problem (Cloninger–Steinerberger), whose extremizers are non-smooth so no $\delta^2$ gain, and whose method costs exponentially in discretization.

6. **Lipschitz-in-$\lambda$ regularity.** $c_\lambda$ is 1-Lipschitz; and if $C_{opt}=c_{\lambda^*}$, then $c_\lambda \ge 2c_{\lambda^*}/(\lambda^{-1}\lambda^*+\lambda{\lambda^*}^{-1})$. So a finite grid in $\lambda$ with spacing $\Delta\lambda$ certifies the max over all $\lambda$.

7. **Solving each fixed-$\lambda$ discrete problem (the LP/eigenvalue + dual structure).** On $V_\delta$ over $[-a,a)$ with $N$ cells: the $H_\lambda$ quadratic form is $\langle f|A_\lambda^2|f\rangle$ where $A_\lambda = \sqrt\lambda\,\mathrm{Id} + b_\lambda |1\rangle\langle1|$ (rank-1 update; $b_\lambda$ the positive root of $\lambda^{-1}=2\sqrt\lambda\,b_\lambda + 2a\,b_\lambda^2$). Whitening $g=A_\lambda f$ turns the ratio into a plain Rayleigh quotient of the symmetric matrix $M_\lambda = 2A_\lambda^{-1} K_w A_\lambda^{-1}$:
   $$c_{\lambda,\delta}^{\text{unconstrained}} = \lambda_{\max}(M_\lambda),\quad\text{found by the power method.}$$
   The $f\ge0$ constraint: by discrete Riesz rearrangement the extremizer is symmetric-decreasing; restrict to the minimal contiguous support of length $\ell$ (sweep $k=1..N$), solve the eigenproblem on each, keep the largest eigenvalue whose eigenvector is nonnegative. This is the "support-restriction" that resolves the inequality constraint exactly. The discrete convolution kernel $\tilde w(s) = \delta^{-2}\int_s^{s+\delta} w(t)(\delta-|t-s|)\,dt$ (triangular smoothing of $w$).

8. **Two-sided certificate.**
   - **Upper bound** on $C_{opt}$: from the certified relation $c_\lambda \le c_{\lambda,\delta} + 16\delta^2/(\pi^2 c_\lambda\lambda^2)$ plus the Lipschitz-in-$\lambda$ control over the grid.
   - **Lower bound** on $C_{opt}$: any admissible discrete $f\ge0$ plugged into the original quotient is a valid test function → $C_{opt} \ge$ that value. The computed eigenvector serves as the certificate.
   The gap $<9\cdot10^{-6}$ for $\chi$, $<1.2\cdot10^{-5}$ for the Gaussian. Polynomial cost in $1/\delta$ (vs. exponential for the max problem).

9. **(Conjectured) fixed-point iteration.** E–L rewrites as
   $$\frac{f^\star}{\|f^\star\|_2^2} = \max\!\Big(\frac{2 f^\star*w}{\langle f^\star|K_w|f^\star\rangle} - \frac1{\|f^\star\|_1},\,0\Big),$$
   a fixed-point map (project-positive-part of a convolution). Converges in practice, hundreds of times faster, gives the matching lower-bound column; convergence unproven.

## Lineage / baselines
- **Cilleruelo–Ruzsa–Vinuesa (CRV, 2010)**: connected the MAX autocorrelation constant $c_{\max}$ to Sidon-set ($B_h[g]$) asymptotics — the seed motivation.
- **Barnard–Steinerberger (2020, JNT 207)**: proposed the AVERAGE problem (relaxation of the max). Proved $\int\!\!\int_{|t|\le1/2} ff \le 0.91\|f\|_1\|f\|_2$ via Fourier: $\int_{-1/2}^{1/2}\widehat{|f|^2\text{-corr}} = \int |\hat f(\xi)|^2 \frac{\sin\pi\xi}{\pi\xi}d\xi$, then Cauchy–Schwarz / Hölder. Lower bound $0.8$ by example.
- **Madrid–Ramos (2021, CPAA 20(1))**: improved $0.91\to0.864$ via the **sharp Hausdorff–Young (Beckner) inequality** in a dual Fourier formulation, optimizing over $p$. Also Gaussian-mean $(8a/27\pi)^{1/4}$, and a Fourier/measure-theoretic existence argument; conjectured (Conj 1.4) that extremizers exist and are compactly supported.
- **Cloninger–Steinerberger (2017)**: computational attack on the MAX problem, $c_{\max}\ge1.28$; cost exponential in discretization, non-smooth extremizers.
- **Matolcsi–Vinuesa (2010)**: $c_{\max}\le1.52$ upper bound, numeric extremizers (non-smooth).
- **Carlen–Jauslin–Lieb–Loss**: $f\ge f*f$ convolution inequality (related autoconvolution).

## Design decisions → why
- **Why $L^{1:2}$ (mixed $L^1\|L^2$) denominator?** Because the inequality compares the bilinear autocorrelation (scales like mass$^2$) against $\|f\|_1\|f\|_2$ — the natural scale-invariant normalization for nonnegative $f$; makes the constant dimensionless.
- **Why introduce $\lambda$ rather than work directly?** $\|f\|_1\|f\|_2$ is not a quadratic form → no spectral theory. The $\lambda$-family $\lambda\|f\|_2^2 + \lambda^{-1}(\int f)^2$ is quadratic for each $\lambda$, and its $\inf_\lambda$ recovers $2\|f\|_1\|f\|_2$ by AM–GM. Converts one hard nonquadratic problem into a 1-parameter family of eigenvalue problems.
- **Why $H_\lambda$ vs $B_\lambda$ (drop the absolute value)?** $B_\lambda$ uses $\int|f|$, nonlinear; $H_\lambda$ uses $\int f$, a linear functional → genuine Hilbert (inner-product) space. They agree exactly on $f\ge0$, which is the admissible cone, so nothing is lost.
- **Why step functions / block-average projection (not splines, not Fourier truncation)?** The block average is an $L^2$- and $H_\lambda$-orthogonal projection, $\|[f]_\delta\|_1=\|f\|_1$ exactly (mass-preserving), $\|[f]_\delta\|_2\le\|f\|_2$ → never increases the constraint norm, so $c_{\lambda,\delta}\le c_\lambda$ automatically (one-sided, the right side for an upper bound). The kernel becomes a finite symmetric matrix.
- **Why the rank-1 whitening $A_\lambda=\sqrt\lambda\,Id+b_\lambda|1\rangle\langle1|$?** $H_\lambda$ form $=\lambda I + \lambda^{-1}|1\rangle\langle1|$ is identity-plus-rank-one; its square root is again identity-plus-rank-one with $b_\lambda$ solving a quadratic. Whitening turns generalized eigenproblem into standard symmetric eigenproblem → power method applies, $O(N^2)$ per iter, no dense solve.
- **Why support-restriction handles $f\ge0$?** Unconstrained top eigenvector may be sign-changing. Discrete Riesz ⇒ a nonnegative symmetric-decreasing maximizer exists; it lives on a contiguous interval; on the interior the constraint is inactive so it is the top eigenvector of the sub-block. Sweeping support length $\ell$ and taking the nonnegative one recovers the constrained optimum exactly. (This is the combinatorial/LP-flavored part: enumerate the active set.)
- **Why $\delta^2$ error matters / why smoothness?** The certified upper bound needs $c_\lambda-c_{\lambda,\delta}=O(\delta^2)$. That needs $\|f'\|_2$ controlled, which comes from the E–L equation's smoothing ($f \propto$ positive part of $(\lambda+\lambda^{-1}I)^{-1} k_w f$, and $k_w$ smooths). For the MAX problem extremizers aren't smooth → only $O(\delta)$ and exponential cost; that's why the average problem is the tractable one.
- **Why power method not full eig?** Only need top eigenvalue/vector; $M_\lambda$ is dense $N\times N$ but power iteration is $O(N^2)$ per step and converges fast (spectral gap). Polynomial in $1/\delta$.
- **Why a grid in $\lambda$ suffices?** $c_\lambda$ is 1-Lipschitz and has the explicit lower envelope around $\lambda^*$, so $\max_\lambda c_\lambda$ on a $\Delta\lambda$ grid is certified to within $O(\Delta\lambda)$.

## Code grounding
The accompanying repo (github.com/jaumededios/suprema-autocorrelations) hosts a notebook implementing Section 3; the public repo currently exposes only README/LICENSE, so the final code is grounded directly on the paper's explicit Section 3.2–3.3 algorithm (kernel matrix $\tilde w$, whitening $A_\lambda$, $M_\lambda=2A_\lambda^{-1}K_wA_\lambda^{-1}$, power method, support sweep, $\lambda$-grid, $\delta^2$ error bound), using numpy/scipy primitives. Flag: notebook source not retrievable → code is reconstructed from the paper's algorithm, not copied.
