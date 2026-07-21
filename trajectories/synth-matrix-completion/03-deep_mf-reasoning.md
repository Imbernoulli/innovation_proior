The convex baseline answered the question I set it, sharply. On rank3-50 explicit nuclear-norm
minimization landed `test_rel_fro = 0.250` against depth-2's `0.452`, and on rank5-100 `0.311` against
`0.596` — a nearly identical fractional improvement ($\approx45\%$, $\approx48\%$) in both well-sampled
environments, exactly what targeting the nuclear-norm fit directly rather than approximating it with Adam
should look like. So the floor's diagnosis is confirmed: depth-2's deficit on the easy and canonical
environments was a *bias-strength* problem, not a data problem. And the `train_mse` column flipped
completely — depth-2 sat at $10^{-9}$ down to $10^{-15}$ (exact interpolation), SVT rose to $0.073$,
$0.189$, $0.554$ (no interpolation at all), the $\tfrac12\|X\|_F^2$ regularizer trading data-fit for
spectral shrinkage as designed, and its relative training residual $\sqrt{0.073}\approx 0.27$ on
rank3-50 essentially matches its $0.250$ test error — the shrink leaves the *same* relative error on
seen and unseen entries alike.

But look at what nuclear norm *could not* do, because that is the crucial reading. On rank3-50 it still
leaves `0.250` — a quarter of the matrix wrong — when $30\%$ of a rank-3 matrix is plenty to recover it
nearly exactly. On rank10-200 it got `0.928`, a hair worse than depth-2's `0.923` and a statistical tie:
the regime where the 11-iteration truncation and the information floor together mean nuclear-norm
strength recovers nothing. So both rungs top out at nuclear-norm strength — depth-2 *approximates* it
(badly, via Adam), SVT *targets* it (cleanly, via the convex program) — and neither beats it, because
neither tries to. The residual $0.250$ on an environment with $2.6\times$ oversampling is the proof that
the *ceiling itself* is what is left to beat: minimum-nuclear-norm and minimum-rank part ways exactly in
the data-poor regime that matters, and the convex surrogate leaves recoverable structure on the table.

The next move is the direction I deferred at step 1: not to approximate nuclear norm, and not to target
it, but to make the implicit bias *stronger than* it. The obvious alternative is an explicit *nonconvex*
surrogate — a Schatten-$p$ quasi-norm $\sum_i\sigma_i^p$ with $p < 1$, or iteratively-reweighted nuclear
norm — which sits closer to rank and would beat SVT in the data-poor regime. But it buys that with
nonconvexity (spurious local minima, a reweighting schedule and a choice of $p$ to babysit) and it still
puts the low-rank preference in *by hand* as a penalty, the thing this trajectory has avoided since the
floor. The implicit-bias route promises the same stronger-than-nuclear-norm behavior as an emergent
property of an unpenalized, unconstrained factorization, with the single free choice being an integer
depth. So the lever I reach for is depth — and the real question is whether it is genuinely *sharper*
than nuclear norm or merely a re-derivation of it.

The tidy first guess is that if depth-2 corresponds to nuclear norm (Schatten-1), depth-$N$ corresponds
to Schatten-$p(N)$ with $p$ shrinking toward $0$ as $N$ grows, since
$\|X\|_{S_p}^p = \sum_r \sigma_r^p \to \text{rank}(X)$ as $p\to0$ — depth as a continuous knob
interpolating the relaxation from nuclear norm down toward rank. Let me test it in the case I can solve:
commuting matrix sensing, balanced near-zero init, gradient flow on the product $W = W_N\cdots W_1$.
Diagonalizing the measurements and starting from $\alpha^N I$ keeps the product diagonal, each entry
following a scalar ODE $\dot s = N(s^2)^{(N-1)/N} g(t)$; integrating for $N\ge3$ and taking $t\to\infty$
then $\alpha\to0$, a nonzero limit entry (since $\alpha^N\to0$) forces the dual
$\mathcal A^\dagger(\nu_\infty)\to1$ on the support, which paired with primal and dual feasibility is
exactly the weak-duality condition of the SDP $\min\langle I, W\rangle$ s.t. $\mathcal A(W)=y$,
$W\succeq0$ — i.e. nuclear-norm minimization. So the deep-factorization limit is *again* a
minimum-nuclear-norm solution, the same conclusion as depth-2. The Schatten-$p$ hypothesis is in
trouble: if the bias were Schatten-$p<1$ it would have to *disagree* with nuclear-norm minimization, but
here it provably agrees.

The contradiction is sharp under a perturbation. Take the PSD min-trace (= min-nuclear-norm) point
$\bar W = \text{diag}(1,1,0,\dots)$; perturbing $\varepsilon$ into the $(1,2),(2,1)$ entries stays
feasible and PSD with eigenvalues $1\pm\varepsilon$, and
$\|\hat W_\varepsilon\|_{S_p}^p = (1+\varepsilon)^p + (1-\varepsilon)^p < 2$ by strict concavity of
$x^p$. So $\bar W$ — the point the factorization goes to — is not even a *local* Schatten-$p$ minimizer
for any $0<p<1$. No single matrix norm or quasi-norm captures the bias. The commuting-sensing world is
special: it is exactly where min-nuclear-norm and min-rank coincide, so a norm-based account cannot be
falsified there. In the data-poor completion regime, where my numbers show nuclear norm visibly failing,
they diverge — and that is where depth must help. So I stop hunting for the norm and analyze the
*dynamics*.

Because the over-parameterization $\phi(W_1,\dots,W_N) = \ell(W_N\cdots W_1)$ is analytic, gradient flow
keeps $W(t)$ analytic, which admits an analytic SVD $W = USV^\top$ I can differentiate. Sandwiching
$\dot W$ by $u_r^\top(\cdot)v_r$ and using $\langle u_r,\dot u_r\rangle = 0$ gives the singular-value
velocity directly, and substituting the end-to-end dynamics (where $[WW^\top]^{(j-1)/N}$ and
$[W^\top W]^{(N-j)/N}$ contribute $N$ identical terms with exponents summing to $(N-1)/N$):

$$\dot\sigma_r(t) = -N\,(\sigma_r^2(t))^{(N-1)/N}\,\big\langle\nabla\ell(W(t)),\,u_r(t)v_r(t)^\top\big\rangle.$$

Depth $N$ enters the value dynamics, given the current $W$, *only* through the factor
$N(\sigma_r^2)^{(N-1)/N}$, exponent $2-2/N$. For $N=1$ the factor is $1$: every mode treated alike, the
flat Frobenius-like behavior of descending on $X$. For $N\ge2$ it is a power of the singular value's own
magnitude multiplying its velocity — large modes amplified, small modes attenuated, the gap sharpening
as $N$ grows (exponent climbing from $1$ at $N=2$ toward $2$). This is rich-get-richer on the spectrum,
and it is *not* a fixed functional being minimized — the same $W$ reached along different trajectories
gives different dynamics, which is exactly why no norm captures it. (The value dynamics depend on the
vectors through $\langle\nabla\ell, u_r v_r^\top\rangle$; deriving $\dot U,\dot V$ shows stationary
vectors require $U^\top\nabla\ell(W)V$ diagonal, i.e. gradient flow rotates $W$'s singular vectors into
alignment with the gradient's, after which the values evolve by a decoupled scalar ODE.)

Making the low-rank-intensifies-with-depth claim quantitative: eliminating the shared time-factor
between two aligned modes and integrating gives, for the faster mode $\to\infty$, a linear coupling
$\sigma_{r_1} = \alpha\sigma_{r_2} + c$ at $N=1$ (no bias); a power law $\sigma_{r_1} = c\,\sigma_{r_2}^\alpha$
with $0<\alpha<1$ at $N=2$ (the weak mode grows polynomially slower — the nuclear-norm-strength regime I
have seen twice); and for $N\ge3$, $\sigma_{r_1} = (\alpha\,\sigma_{r_2}^{-(N-2)/N} + c)^{-N/(N-2)}$, so
as $\sigma_{r_2}\to\infty$ the weak mode *saturates* at a finite asymptote — lower the larger $N$ is.
That is the hard low-rank bias: depth $\ge3$ does not merely slow the small modes, it *caps* them,
producing a few large singular values, a sharp shoulder, and the rest frozen near zero — the structure
nuclear norm cannot produce, and exactly what I need to push past the $0.25$ ceiling SVT left on
rank3-50.

The transition is at $N=3$, and that fixes the depth. The exponent $2-2/N$ is $0$ at $N=1$, $1$ at
$N=2$, $4/3$ at $N=3$, $3/2$ at $N=4$, climbing toward $2$. The jump from $N=2$ to $N=3$ raises it by
$0.33$ and, more importantly, *changes the kind* of solution — from a power law (weak mode unbounded,
just slower) to the saturating form (weak mode frozen at a cap). The jump from $N=3$ to $N=4$ raises it
by only $0.17$ and leaves the form unchanged, a slightly lower cap. So the qualitative transition
happens exactly at $N=3$, everything past it a diminishing quantitative tweak, and each extra layer
multiplies the per-iteration cost of the $n\times n$ end-to-end product for a shrinking return. Depth $3$
is the cheapest depth on the far side of the power-law-to-saturation transition — the `depth=3` the fill
ships.

The other choices follow, each against its knob. Near-zero init, so every $\sigma_r(0)$ starts on the
throttled plateau and the dynamics get to *select* which modes switch on (the data-aligned ones);
initialize large and modes start off the throttle and cannot be separated. Balanced init
($W_{j+1}^\top W_{j+1} = W_j W_j^\top$), which near-zero Gaussian factors satisfy approximately, for the
clean end-to-end ODE. Full hidden dimension $=n$, since capping the inner dimension would be *explicit*
low-rank factorization and the whole point is that low rank emerges implicitly from the depth dynamics.
The init magnitude is deliberately unchanged from the floor: with per-layer std
$\texttt{init\_scale}^{1/N}\cdot n^{-1/2}$, a depth-$N$ end-to-end entry is a sum of $n^{N-1}$ $N$-fold
Gaussian products, variance $n^{N-1}\cdot(\texttt{init\_scale}^{2/N}/n)^N = \texttt{init\_scale}^2/n$ —
the $N$ cancels — so the end-to-end Frobenius norm is $\texttt{init\_scale}\sqrt n$ *independent of
depth*, $0.01$ at $n=100$ for both the depth-2 floor and this depth-3 fill. That is the point of the
$\texttt{init\_scale}^{1/N}$ power: it holds the starting product at the same near-zero magnitude no
matter how many factors multiply, so the only thing changing between the floor and here is the *shape* of
the dynamics — the $2-2/N$ exponent — and not the starting energy.

The learning-rate drop from the floor's `5e-3` to `1e-3` is not cosmetic either. The bias *is* the
gradient flow, and Adam with a finite step is a discretization whose error walks the iterate off the
curved selecting manifold. Three composed layers make the end-to-end product a degree-3 function of the
parameters, so its curvature — and the product's sensitivity to a step in any single factor — is larger
than the depth-2 product's, and the same step size produces a larger off-manifold excursion. Cutting
`lr` by $5\times$ keeps the discretized trajectory tracking the flow whose saturating bias I am relying
on; the cost is iterations, but the saturating dynamics spend most of their time on the near-zero plateau
before the dominant modes switch on, so the smaller step mostly buys fidelity during the critical
switch-on rather than slowing a long tail. If the budget were the binding constraint I would see a
`train_mse` that fails to reach the $10^{-7}$ floor.

So the scaffold edit is a small, exact diff from the floor: the same `nn.Sequential` of
`nn.Linear(n,n,bias=False)` layers, the same `_e2e` left-to-right fold and Adam masked-MSE loop with the
same `1e-7` early stop, changing only `depth=3` (the $1/N$ init power now spread across three factors,
guarded by `if depth < 2: raise`) and `lr=1e-3`. No penalty, no rank cap; the entire difference from the
floor is one more layer and a smaller step. The full module is in the answer.

I stay cautious about rank10-200, because a sharper bias cannot manufacture information the samples do
not contain. That environment observes $4000$ entries of a rank-10 matrix with $3900$ degrees of
freedom — an oversampling ratio of $1.03$ — against a completion sample complexity
$nr\log n \approx 10\cdot200\cdot\ln200 \approx 10600$, nearly three times short; both lower rungs sat at
$\sim0.92$ there. The saturating depth-3 bias separates modes more sharply once the data pins down which
modes are real, but if $4000$ entries are too few to locate a 10-dimensional row-and-column space at all,
no spectral sharpening rescues it. So the clean tests are the two well-sampled environments, where the
data is not the excuse: if the saturating bias genuinely beats nuclear-norm strength it should crush
both — near-exact recovery where the truth is identifiable, an order of magnitude below the nuclear-norm
baseline — while rank10-200 stays high near $0.9$, the information floor asserting itself rather than the
method failing.
