The convex baseline answered the question I set it, and the answer is sharp. On rank3-50 explicit
nuclear-norm minimization landed `test_rel_fro = 0.250` against depth-2's `0.452` — a clear improvement,
exactly as predicted: when I asked for the low-nuclear-norm fit *directly* instead of hoping Adam
approximated it, I reclaimed most of the gap on the generously-sampled environment. On rank5-100 it got
`0.311` against depth-2's `0.596` — again a decisive win for the explicit method. So the diagnosis from
the floor is confirmed: depth-2's deficit on the easy and canonical environments was a *bias-strength*
problem, not a data problem; the implicit bias of a depth-2 factorization really is only
nuclear-norm-strength, and run properly the convex program does better. But look at what nuclear norm
*could not* do, and this is the crucial reading. On rank3-50 it still leaves `0.250` — a quarter of the
matrix wrong — when $30\%$ of a rank-3 matrix is plenty of information to recover it nearly exactly. And
its `train_mse` is now *large* — `0.073`, `0.189`, `0.554` — it does not even fit the observed entries,
because the $\tfrac12\|X\|_F^2$ regularizer in the SVT objective trades data-fit for spectral shrinkage.
On rank10-200 it got `0.928`, essentially tied with depth-2's `0.923` and no better — the regime where
the truncated 11-iteration budget and the information floor together mean nuclear-norm-strength recovers
nothing.

Let me read the two rungs' numbers against each other before I decide the lever, because the pattern is
specific. On rank3-50 the error went $0.452\to0.250$, a $45\%$ reduction; on rank5-100 $0.596\to0.311$, a
$48\%$ reduction — nearly the same fractional gain in both well-sampled environments, which says the convex
program bought a *uniform* improvement over depth-2's approximation, exactly what targeting the
nuclear-norm fit directly instead of approximating it with Adam should look like. On rank10-200 the error
went $0.923\to0.928$ — a hair *worse*, statistically a tie — confirming that whatever nuclear-norm strength
buys, it buys nothing there. And the `train_mse` columns flipped completely: depth-2 sat at $10^{-9}$ down
to $10^{-15}$ (exact interpolation), SVT rose to $0.073$, $0.189$, $0.554$ (no interpolation at all). I can
even check that SVT's training residual is consistent with its test error: on rank3-50 the relative
training residual is $\sqrt{0.073}\approx0.27$, essentially equal to its $0.250$ test error, so SVT leaves
the *same* relative error on observed and unobserved entries — its $\tfrac12\|X\|_F^2$ regularizer shrinks
the whole spectrum uniformly, seen and unseen alike. So the two rungs bracket the ceiling from below in two
different ways and neither breaks it, and the residual $0.250$ on an environment with $2.6\times$
oversampling is the proof that the ceiling — not the data, not the optimizer — is what is left to beat.

So both rungs so far top out at nuclear-norm strength. Depth-2 *approximates* it (badly, via Adam); SVT
*targets* it (cleanly, via the convex program). Neither beats it, because neither tries to. And the
residual error on the well-sampled environments — $0.25$ on rank3-50, where the truth is recoverable —
says the ceiling itself is the problem: minimum-nuclear-norm and minimum-rank *part ways* exactly in the
data-poor regime that matters, and the convex surrogate leaves recoverable structure on the table. The
natural next move is the one direction I deferred at step 1: do not approximate nuclear norm, and do not
target it — make the implicit bias *stronger than* nuclear norm. The lever is depth. If a depth-2
factorization buys nuclear-norm-strength bias, more layers should sharpen it toward rank, and reclaim the
error nuclear norm cannot.

Before I commit to depth I should name the obvious alternative and say why I am not taking it. If nuclear
norm is too weak because it is the convex *envelope* of rank — the tightest convex thing, but still a
relaxation — the direct fix is an explicit *nonconvex* surrogate: minimize a Schatten-$p$ quasi-norm
$\sum_i\sigma_i^p$ with $p < 1$, or iteratively-reweighted nuclear norm, which pushes small singular values
toward zero harder than the $\ell_1$-on-the-spectrum that nuclear norm applies. It sits closer to rank and
would, in principle, beat SVT in the data-poor regime. But it buys that with nonconvexity — spurious local
minima, sensitivity to the reweighting schedule and to the choice of $p$, and a fit surface I would have to
babysit — and it still puts the low-rank preference in *by hand* as a penalty, which is the thing this
trajectory has been trying to avoid since the floor. The implicit-bias route promises the same
"stronger-than-nuclear-norm" behavior as an emergent property of an unpenalized, unconstrained
factorization, with the single free choice being an integer depth. If a depth knob delivers a bias sharper
than nuclear norm without any explicit penalty, that is both cheaper to run and more to the point of the
benchmark than hand-tuning a nonconvex regularizer. So the lever I reach for is depth, and the real
question is whether it *is* sharper than nuclear norm or merely a re-derivation of it — which is what the
next stretch of reasoning has to settle.

My first guess is the tidy one: if depth-2 corresponds to nuclear norm (Schatten-1), maybe depth-$N$
corresponds to Schatten-$p(N)$ for some $p$ shrinking toward $0$ as $N$ grows, since
$\|X\|_{S_p}^p = \sum_r \sigma_r^p \to \text{rank}(X)$ as $p\to0$. Depth would be a continuous knob
interpolating the relaxation from nuclear norm down toward rank. Let me try to confirm it on the case I
can solve — commuting matrix sensing, balanced near-zero init, gradient flow on the end-to-end product
$W = W_N\cdots W_1$. The product obeys $\dot W = -\sum_{j=1}^N [WW^\top]^{(j-1)/N}\nabla\ell(W)
[W^\top W]^{(N-j)/N}$. Diagonalizing the commuting measurements and starting from $\alpha^N I$ keeps the
product diagonal for all time, so each diagonal entry follows a scalar ODE $\dot s = N(s^2)^{(N-1)/N}
g(t)$. A sign lemma (the exponent $1-1/N \ge \tfrac12$) shows each entry keeps the sign of its init and a
near-zero entry is held near zero by the throttle — so the product stays PSD and analyzable on the cone.

Integrate the scalar ODE for $N\ge3$ and take $t\to\infty$ then $\alpha\to0$. For a diagonal entry of
the limit to be nonzero (since $\alpha^N\to0$), the bracket $[I - \mathcal A^\dagger(\nu_\infty(\alpha))]$
must go to zero there, forcing $\mathcal A^\dagger_{kk}(\nu_\infty)\to1$ on the support. Undiagonalizing,
$\langle I - \mathcal A^\dagger(\nu_\infty), W^\*\rangle\to0$, and pairing that with primal feasibility
and dual feasibility ($\mathcal A^\dagger(\nu)\preceq I$) is exactly the weak-duality gap of the SDP
$\min\langle I, W\rangle$ s.t. $\mathcal A(W) = y, W\succeq0$ — which for PSD $W$ *is* nuclear-norm
minimization. So $\|W^\*\|_\* = \text{OPT}$: the deep-factorization limit is a minimum-nuclear-norm
solution. But that is the *same* conclusion as depth-2. Extending the argument to arbitrary $N\ge3$ still
says nuclear norm. The Schatten-$p$ hypothesis is in trouble: if the bias were Schatten-$p<1$, it would
have to *disagree* with nuclear-norm minimization, but here it provably *agrees*.

Let me make the contradiction sharp. Take diagonal $A_i$ encoding $W_{11}=W_{22}$ and $W_{11}=W_{kk}+1$;
among PSD matrices the min-trace (= min-nuclear-norm) point is $\bar W = \text{diag}(1,1,0,\dots)$ with
$\|\bar W\|_{S_p}^p = 2$. Perturb $\varepsilon$ into the $(1,2)$ and $(2,1)$ entries: still feasible,
PSD, eigenvalues $1\pm\varepsilon$, and $\|\hat W_\varepsilon\|_{S_p}^p = (1+\varepsilon)^p +
(1-\varepsilon)^p < 2$ by strict concavity of $x^p$. So $\bar W$ is not even a local Schatten-$p$
minimizer for any $0<p<1$ — the factorization provably goes to a point those quasi-norms want to move
*away* from. No single matrix norm or quasi-norm captures the bias. The commuting-sensing world is
special: it is exactly where min-nuclear-norm and min-rank coincide, so a norm-based account cannot be
falsified there. In the data-poor completion regime — where my numbers show nuclear norm visibly failing
— they diverge, and that is where depth must help. So I stop hunting for the norm and analyze the
*dynamics*.

Set up the singular-value dynamics. Because the over-parameterization $\phi(W_1,\dots,W_N) = \ell(W_N
\cdots W_1)$ is analytic, gradient flow keeps $W(t)$ analytic, which admits an analytic SVD $W = USV^\top$
I can differentiate. From $\dot W = U\dot S V^\top + \dot U S V^\top + US\dot V^\top$, sandwiching by
$u_r^\top(\cdot)v_r$ and using $\langle u_r,\dot u_r\rangle = 0$ gives $\dot\sigma_r = u_r^\top \dot W
v_r$ — the singular-value velocity is the gradient-flow velocity projected onto the $r$-th singular
direction. Substituting the end-to-end dynamics, $WW^\top = US^2U^\top$ and $W^\top W = VS^2V^\top$, so
$[WW^\top]^{(j-1)/N}$ and $[W^\top W]^{(N-j)/N}$ pick out the $r$-th diagonal of each power, and the
exponents add to $(N-1)/N$ independent of $j$ — $N$ identical terms:

$$\dot\sigma_r(t) = -N\,(\sigma_r^2(t))^{(N-1)/N}\,\big\langle\nabla\ell(W(t)),\,u_r(t)v_r(t)^\top\big\rangle.$$

There it is. Depth $N$ enters the value dynamics, given the current $W$, *only* through the factor
$N(\sigma_r^2)^{(N-1)/N}$, exponent $2-2/N$. For $N=1$ (no factorization) the factor is $1$: bare
dynamics, every mode treated alike — that is the flat, Frobenius-like behavior of descending on $X$. For
$N\ge2$ the factor is a *power of the singular value's own magnitude* multiplying its velocity: large
modes are amplified, small modes attenuated, and the gap sharpens as $N$ grows (exponent climbs from $1$
at $N=2$ toward $2$). This is the mechanism — rich-get-richer on the spectrum — and crucially it is *not*
a fixed functional of $W$ being minimized; the same $W$ reached along different trajectories gives
different dynamics, which is precisely why no norm captures it.

The value dynamics depend on the *vectors* through $\langle\nabla\ell, u_r v_r^\top\rangle$, and the
vectors move too, so I need them to settle. Deriving $\dot U, \dot V$ and forming $U^\top\dot U S -
SV^\top\dot V = -\bar I\odot G\odot[U^\top\nabla\ell V]$ (with $G$ having no zero entries) shows that if
the vectors are stationary, $U^\top\nabla\ell(W)V$ must be *diagonal* — the singular vectors of $W$ align
with those of $\nabla\ell(W)$. That is the converse alignment statement: stationary $\Rightarrow$
aligned. So gradient flow rotates $W$'s singular vectors into alignment with the gradient's, and once
aligned the singular values evolve by the decoupled scalar ODE.

Make the low-rank-intensifies-with-depth claim quantitative on a single-measurement toy with stationary,
aligned vectors: $\dot\sigma_r = -N(\sigma_r^2)^{1-1/N}\delta(t)e_r\rho_r$. Eliminating the shared
time-factor $\delta(t)$ between two modes and integrating gives, for $0<\alpha<1$ (mode $r_2$ the faster
one): linear coupling $\sigma_{r_1} = \alpha\sigma_{r_2} + c$ at $N=1$ (no bias — the weak mode grows in
lockstep); a power law $\sigma_{r_1} = c\,\sigma_{r_2}^\alpha$ at $N=2$ (the weak mode grows
polynomially slower — *some* bias, the nuclear-norm-strength regime I have already seen twice); and for
$N\ge3$, $\sigma_{r_1} = (\alpha\,\sigma_{r_2}^{-(N-2)/N} + c)^{-N/(N-2)}$, so as $\sigma_{r_2}\to\infty$
the weak mode *saturates* at a finite asymptote — lower the larger $N$ is. That is the hard low-rank
bias: depth $\ge3$ does not merely slow the small modes, it *caps* them. A few large singular values, a
sharp shoulder, the rest frozen near zero. This is the structure nuclear norm cannot produce — and it is
exactly what I need to push past the $0.25$ ceiling SVT left on rank3-50.

Let me put numbers on "saturates at a finite asymptote, lower the larger $N$," because that is the whole
case for depth $\ge 3$. The exponent in the value dynamics is $2 - 2/N$: $N = 1$ gives $0$ (no throttle,
flat), $N = 2$ gives $1$, $N = 3$ gives $4/3\approx1.33$, $N = 4$ gives $3/2$, and $N\to\infty$ climbs
toward $2$. The marginal sharpening per added layer is front-loaded: the jump from $N = 2$ to $N = 3$
raises the exponent by $0.33$ and, more importantly, *changes the kind* of solution — from a power law
$\sigma_{r_1} = c\,\sigma_{r_2}^\alpha$ (weak mode grows without bound, just slower) to the saturating form
$\sigma_{r_1} = (\alpha\,\sigma_{r_2}^{-(N-2)/N} + c)^{-N/(N-2)}$ (weak mode frozen at a finite cap). The
jump from $N = 3$ to $N = 4$ raises the exponent by only $0.17$ and leaves the *form* unchanged — still
saturating, merely a slightly lower cap. So the qualitative transition happens exactly at $N = 3$, and
everything past it is a diminishing quantitative tweak. That is the arithmetic reason to ship depth $3$ and
not $4$ or $10$: it is the cheapest depth on the far side of the power-law-to-saturation transition, and
each extra layer past it multiplies the per-iteration cost of the $n\times n$ end-to-end product for a
shrinking return.

Every design choice now follows, and I check each against the harness knob. **Depth $N\ge3$:** the bias
exponent $2-2/N$ is monotone in $N$; $N=2$ gives only the power-law gap (the nuclear-norm regime my first
two rungs lived in), $N\ge3$ gives the saturating cap that beats nuclear norm in the data-poor regime,
and empirically $N=4$ is indistinguishable from $N=3$ — so depth $3$ is the cheapest depth buying the
strong bias. That is the `depth=3` the fill ships. **Near-zero init:** every $\sigma_r(0)$ must be tiny
so every mode starts on the throttled plateau and the dynamics get to *select* which modes switch on (the
data-aligned ones); initialize large and modes start off the throttle and cannot be separated. **Balanced
init:** the clean end-to-end ODE requires $W_{j+1}^\top W_{j+1} = W_j W_j^\top$ at init; near-zero
Gaussian factors satisfy it approximately. **Full hidden dimension $=n$:** I deliberately do *not* cap
the inner dimension — capping it would be *explicit* low-rank factorization, and the whole point is that
low rank emerges *implicitly* from the depth dynamics. **Small step, many iters:** Adam with a small step
approximates the gradient flow and drives training error to interpolation; the bias supplies the rest.

There is an arithmetic check hiding in the initialization that I want to run, because it is what makes the
depth-2-to-depth-3 comparison clean. The per-layer std is $\texttt{init\_scale}^{1/N}\cdot n^{-1/2}$, so
$\text{std}^2 = \texttt{init\_scale}^{2/N}/n$. For the depth-$N$ end-to-end product, an entry is a sum of
$n^{N-1}$ independent $N$-fold products of these Gaussians, so its variance is $n^{N-1}\cdot(\text{std}^2)^N
= n^{N-1}\cdot\texttt{init\_scale}^2/n^N = \texttt{init\_scale}^2/n$ — the $N$ cancels. The end-to-end
Frobenius norm is $\sqrt{n^2\cdot\texttt{init\_scale}^2/n} = \texttt{init\_scale}\sqrt n$, *independent of
depth*. At $n = 100$ with `init_scale = 1e-3` that is $0.01$ for both the depth-2 floor and this depth-3
fill, against a target $\|M^\*\|_F = 100$. So the $\texttt{init\_scale}^{1/N}$ power is doing precise work:
it holds the starting product at the same near-zero magnitude no matter how many factors I multiply, which
means the only thing changing between the floor and this rung is the *shape* of the dynamics — the
$2 - 2/N$ exponent — and not the starting energy. When the numbers come back better, I will know it is the
saturating bias and not a lucky change in initialization scale.

The learning-rate drop from the floor's `5e-3` to `1e-3` is not cosmetic either, and I can say why in the
same dynamical language. The bias *is* the gradient flow; Adam with a finite step is a discretization of
it, and the discretization error is what walks the iterate off the curved selecting manifold. Three
composed layers make the end-to-end product a degree-$3$ function of the parameters, so its curvature — and
the sensitivity of the product to a step in any single factor — is larger than the depth-2 product's, which
means the same step size produces a larger off-manifold excursion. Cutting `lr` by $5\times$ compensates:
it keeps the per-step move small enough that the discretized trajectory tracks the flow whose saturating
bias I am relying on. The cost is iterations, but the saturating dynamics spend most of their time on the
near-zero plateau before the dominant modes switch on, so a smaller step mostly buys fidelity during the
critical switch-on rather than slowing a long tail. If the budget were the binding constraint I would see
it in a `train_mse` that fails to reach the $10^{-7}$ floor; the design bet is that `1e-3` is small enough
for fidelity and still large enough to interpolate within budget.

Now the literal scaffold edit, and the diff against the depth-2 floor is small and exact — which is the
point. The fill is the same `nn.Sequential` of `nn.Linear(n,n,bias=False)` layers, the same `_e2e`
left-to-right fold, the same Adam masked-MSE loop with the same `train_thres = 1e-7` early stop. Two
things change. `depth=3` instead of `2` (with a guard `if depth < 2: raise`), so the per-layer
initialization std is `(init_scale ** (1/depth)) * (n ** -0.5)` — the $1/N$ power now spreading the small
overall scale across *three* multiplied factors. And `lr=1e-3` instead of the depth-2 rung's `5e-3` — a
smaller step, which is the right move both because three layers compose into a more sensitive product and
because the smaller step tracks the gradient flow more faithfully, and the bias *is* the flow. There is
no nuclear-norm penalty, no rank cap, no extra regularizer; the entire difference from the floor is one
more layer and a smaller learning rate. This is the default the scaffold ships precisely because it is
the strongest of the three.

I want to be careful about rank10-200 specifically, because a sharper bias cannot manufacture information
that is not in the samples. That environment observes $0.10\cdot200^2 = 4000$ entries of a rank-10 matrix
with $10(400 - 10) = 3900$ degrees of freedom — an oversampling ratio of $1.03$, a razor above the
parameter count — and the completion sample complexity scales as $nr\log n \approx 10\cdot200\cdot\ln200
\approx 10600$, nearly three times what I am given. Both lower rungs sat at $\sim 0.92$ there: depth-2's
weak bias could not identify the subspace, and SVT's convex program could not either (and its 11-iteration
truncation only made it worse). The saturating depth-3 bias is *sharper* at separating modes once the data
pins down which modes are real, but if $4000$ entries are simply too few to pin down a 10-dimensional
row-and-column space, no amount of spectral sharpening rescues it — a sharper cap on modes I cannot even
locate does nothing. So I expect rank10-200 to stay near $0.9$, perhaps a hair better than the other two,
and I will read that not as a failure of the method but as the information floor asserting itself where the
sampling is below threshold. The clean tests of the *bias* are the two well-sampled environments, where the
data is not the excuse.

The falsifiable expectations against the two rungs below. On rank3-50, depth-2 got `0.452` and SVT
`0.250`; if the saturating depth-3 bias genuinely beats nuclear-norm strength, it should crush both —
I expect near-exact recovery, an order of magnitude better than the nuclear-norm baseline — down into the
low single-digit percents — because $30\%$ of a rank-3 matrix *is* recoverable and only a strong-enough
low-rank bias was missing. On rank5-100, the
canonical setup, depth-2 got `0.596` and SVT `0.311`; depth-3 should again win decisively, landing well
below `0.1` — this is the environment the whole depth phenomenon was characterized on. On rank10-200 I am
more cautious: depth-2 and SVT both sat at `~0.92`–`0.93`, and $10\%$ of a rank-10 $200\times200$ matrix
is right at the information floor. The saturating bias is sharper, but if the entries are too few to
identify the subspace at all, even depth-3 may not escape — so I expect it to stay high there, perhaps
marginally better than the others but still around `0.9`. The clean tests are rank3-50 and rank5-100: if
depth-3 drops them by an order of magnitude below the nuclear-norm baseline, it confirms that the
implicit bias of a deep factorization is *stronger than* any norm-based surrogate — the thing neither of
the first two rungs could reach.
