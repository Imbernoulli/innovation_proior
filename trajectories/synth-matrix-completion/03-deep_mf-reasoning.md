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

So both rungs so far top out at nuclear-norm strength. Depth-2 *approximates* it (badly, via Adam); SVT
*targets* it (cleanly, via the convex program). Neither beats it, because neither tries to. And the
residual error on the well-sampled environments — $0.25$ on rank3-50, where the truth is recoverable —
says the ceiling itself is the problem: minimum-nuclear-norm and minimum-rank *part ways* exactly in the
data-poor regime that matters, and the convex surrogate leaves recoverable structure on the table. The
natural next move is the one direction I deferred at step 1: do not approximate nuclear norm, and do not
target it — make the implicit bias *stronger than* nuclear norm. The lever is depth. If a depth-2
factorization buys nuclear-norm-strength bias, more layers should sharpen it toward rank, and reclaim the
error nuclear norm cannot.

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

The falsifiable expectations against the two rungs below. On rank3-50, depth-2 got `0.452` and SVT
`0.250`; if the saturating depth-3 bias genuinely beats nuclear-norm strength, it should crush both —
I expect near-exact recovery, an order of magnitude better, something like `0.01`, because $30\%$ of a
rank-3 matrix *is* recoverable and only a strong-enough low-rank bias was missing. On rank5-100, the
canonical setup, depth-2 got `0.596` and SVT `0.311`; depth-3 should again win decisively, landing well
below `0.1` — this is the environment the whole depth phenomenon was characterized on. On rank10-200 I am
more cautious: depth-2 and SVT both sat at `~0.92`–`0.93`, and $10\%$ of a rank-10 $200\times200$ matrix
is right at the information floor. The saturating bias is sharper, but if the entries are too few to
identify the subspace at all, even depth-3 may not escape — so I expect it to stay high there, perhaps
marginally better than the others but still around `0.9`. The clean tests are rank3-50 and rank5-100: if
depth-3 drops them by an order of magnitude below the nuclear-norm baseline, it confirms that the
implicit bias of a deep factorization is *stronger than* any norm-based surrogate — the thing neither of
the first two rungs could reach.
