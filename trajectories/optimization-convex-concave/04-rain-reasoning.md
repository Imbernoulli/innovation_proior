The SEAG numbers are the best so far and they point at the exact ceiling I now have to break. The mean
`final_gradient_norm` dropped from SEG's `0.182141` to `0.135449`, and both halves improved as I
predicted: `bilinear_fgn` `0.173788→0.160590` and `delta_nu_fgn` `0.190493→0.110307`. The decaying
anchor added the early contraction SEG lacked without R-SEG's permanent bias — the bilinear bias
floor stayed gone *and* the merely-monotone `(δ,ν)` field finally got contracted. The signature I told
myself to watch confirms the mechanism cleanly: `auc_log_iteration_log_grad` plunged from SEG's
`−0.346938` to `−1.107236`, a far steeper log-log descent, while the *final* norm improved only
moderately. That gap — strongly negative AUC, not-dramatically-smaller final value — is exactly the
fingerprint of "fast `1/k²` transient, then a noise-set floor." And the high-noise column proves the
floor is the limit: `delta_nu_fgn` still blows up to `0.582557` there (versus `0.110307` at default
noise), dragging the high-noise mean to `0.380617`. So the `k²`-amplified noise is precisely what I
warned about — the acceleration's own weighting amplifies the accumulated variance, and the method
stalls at a floor whose size scales with `σ`.

So the binding defect is now sharp and it is *statistical*, not optimization. SEAG reaches its floor
fast but cannot get below it, because its single fixed anchor at `z0` regularizes with a strength that
must stay small (it decays toward zero) and a small regularization is a weak strong-monotonicity, and
weak strong-monotonicity is exactly the property that leaves the noise floor large. Let me make that
precise, because the fix has to attack this coupling directly. The anchor at strength `λ` toward a
point `a` turns `F` into `G(z)=F(z)+λ(z−a)`, `λ`-strongly monotone, with noise floor `~ησ²/λ` — so a
*large* `λ` would crush the noise. But a large fixed `λ` toward `z0` reintroduces R-SEG's bias
`λ‖z0−z*‖`, which is the disease I escaped two rungs ago. The strength `λ` that fights noise and the
bias that a fixed far anchor incurs are coupled through the distance from the anchor to `z*`. That is
the trap SEAG sits in: it can only make `λ` small (by decaying it), so it can never crush the noise.

Here is the thing that nags me, and it is the same non-expansiveness fact I proved back at the R-SEG
rung. The anchored solution `w*` (the zero of `G`) is *closer* to `z*` than `z0` was:
`‖w*−z*‖ ≤ ‖z*−z0‖` and `‖w*−z0‖ ≤ ‖z*−z0‖`, both from `λ‖w*−z*‖² ≤ λ(z*−z0)ᵀ(z*−w*)` and
polarization. So what if I do not anchor once at the far `z0`? What if I solve the anchored problem
approximately, get a point closer to `z*`, *re-anchor there*, and crank `λ` up — because now the
anchor is closer, so the bias `λ·(distance)` can tolerate a bigger `λ`? A bigger `λ` makes the next
subproblem more strongly monotone, hence better conditioned, hence cheaper to solve under noise. A
chain of warm restarts, each with a closer anchor and a larger penalty. That breaks the
strength-versus-bias coupling: the bias stays bounded because each anchor is close, while `λ` climbs
geometrically, so the noise floor shrinks geometrically.

I have to redo the recursion at the operator level, never touching function values, because the
single-player convex version of this trick rests on `min f ≤ f`, an inequality that has no saddle-
point analogue (neither `min-max f ≤ f` nor `≥ f` holds). Define a recursively regularized sequence:
`f^(s+1) = f^(s) + (λ_{s+1}/2)‖x−x_{s+1}‖² − (λ_{s+1}/2)‖y−y_{s+1}‖²`, where `x_{s+1},y_{s+1}` is the
approximate solution returned by round `s` and the strengths grow geometrically,
`λ_{s+1}=(1+γ)λ_s`. In operator form, after `s` anchors the gradient operator is
`F^(s)(z)=F(z)+Σ_{i=1}^s λ_i(z−z_i)` — the penalties accumulate, each a fresh anchor `z_i` with its
own strength. Run `S=⌊log(L/λ)⌋` rounds so the accumulated strength climbs from `λ` up to about `L`:
then `F^(s)` is at least `~λ(1+γ)^s`-strongly monotone (strength growing per round) while staying
`≤2L`-Lipschitz, so its condition number `2L/(λ(1+γ)^s)` *shrinks* toward `O(1)`. The later
subproblems are well-conditioned, so even under noise they are cheap to solve accurately — which is
the whole payoff.

The central question is whether driving each subproblem's residual small makes the final `‖F(z_S)‖`
small. Peeling off the penalties and using strong monotonicity of `F^(S-1)` to convert distances back
into residuals, plus the non-expansiveness `‖z*_j−z*_{j-1}‖ ≤ ‖z*_{j-1}−z_j‖` to chain the exact
solutions, the messy double sum collapses to a clean recursive anchoring lemma:

  `‖F(z_S)‖ ≤ 16λ Σ_{s=1}^S (1+γ)^{s-1} ‖z*_{s-1}−z_s‖`,

the final gradient norm is a *geometrically-weighted sum of per-round subroutine errors*. The weight
`(1+γ)^{s-1}` grows, but so does the strong monotonicity `λ(1+γ)^s` of subproblem `s`, so I can drive
the per-round error down at the same geometric rate and keep the product small — a fair fight. Solving
each subproblem with a two-phase epoch-SEG (a fixed step to kill optimization error, then a shrinking
step to eat the statistical floor) lands the total oracle cost at the additive statistical floor
`Õ(σ²ε^{-2}+κ)`, with no spurious `L²` and no `ε^{-4}` — the recursion buys back the two factors of
`ε` that a single far anchor cannot. The merely-monotone case reduces to this via one cold anchor at
`z0` with `λ=min(ε/D,L)`. This is the method the strongest rung implements; the theory is the point of
it, but the algorithm as stated is a triple-nested loop and I want it as one loop.

Collapse it. Set the inner counts to one — `N_s=1`, `K_s=0`, a single SEG iteration per round — so
"round `s`" and "SEG step `t`" become the same index, and the accumulated penalty
`Σ_i λ_i(z−z_i)` becomes, at iteration `t`, an anchor pulling toward the stored past iterates with
geometric weights. One extragradient step with that running anchor:

  predictor: `w = z − τF(z) + τλ·Σ_j w_j(z_j − z) + noise`,
  corrector: `z_next = z − τF(w) + τλ·Σ_j w_j(z_j − w) + noise`,

the regularizer being a geometrically-weighted running average of the trajectory already written into
state. Two implementation facts make this runnable in `O(d)`. First, I never need the iterates
individually: `λΣ_j w_j(z_current − z_j) = λ[(Σ_j w_j)z_current − Σ_j w_j z_j]`, so I keep two running
buffers — a scalar `weight_sum = Σ_j w_j` and a vector `weighted_flow_sum = Σ_j w_j z_j` — and the
anchor contribution is `τλ(weighted_flow_sum − weight_sum·z_current)`, an `O(d)` evaluation. Second,
the sign: that expression is a pull from the current point *toward* the weighted average of past
points, which is the moving anchor — re-anchoring toward where the trajectory has been, weighted to
favor the recent past, with `γ` setting how fast the weighting grows. Each new iterate `z_next` is
inserted with weight `current_weight = γ(1+γ)^{step_index+1}` (the stored-iterate convention is
one-based even though the displayed recurrence is zero-based, because `z_next` is written *after* the
step is taken), updating both buffers by adding `current_weight` and `current_weight·z_next`.

Let me reason about why this should finally cross below SEAG's floor, and where its own limits are. The
moving anchor with *growing* strength is the structural difference from SEAG's single decaying anchor.
SEAG's regularization decayed toward zero, so its strong-monotonicity vanished and its noise floor was
fixed; this method's effective regularization toward the recent trajectory *grows* (the geometric
weights `(1+γ)^j` accumulate), so the late iterations are increasingly strongly monotone around where
the trajectory has settled — which is increasingly close to `z*`. The contraction keeps tightening
instead of stalling, so the noise floor keeps shrinking. The bias stays bounded because the anchor
tracks the moving trajectory, not the far fixed `z0`. The harness fixes the constants:
`τ=0.1, λ=0.1, γ=0.001` on bilinear and `τ=1.0, λ=0.01, γ=0.0001` on `(δ,ν)` — `γ` is deliberately
tiny so `(1+γ)^t` cannot overflow over the 900/6000 iterations while the order of the method is
unchanged; the geometric weighting is gentle but its *accumulation* over thousands of steps is what
grows the effective strength. The state carries `z`, `step_index`, and the two buffers; the first step
has empty buffers (`weight_sum=0`), so it is a plain extragradient step, and the anchor switches on as
the trajectory accumulates. Two operator evaluations, two noise draws, `O(d)` buffer updates; the full
module is in the answer.

So the falsifiable bar this rung must clear, stated against the SEAG numbers it has to beat. The whole
point of the growing moving anchor is to eat the statistical floor SEAG could not, so I expect a
large drop in the mean `final_gradient_norm` from SEAG's `0.135449` — not a marginal one, because the
mechanism attacks the binding constraint (the σ-set floor) rather than the transient. I expect both
halves to fall substantially: `bilinear_fgn` from `0.160590` toward the `0.02` range, since the near-
noiseless rotation (`σ=0.001`) lets the growing contraction run almost unimpeded, and `delta_nu_fgn`
from `0.110307` toward the `0.02` range as well, because the growing strength around the settled
trajectory directly crushes the noise that left SEAG stranded at `0.11`. So the mean should land
roughly an order of magnitude below SEAG, in the low-`0.0X` range. The single cleanest test is the
high-noise `delta_nu`: SEAG blew up to `0.582557` there because its floor scaled with `σ`; if the
growing anchor genuinely controls variance, this rung's high-noise `delta_nu` should be *far* smaller
— if it instead still blows up to several tenths, then the moving anchor is not actually crushing the
noise floor and the recursion's promise has not materialized in the collapsed single-loop form. That
high-noise `delta_nu` number, against SEAG's `0.582557`, is the decisive falsifiable comparison.
