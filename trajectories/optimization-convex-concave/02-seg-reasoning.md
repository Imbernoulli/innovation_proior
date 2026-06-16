The R-SEG numbers confirm the diagnosis exactly, and they tell me the cure is to *remove* something,
not add it. The mean `final_gradient_norm` landed at `0.751257`, and the split is the whole story:
`bilinear_fgn = 1.409909`, `delta_nu_fgn = 0.092606`. The bilinear half is pinned at almost precisely
the irreducible-bias floor I predicted, `λ‖z0−z*‖ ≈ 0.1 × √200 ≈ 1.41` — the anchored extragradient
converged fine, but it converged to the *regularized* solution `w*`, which on this rotation field
sits a distance `‖F(w*)‖/λ` from the origin, and since `‖F(z)‖=‖z‖` here that residual is exactly the
`1.41` on the board. The fixed pull toward `z0=[10,10]ᵀ` did its job too well: it dragged the iterate
back toward the worst possible point and held it there. Meanwhile `delta_nu` came out small,
`0.0926`, because there `z0 ~ N(0,I)` starts near the solution and `λ=0.01` is tiny, so the bias term
is negligible and the mild regularization only stabilizes. The mean is dominated by the bilinear
half, and that half is bias, not optimization error. More iterations would not have helped; the floor
is structural. The high-noise column (`0.774997`, bilinear `1.421081`) barely moves from default,
which is itself a tell — the score is set by the deterministic anchor bias, not by the noise, so
turning the noise up changes almost nothing. That is the signature of a method limited by bias rather
than by variance.

So the lesson is sharp: the anchor at `z0` was supposed to buy strong monotonicity and noise
robustness, but its bias `λ‖z0−z*‖` swamped any benefit because the start is far from the solution on
the very instance that dominates the mean. The obvious correction is to throw the anchor away and run
the bare extragradient step — the scaffold default — and see how far the look-ahead contraction goes
when nothing is dragging it toward `[10,10]`. Let me re-derive that step from scratch so I know
precisely what it does and does not promise, because I am now trusting it to do the bilinear work that
the anchor sabotaged.

Strip the problem to the bone, `f(x,y)=xy`, the instance that produced the `1.41`. The joint field is
`F(z) = [∂f/∂x, −∂f/∂y] = [y, −x] = Jz` with `J=[[0,1],[−1,0]]`, the skew-symmetric rotation
generator. Every evaluation of `F` points *around* the origin, never toward it. The plain forward
step `z_{t+1}=z_t−τF(z_t)` is the operator `M=I−τJ`, eigenvalues `1∓iτ`, modulus `√(1+τ²)>1` for any
`τ>0` — geometric divergence, an outward spiral, and shrinking `τ` only slows the blow-up without
ever crossing below `1`. The structural reason is that `J` is skew, so its eigenvalues are pure
imaginary: the field is monotone (`⟨Jz,z⟩=zᵀJz=0` since `Jᵀ=−J`) but with a *zero* strong-
monotonicity margin. There is no contractive component anywhere for a single forward evaluation to
grab. This is exactly why a plain gradient step cannot touch the bilinear half — and why R-SEG's
contraction came entirely from the artificial `λ` it injected, at the cost of the bias.

The fix that needs no artificial `λ` is to stop trusting `F` at where I am. The implicit/backward
step `z_{t+1}=z_t−τF(z_{t+1})=(I+τF)^{-1}(z_t)` is the resolvent; on the bilinear field
`(I+τJ)^{-1}` has eigenvalues `1/(1∓iτ)`, modulus `1/√(1+τ²)<1` for *every* `τ>0` — it spirals
inward unconditionally, no step-size restriction, no `λ` needed. That is the ideal. But it is
implicit: `z_{t+1}` appears inside `F` on both sides, a nonlinear solve per step, unrunnable on a
generic field. So I imitate it explicitly. Guess the future point cheaply with one forward step,
`w = z_t − τF(z_t)` — the look-ahead — then evaluate the field at `w` and take the actual step from
the *original* `z_t`:

  `z_{t+1} = z_t − τF(w)`.

Anchor at `z_t`, aim with `F(w)`. Two operator evaluations: a predictor at `z_t` and a corrector at
`w`, both fully explicit. I have to check I have not smuggled in two forward steps, which would still
diverge. Grind it out on the rotation. `F(z_t)=Jz_t`, so `w=(I−τJ)z_t`; then
`F(w)=Jw=(J−τJ²)z_t=(J+τI)z_t` since `J²=−I`. Therefore

  `z_{t+1} = z_t − τ(J+τI)z_t = (I−τJ−τ²I)z_t`.

A `−τ²I` term appeared that was not in the forward step. The eigenvalues are now `1−τ²∓iτ`, modulus
`√((1−τ²)²+τ²)=√(1−τ²(1−τ²))<1` for `τ<1`. At `τ=0.1` that is `√(1−0.01·0.99)≈0.99504` — below one,
the spiral turns inward. The extra evaluation manufactured, for free, the contractive `−τ²I` the
forward step lacked, and — crucially — it did so *without any anchor bias*, because the contraction
comes from the operator's own curvature, not from a pull toward an external point. This is the deep
contrast with R-SEG: there the inward force was `λ(z0−z)`, an artificial spring with a built-in bias;
here the inward force is `−τ²I`, the leading term of the resolvent itself, with no bias. On the
bilinear field the iterate now contracts toward the *true* origin, not toward a regularized `w*`, so
the `1.41` floor should simply vanish.

Let me also pin down the deterministic convergence in general, not just on the toy, because I want to
know what guarantee I am actually leaning on for the `(δ,ν)` half. With `z*` such that `F(z*)=0`,
`w=z_t−τF(z_t)`, `z_{t+1}=z_t−τF(w)`, completing the square gives the one-step identity
`‖z_{t+1}−z*‖² = ‖z_t−z*‖² − 2τ⟨F(w),w−z*⟩ + τ²‖F(w)−F(z_t)‖² − ‖w−z_t‖²`. The middle term is
`≤0` by monotonicity (`⟨F(w)−F(z*),w−z*⟩≥0`, `F(z*)=0`) — that is the progress. The last two,
`τ²‖F(w)−F(z_t)‖²−‖w−z_t‖²`, are the discretization error of using `F(w)` instead of the true
implicit point; Lipschitzness bounds them by `(τ²L²−1)‖w−z_t‖²`, strictly negative for `τ<1/L`. So
the distance is Fejér-decreasing, and the step-size ceiling `τ≤1/L` is forced by exactly this
inequality — at `τ=1/L` the error term vanishes, below it there is strict contraction. This is the
guarantee I keep: it needs only monotonicity and Lipschitzness, no `λ`, no strong-monotonicity margin.
What it does *not* give, when `μ=0`, is a *rate* on the last iterate or on the gradient norm — only
the ergodic gap.

Let me confirm the `−τ²I` is the resolvent's curvature, so I trust it beyond the toy. The backward
step is `(I+τJ)^{-1}=(1/(1+τ²))(I−τJ)=(1−τ²+O(τ⁴))(I−τJ)=I−τJ−τ²I+O(τ³)`. The forward step keeps only
`I−τJ`, dropping the inward `−τ²I`; the corrected step keeps it. So extragradient is the `O(τ²)`
explicit approximation of the implicit step, where the forward step is only `O(τ)`. In general, for
`F` `L`-Lipschitz, if `w_imp` is the true implicit point then `‖z_{eg}−w_imp‖ ≤ τL‖w−w_imp‖ ≤
τ²L²‖z_t−w_imp‖`, so the corrector matches the implicit step to `O(τ²)` — an improvement exactly when
`τL<1`, which is why the method wants small steps. That is the entire content of the scaffold default,
and it needs no `λ`.

Now the part R-SEG was trying to buy and this rung deliberately gives up: contraction and noise
robustness. Without the `λ` penalty the operator is merely monotone, so the one-step identity
`‖z_{t+1}−z*‖² = ‖z_t−z*‖² − 2τ⟨F(w),w−z*⟩ + τ²‖F(w)−F(z_t)‖² − ‖w−z_t‖²` has the monotonicity term
`−2τ⟨F(w),w−z*⟩ ≤ 0` (progress) and the discretization error `(τ²L²−1)‖w−z_t‖² ≤ 0` for `τ≤1/L`, so
the distance is Fejér-decreasing — but with `μ=0` there is no geometric rate, only the `O(1/k)`
ergodic gap, and under noise no last-iterate gradient-norm contraction. The noise enters as a
neighborhood: with additive update perturbations the iterate converges to an `O(τσ²)` ball rather than
to `z*` exactly. So I am trading R-SEG's artificial contraction-plus-bias for honest no-bias-but-
slow. On `bilinear` that is a clear win, because the bias was the problem. On `delta_nu` it is a
question mark: R-SEG's small `λ=0.01` there gave a stable `0.0926`, and dropping `λ` removes that
stabilization, so I should watch whether `delta_nu` gets worse even as `bilinear` gets dramatically
better.

For the noise, one design rule matters: both evaluations within a step must use the *same* operator.
The predictor-corrector logic is that `F(w)` stands in for `F(z_{t+1})` of one operator; if the two
half-steps used independently sampled operators, predictor and corrector would concern different
operators and the `O(τ²)` approximation would collapse, reintroducing divergence on stochastic
bilinear. Here the harness's stochasticity is additive *update* noise around a single deterministic
`F`, not independently sampled operators, so this is automatic — `oracle.grad` is the same
deterministic field at both evaluations, and the noise is injected by `oracle.noise()` after each
half-step. The step uses `τ=0.1` on bilinear (`L=1`, so `τ<1/L` with margin, matching the contraction
modulus that wants `τ` well below 1) and `τ=1.0` on `(δ,ν)` (the clipped-monotone field has slope at
most about one). Two operator evaluations, two noise draws, no anchor state at all — this is the
literal scaffold default; the full module is in the answer.

So the falsifiable expectations against the R-SEG numbers. The bilinear half should collapse: with no
anchor pulling toward `[10,10]`, the look-ahead contraction modulus `0.995` per step over 900
iterations drives the iterate toward the true origin, so `bilinear_fgn` should fall from `1.41` to a
small value limited only by the additive noise — I expect it well under `0.2`, an order of magnitude
better, and that alone should slash the mean. The `delta_nu` half is the risk: removing the
stabilizing `λ` could let it drift up from `0.0926` toward `~0.2`, because the clipped field is flat
and merely monotone with no margin, so the bare extragradient has no contraction there and lands
wherever the `O(1/k)` ergodic behavior and the noise leave it. So I predict the mean drops sharply
from `0.751` — driven entirely by the bilinear collapse — to roughly the `0.15–0.20` range, with the
two halves now *closer in magnitude* (bilinear no longer dominating, possibly `delta_nu` now the
larger of the two). If instead `delta_nu` blows up past `bilinear`, that tells me the next rung must
restore a *non-biased* form of anchoring — something that contracts the merely-monotone field without
the fixed-`z0` bias that sank this rung's predecessor.
