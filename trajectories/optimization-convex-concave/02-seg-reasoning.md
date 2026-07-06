The R-SEG numbers confirm the diagnosis exactly, and they tell me the cure is to *remove* something,
not add it. The mean `final_gradient_norm` landed at `0.751257`, and that is precisely
`(1.409909 + 0.092606)/2 = 0.751258` — the harness's mean really is the flat average of the two
per-problem norms, so nothing subtle is hiding in the aggregation; the split is the whole story.
`bilinear_fgn = 1.409909` sits fifteen times higher than `delta_nu_fgn = 0.092606`
(`1.409909/0.092606 = 15.2`), so the bilinear half alone accounts for `1.409909/1.502515 = 93.8%` of
the summed error. Whatever I do next lives or dies on the bilinear number; the `delta_nu` half is a
rounding correction by comparison. And the bilinear number is bias, not optimization error.

Let me pin that down rather than assert it, because the exact value is a fingerprint. The anchored
operator on `f=xy` is `G(z)=Jz+λ(z−z0)`, whose zero solves `(J+λI)w*=λz0`; inverting
`J+λI=[[λ,1],[−1,λ]]` gives `w*=(10λ/(λ²+1))[λ−1,λ+1]`, and since `‖F(z)‖=‖z‖` on the rotation the
exact residual there is `‖F(w*)‖=‖w*‖=10√2·λ/√(λ²+1)`, which at `λ=0.1` is `1.40720`. The measured
`1.409909` sits `0.0027` above that — `0.19%` — the small excess being the finite-iteration tail plus
the `σ=0.001` noise slack riding on top of a converged anchored solve. The low-noise column makes this
even sharper: `bilinear_fgn` there is `1.407480`, a hair *below* default and only `0.0003` above the
structural `1.40720`, so as the noise is turned down the measured value converges straight onto the
deterministic anchored-solution norm. That is exactly what a bias-limited score looks like. The
high-noise column confirms it from the other direction: `bilinear_fgn` rises only to `1.421081`, a mere
`+0.8%`, so turning the noise *up* barely moves the bilinear half — because its value is set by a
deterministic pull toward `[10,10]ᵀ`, not by variance. Meanwhile `delta_nu_fgn` swings from `0.092606`
to `0.128912` across the same noise change, `+39%`, so the `delta_nu` half is the one carrying the
variance. More iterations would not have helped the bilinear half; the `1.4` floor is structural, and
the fixed pull toward `z0=[10,10]ᵀ` did its job too well, dragging the iterate back toward the worst
point on the field and holding it there. The aggregate confirms this at a glance: the mean
`final_gradient_norm` runs `0.749993` at low noise, `0.751257` at default, `0.774997` at high — a
`3%` spread across a wide change in `σ`, and the `auc_log_iter_log_grad` sits at a shallow `−0.669446`
because the bilinear trajectory it integrates never descends. A method whose score barely responds to
the noise level and whose descent curve is flat is telling me, unambiguously, that it is limited by a
deterministic bias, and the only cure for a bias is to remove its source.

So the lesson is sharp, and it dictates the design space. The anchor at `z0` was supposed to buy
strong monotonicity and noise robustness, but its bias `λ‖z0−z*‖` swamped any benefit because the
start is far from the solution on the very instance that dominates the mean. Three moves are on the
table. The first is to keep the anchor and shrink `λ` to cut the bias — tempting, because the bias
falls almost linearly, `10√2·λ`, so dropping `λ` from `0.1` to about `0.007` would cut the bilinear
residual by an order of magnitude to `≈0.1`. But that is a point on the same bias-versus-conditioning
lever I already balanced: at `λ=0.007` the conditioning `L/λ≈140` makes the anchored solve fourteen
times slower, so `‖G‖` would not converge inside the fixed 900 iterations and the transfer bound
`2‖G‖+λ‖z0−z*‖` would just be dominated by the unconverged `2‖G‖` term instead of the bias I was
chasing — and the noise floor `ησ²/λ` would explode by the same factor of fourteen. There is no `λ>0`
that gives both a small bias and a fast contraction, because bias and conditioning are the two ends of
one stick. The second move is to slide the anchor closer to `z*`, where a given `λ` buys the same
strong monotonicity at a smaller bias — but I do not know `z*`, and the only special point I have is
the far `z0`, so this is not available yet. The third move is the qualitatively different one: set
`λ=0` *exactly*, drop the anchor, and make the contraction come from somewhere other than an external
spring. This is not the same as `λ→0`, because at `λ→0` I lose the contraction entirely, whereas at
`λ=0` I can lean on a contraction the extragradient step manufactures out of the operator's own
curvature — and crucially that one carries no bias, because it is not a pull toward any external point.
That is the move. Let me re-derive the bare step from scratch so I know precisely what it does and does
not promise, because I am now trusting it to do the bilinear work the anchor sabotaged.

It is worth being explicit about why `λ=0` is a switch and not a limit, because that is the whole
justification for expecting bilinear to improve rather than merely stop getting worse. In the
strong-monotonicity accounting from the previous rung, everything was proportional to `λ`: the
contraction rate was `(1−ηλ)`, the bias was `λ‖z0−z*‖`, and the noise floor was `ησ²/λ`. Take `λ→0⁺`
in that ledger and it says the contraction rate goes to `1` (no contraction), the bias goes to `0`,
and the floor blows up — a method that neither converges nor is biased, which sounds useless. But that
ledger is blind to the one mechanism that survives at `λ=0`: the `−τ²I` term extragradient extracts
from the operator's *own* curvature, which never appears in the `λ`-accounting because it is not a
strong-monotonicity margin of `F` at all — it is a discretization artifact of the look-ahead. So
setting `λ=0` does not land me at the useless limit of the ledger; it hands the contraction job from
the artificial spring `λ(z0−z)` to the operator's intrinsic curvature, and on the rotation that
curvature is real and inward. That is why I expect bilinear to *fall*, not stall, when the anchor is
removed — and it is exactly the thing the strong-monotonicity view cannot see, so I have to verify it
directly on the field.

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
`√((1−τ²)²+τ²)=√(1−τ²(1−τ²))<1` for `τ<1`. At `τ=0.1` that is `√(1−0.01·0.99)=√0.9901≈0.99504` —
below one, the spiral turns inward. The extra evaluation manufactured, for free, the contractive
`−τ²I` the forward step lacked, and — crucially — it did so *without any anchor bias*, because the
contraction comes from the operator's own curvature, not from a pull toward an external point. This is
the deep contrast with R-SEG: there the inward force was `λ(z0−z)`, an artificial spring with a
built-in bias toward `[10,10]`; here the inward force is `−τ²I`, the leading term of the resolvent
itself, aimed at the *origin* of the field. On the bilinear field the iterate now contracts toward the
*true* origin, not toward a regularized `w*`, so the `1.41` floor should simply vanish.

Let me also pin down the deterministic convergence in general, not just on the toy, because I want to
know what guarantee I am actually leaning on for the `(δ,ν)` half. With `z*` such that `F(z*)=0`,
`w=z_t−τF(z_t)`, `z_{t+1}=z_t−τF(w)`, I complete the square rather than quote the result. Expand
`‖z_{t+1}−z*‖² = ‖z_t−z*‖² − 2τ⟨F(w), z_t−z*⟩ + τ²‖F(w)‖²`. The awkward term is the inner product
against `z_t−z*` rather than `w−z*`; rewrite `z_t−z* = (w−z*) + (z_t−w)` and note `z_t−w = τF(z_t)`, so
`−2τ⟨F(w), z_t−z*⟩ = −2τ⟨F(w), w−z*⟩ − 2τ²⟨F(w), F(z_t)⟩`. Now use the polarization
`−2τ²⟨F(w),F(z_t)⟩ = τ²‖F(w)−F(z_t)‖² − τ²‖F(w)‖² − τ²‖F(z_t)‖²`, and recognize `τ²‖F(z_t)‖²=‖w−z_t‖²`.
The two `τ²‖F(w)‖²` cancel, and what is left is the one-step identity
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
robustness. Without the `λ` penalty the operator is merely monotone, so the one-step identity above
has the monotonicity term `−2τ⟨F(w),w−z*⟩ ≤ 0` (progress) and the discretization error
`(τ²L²−1)‖w−z_t‖² ≤ 0` for `τ≤1/L`, so the distance is Fejér-decreasing — but with `μ=0` there is no
geometric rate, only the `O(1/k)` ergodic gap, and under noise no last-iterate gradient-norm
contraction. The noise enters as a neighborhood: with additive update perturbations the iterate
converges to an `O(τσ²)` ball rather than to `z*` exactly. So I am trading R-SEG's artificial
contraction-plus-bias for honest no-bias-but-slow. On `bilinear` that is a clear win, because the bias
*was* the problem. On `delta_nu` it is a question mark: R-SEG's small `λ=0.01` there gave a stable
`0.0926`, and dropping `λ` removes that stabilization, so I should watch whether `delta_nu` gets worse
even as `bilinear` gets dramatically better.

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

I do not want to predict "bilinear collapses" without a number, because the contraction modulus lets
me compute one. The deterministic map is `(I−τJ−τ²I)`, contracting `‖z‖` by `0.99504` per step, and
the metric on bilinear is measured post-step, i.e. at the last iterate `z_{900}`. Over 900 steps the
norm shrinks by `0.99504^{900} = e^{900·ln 0.99504} = e^{−4.477} = 0.01138`, so from `‖z0‖=√200≈14.14`
the deterministic iterate lands at `‖z_{900}‖ ≈ 14.14·0.01138 ≈ 0.161`, and since `‖F(z)‖=‖z‖` the
predicted `bilinear_fgn` is about `0.16`, sitting on top of a `σ=0.001` noise floor that is far too
small to matter. That is a concrete, falsifiable number, and it is roughly an order of magnitude below
R-SEG's `1.41`. A limit check keeps me honest about the step size: at `τ=1/L=1` the modulus is
`√(1−1·0)=1`, no contraction, so I would gain nothing — the contraction is real only strictly inside
`τ<1/L`, and `τ=0.1` sits comfortably inside. If bilinear does not land near `0.16`, either my
step-count arithmetic or the Fejér story is wrong.

A step-size question surfaces here that I want to settle rather than leave implicit: if the modulus is
`√(1−τ²(1−τ²))`, could I not just make it smaller by choosing a bigger `τ`? Minimizing
`1−τ²(1−τ²)=1−τ²+τ⁴` over `τ²` gives `−1+2τ²=0`, so `τ²=1/2`, `τ=1/√2≈0.707`, where the modulus is
`√(1−0.5+0.25)=√0.75≈0.866` — dramatically faster than `0.995`, and `0.866^{900}` is astronomically
small. So a noiseless bilinear solver would want the big step. I keep `τ=0.1` anyway, because the
reason I trust extragradient at all is that it is the `O(τ²)` explicit approximation of the
unconditionally-stable implicit step, and that approximation is only clean when `τL≪1`; at `τ=0.707`,
`τL=0.707` is not small, so I would be running the look-ahead outside the regime where its Fejér
guarantee is comfortable — a bad trade on the merely-monotone `(δ,ν)` field where I have no curvature
safety net. And `τ=0.1` already reaches `0.16` inside the 900-step budget with margin to spare, so
there is nothing to buy by pushing the step and much to risk. The `0.16` is the price of a
conservative, trustworthy step, not a tuning miss.

I can also size the noise floor so I know whether `0.16` is really the deterministic value or whether
variance will overwrite it. The additive perturbations put the iterate in an `O(τσ²)` ball, radius
about `√(τσ²)`; on bilinear that is `√(0.1·0.001²)=√(10^{-7})≈3.2·10^{-4}`, three orders of magnitude
below the `0.16` deterministic value, so bilinear should read essentially its deterministic
contraction and be indifferent to which of the three noise regimes it runs in — the same
noise-immunity R-SEG's bilinear half showed, but now around a converging iterate instead of a frozen
one. That immunity is itself a prediction: the low-, default-, and high-noise `bilinear_fgn` columns
should be nearly identical, clustered near `0.16`. It also tells me something about the AUC. R-SEG's
`auc_log_iter_log_grad` was `−0.669446`, and that shallow value came from a bilinear trajectory that
was flat at `1.4` the whole way — no descent to integrate. A bilinear trajectory that actually spirals
inward at `0.995` per step traces a genuine downhill log-log curve, so I expect SEG's AUC to be more
negative than R-SEG's, reflecting that the score now comes from motion rather than from a pinned floor.

The `delta_nu` half is the honest risk, and I will not pretend to a tight number there because I have
no contraction to lean on. It helps to look at what the field is made of. The `(δ,ν)` operator is a
clipped-monotone component plus a small skew coupling: the skew part is a weak rotation, exactly the
kind of thing the `−τ²I` curvature contracts, so extragradient handles it; but the clipped-monotone
part is *flat* — monotone with slope pushed toward zero inside the clipping region — and flatness is
precisely zero strong-monotonicity, so there is no restoring force there for the look-ahead to
manufacture curvature from. On that flat part the iterate has no reason to converge to a point; it
diffuses under the `σ=0.02` noise along the level set, and the metric, measured pre-step at the last
iterate, reads wherever that diffusion has wandered. This is why the guarantee I actually hold and the
quantity I am graded on come apart on `delta_nu`: extragradient promises only the *ergodic* `O(1/k)`
gap of the averaged iterate, but the harness scores the *last-iterate* gradient norm, and with `μ=0`
there is no theorem tying the last iterate's `‖F‖` to that ergodic gap. On bilinear the mismatch does
not bite because the `−τ²I` gives the last iterate its own linear rate; on the flat `delta_nu` part it
bites directly, and that is where the fragility lives. Removing R-SEG's `λ=0.01` strips the only
strong-monotonicity the flat field had, so I expect `delta_nu_fgn` to *rise*, plausibly from `0.0926`
toward the `0.15–0.20` range, and it would not surprise me if it overtook the bilinear half and became
the larger of the two.

So the falsifiable expectations against the R-SEG numbers. The bilinear half should collapse from
`1.409909` to roughly `0.16` — the `0.99504^{900}` deterministic contraction, order of magnitude
better — which alone should slash the mean, since bilinear was `94%` of R-SEG's summed error. The
`delta_nu` half is the risk and should rise from `0.092606` toward `~0.2`. So I predict the mean drops
sharply from `0.751257`, driven entirely by the bilinear collapse, to roughly the `0.15–0.20` range,
with the two halves now *closer in magnitude* and possibly `delta_nu` the larger. The cleanest
distinguishing signature will be the high-noise column: because bilinear is deterministic-contraction-
limited it should stay near `0.16` there too, while the bare merely-monotone `delta_nu` has nothing to
damp accumulated variance and could blow up badly. If it does — if high-noise `delta_nu` runs off past
`bilinear` — that tells me the next rung must restore a *non-biased* form of anchoring: something that
contracts the merely-monotone field, and controls its noise, without the fixed-`z0` bias that sank
this rung's predecessor.
