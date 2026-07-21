The R-SEG numbers confirm the diagnosis exactly, and they tell me the cure is to *remove* something,
not add it. The mean `final_gradient_norm` landed at `0.751257`, and that is precisely
`(1.409909 + 0.092606)/2 = 0.751258` — the harness's mean really is the flat average of the two
per-problem norms, so nothing subtle is hiding in the aggregation; the split is the whole story.
`bilinear_fgn = 1.409909` sits fifteen times higher than `delta_nu_fgn = 0.092606`
(`1.409909/0.092606 = 15.2`), so the bilinear half alone accounts for `1.409909/1.502515 = 93.8%` of
the summed error. Whatever I do next lives or dies on the bilinear number; the `delta_nu` half is a
rounding correction by comparison. And the bilinear number is bias, not optimization error.

The exact anchored-solution norm on `f=xy` is `‖w*‖=10√2·λ/√(λ²+1) = 1.40720` at `λ=0.1` (the `w*`
computation from before). The measured `1.409909` sits `0.0027` above that — `0.19%` — the small excess being the finite-iteration tail plus
the `σ=0.001` noise slack riding on top of a converged anchored solve. The low-noise column makes this
even sharper: `bilinear_fgn` there is `1.407480`, a hair *below* default and only `0.0003` above the
structural `1.40720`, so as the noise is turned down the measured value converges straight onto the
deterministic anchored-solution norm. That is exactly what a bias-limited score looks like. The
high-noise column confirms it from the other direction: `bilinear_fgn` rises only to `1.421081`, a mere
`+0.8%`, so turning the noise *up* barely moves the bilinear half — because its value is set by a
deterministic pull toward `[10,10]ᵀ`, not by variance. Meanwhile `delta_nu_fgn` swings from `0.092606`
to `0.128912` across the same noise change, `+39%`, so the `delta_nu` half carries the variance.
More iterations would not have helped the bilinear half; the `1.4` floor is structural. The aggregate
agrees: the mean runs `0.749993 / 0.751257 / 0.774997` across low/default/high, a `3%` spread over a
wide change in `σ`, and `auc_log_iter_log_grad` sits at a shallow `−0.669446` because the bilinear
trajectory it integrates never descends. A score deaf to the noise level, with a flat descent curve,
is bias-limited — and the only cure for a bias is to remove its source.

So the lesson is sharp, and it dictates the design space. The anchor at `z0` was supposed to buy
strong monotonicity and noise robustness, but its bias `λ‖z0−z*‖` swamped any benefit because the
start is far from the solution on the very instance that dominates the mean. Three moves are on the
table. The first is to keep the anchor and shrink `λ` to cut the bias — but that is the same
bias-versus-conditioning lever I already balanced: dropping `λ` to `≈0.007` for a `0.1` bias makes the
conditioning `L/λ≈140`, the anchored solve fourteen times slower so `‖G‖` never converges in 900
steps, and the noise floor `ησ²/λ` explodes by the same factor. No `λ>0` gives both small bias and
fast contraction. The second move is to slide the anchor closer to `z*`, where a given `λ` buys the same
strong monotonicity at a smaller bias — but I do not know `z*`, and the only special point I have is
the far `z0`, so this is not available yet. The third move is the qualitatively different one: set
`λ=0` *exactly*, drop the anchor, and make the contraction come from somewhere other than an external
spring. This is not the same as `λ→0`, because at `λ→0` I lose the contraction entirely, whereas at
`λ=0` I can lean on a contraction the extragradient step manufactures out of the operator's own
curvature — and crucially that one carries no bias, because it is not a pull toward any external point.
That is the move. Let me re-derive the bare step from scratch so I know precisely what it does and does
not promise, because I am now trusting it to do the bilinear work the anchor sabotaged.

`λ=0` is a switch, not a limit, and that is the whole justification for expecting bilinear to improve
rather than merely stop getting worse. In the earlier ledger everything was proportional to
`λ`: rate `(1−ηλ)`, bias `λ‖z0−z*‖`, floor `ησ²/λ`. Take `λ→0⁺` and it reads useless — no
contraction, no bias, exploding floor. But that ledger is blind to the mechanism that survives at
`λ=0`: the `−τ²I` term extragradient extracts from the operator's *own* curvature, a discretization
artifact of the look-ahead, not a strong-monotonicity margin of `F`. Setting `λ=0` hands the
contraction job from the artificial spring `λ(z0−z)` to the operator's intrinsic curvature, which on
the rotation is real and inward — so I expect bilinear to *fall*, not stall, and I have to verify it
directly on the field.

On `f=xy` the field is `F(z)=Jz`, `J=[[0,1],[−1,0]]`, every evaluation pointing *around* the origin,
never toward it. The plain forward step diverges (modulus `√(1+τ²)>1`, from the opening) because `J`
is skew: monotone (`zᵀJz=0`) but with zero strong-monotonicity margin, no contractive component for a
single forward evaluation to grab. That is why R-SEG's contraction had to come entirely from the
injected `λ`, at the cost of the bias.

The fix that needs no artificial `λ` is to stop trusting `F` at where I am. The implicit/backward
step `z_{t+1}=z_t−τF(z_{t+1})=(I+τF)^{-1}(z_t)` is the resolvent; on the bilinear field
`(I+τJ)^{-1}` has eigenvalues `1/(1∓iτ)`, modulus `1/√(1+τ²)<1` for *every* `τ>0` — it spirals
inward unconditionally, no step-size restriction, no `λ` needed. That is the ideal. But it is
implicit: `z_{t+1}` appears inside `F` on both sides, a nonlinear solve per step, unrunnable on a
generic field. So I imitate it explicitly. Guess the future point cheaply with one forward step,
`w = z_t − τF(z_t)` — the look-ahead — then evaluate the field at `w` and take the actual step from
the *original* `z_t`:

  `z_{t+1} = z_t − τF(w)`.

Anchor at `z_t`, aim with `F(w)`. Two operator evaluations, both fully explicit. On the rotation,
`F(z_t)=Jz_t` so `w=(I−τJ)z_t`; then `F(w)=Jw=(J+τI)z_t` since `J²=−I`, and

  `z_{t+1} = z_t − τ(J+τI)z_t = (I−τJ−τ²I)z_t`.

A `−τ²I` term appeared that was not in the forward step. The eigenvalues are `1−τ²∓iτ`, modulus
`√(1−τ²(1−τ²))<1` for `τ<1`; at `τ=0.1`, `√0.9901≈0.99504`, the spiral turns inward. The extra
evaluation manufactured the contractive `−τ²I` for free, and *without any anchor bias* — the inward
force is the leading term of the resolvent, aimed at the operator's *true* origin, not R-SEG's
artificial spring toward `[10,10]`. So the `1.41` floor should simply vanish.

For the general deterministic convergence — the guarantee I actually lean on for the `(δ,ν)` half —
with `F(z*)=0`, `w=z_t−τF(z_t)`, `z_{t+1}=z_t−τF(w)`, expand
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

The `−τ²I` is the resolvent's curvature: `(I+τJ)^{-1}=(1−τ²+O(τ⁴))(I−τJ)=I−τJ−τ²I+O(τ³)`. The
forward step keeps only `I−τJ`; the corrected step keeps the inward `−τ²I`. So extragradient is the
`O(τ²)` explicit approximation of the unconditionally-stable implicit step, an improvement exactly
when `τL<1` — which is why the method wants small steps, and needs no `λ`.

This step deliberately gives up what R-SEG bought: contraction and noise robustness. With `μ=0` the
Fejér identity holds but there is no geometric rate, only the `O(1/k)` ergodic gap, and under additive
noise the iterate converges to an `O(τσ²)` ball rather than to `z*`. So I am trading R-SEG's
artificial contraction-plus-bias for honest no-bias-but-slow. On `bilinear` that is a clear win, since
the bias *was* the problem. On `delta_nu` it is a question mark: R-SEG's `λ=0.01` gave a stable
`0.0926`, and dropping `λ` removes that stabilization, so I watch whether `delta_nu` worsens even as
`bilinear` improves.

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

The contraction modulus lets me put a number on the collapse. The deterministic map contracts `‖z‖`
by `0.99504` per step, and bilinear is measured post-step at `z_{900}`, so `0.99504^{900}=0.01138`
takes `‖z0‖=√200≈14.14` to `‖z_{900}‖≈0.161`, and since `‖F(z)‖=‖z‖` the predicted `bilinear_fgn` is
about `0.16` — roughly an order of magnitude below R-SEG's `1.41`, with the `σ=0.001` floor far too
small to matter. If it does not land near `0.16`, either my step-count arithmetic or the Fejér story
is wrong.

I keep `τ=0.1` rather than pushing it: a bigger `τ` would shrink the modulus (minimizing
`1−τ²+τ⁴` gives `τ=1/√2`, modulus `0.866`), but the Fejér guarantee is only clean when `τL≪1`, and at
`τ=0.707` I would be running the look-ahead outside that regime on the merely-monotone `(δ,ν)` field
where I have no curvature safety net. `τ=0.1` already reaches `0.16` inside the budget with margin.

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

The `delta_nu` half is the honest risk, and I will not pretend to a tight number there. The `(δ,ν)`
operator is a clipped-monotone component plus a small skew coupling: the skew part is a weak rotation
the `−τ²I` curvature contracts, but the clipped-monotone part is *flat* — zero strong-monotonicity, no
restoring force for the look-ahead to manufacture curvature from. On that flat part the iterate has no
reason to converge; it diffuses under the `σ=0.02` noise along the level set, and the pre-step metric
reads wherever that diffusion wandered. This is the mismatch: extragradient promises only the ergodic
`O(1/k)` gap, the metric scores the last-iterate `‖F‖`, and with `μ=0` nothing ties them. On bilinear
the `−τ²I` gives the last iterate its own rate so the mismatch does not bite; on the flat `delta_nu`
part it bites directly. So removing R-SEG's `λ=0.01` should push `delta_nu_fgn` *up* from `0.0926`,
plausibly into the `0.15–0.20` range, perhaps overtaking the bilinear half.

So the expectations against the R-SEG numbers. The bilinear half should collapse from `1.409909` to
roughly `0.16`, which alone slashes the mean since bilinear was `94%` of the summed error; `delta_nu`
rises from `0.092606` toward `~0.2`; the mean drops from `0.751257` into roughly the `0.15–0.20`
range, the two halves now closer in magnitude and possibly `delta_nu` the larger. The distinguishing
signature is the high-noise column: bilinear should stay near `0.16` there, but the merely-monotone
`delta_nu` has nothing to damp accumulated variance and could blow up. If it runs off past bilinear,
the next step must restore a *non-biased* form of anchoring — one that contracts the merely-monotone
field and controls its noise without the fixed-`z0` bias.
