The SEG numbers vindicate dropping the anchor and tell me precisely what the next move has to fix. The
mean `final_gradient_norm` fell from R-SEG's `0.751257` to `0.182141` — a four-fold improvement —
and the split shows exactly why: `bilinear_fgn` collapsed from `1.409909` to `0.173788`, an order of
magnitude, because with no pull toward `[10,10]ᵀ` the look-ahead contraction modulus `≈0.995` drove
the iterate toward the *true* origin over the 900 iterations. The bias floor I predicted simply
vanished, exactly as the `−τ²I` contraction story said it would. But the other half of my prediction
also came true, and it is now the binding constraint: `delta_nu_fgn` *rose* from `0.092606` to
`0.190493`. Removing R-SEG's stabilizing `λ=0.01` cost me on the `(δ,ν)` instance, where the clipped
field is flat and merely monotone — no strong-monotonicity margin, so bare extragradient has no
contraction there and lands wherever the `O(1/k)` ergodic behavior and the noise leave it. The two
halves have crossed over: `delta_nu` (`0.190`) is now slightly the *larger* of the two, and the mean
is no longer dominated by one instance but split evenly between them. The high-noise column makes the
fragility blunt — `delta_nu_fgn` blows up to `0.936626` and drags the high-noise mean to `0.582105`
— confirming that the bare merely-monotone step on `(δ,ν)` is variance-limited, with nothing to
contract the noise it accumulates.

So the situation is symmetric to where I started, but inverted. R-SEG's anchor contracted everything
but biased the bilinear half into the floor. SEG dropped the anchor, fixed bilinear, but lost the
contraction on `delta_nu` and is now noise-limited there. What I want is the thing that sat between
these two failures: a form of anchoring that *contracts* the merely-monotone field — including under
noise — without paying R-SEG's fixed-`z0` bias. The defect in R-SEG was not anchoring per se; it was
that the anchor weight stayed *constant* forever, so the pull toward `z0` never died and the bias
`λ‖z0−z*‖` was permanent. What if the anchor pull *decays* as the iteration proceeds — strong enough
early to contract and kill the rotation, but vanishing so that I am not dragged toward `z0` at the
end? Then late iterates would not be biased toward the anchor at all, and the bilinear collapse SEG
just demonstrated would survive, while the early contraction would tame `delta_nu`.

Let me derive the right decaying schedule rather than guess it, because the rate of decay is the whole
design. This is the anchoring idea — pull each iterate back toward a fixed reference `z0`, the
Halpern device — but planted inside the extragradient step I now trust, with a time-decaying weight.
The continuous picture is the cleanest place to find the schedule. The anchored flow is
`ż(t) = −F(z(t)) − β(t)(z(t) − z0)`: the operator drives toward a zero of `F`, a decaying spring
pulls back toward the start. There are two competing speeds in `β(t)`. The contracting speed: the
spring alone, `ż=−β(z−z0)`, contracts the iterate and is what kills the rotation and stabilizes the
flat `(δ,ν)` field. The vanishing speed: but I must not converge to `z0`, I want a zero of `F`, so the
spring must eventually die — this is precisely what R-SEG's constant weight failed to do. Parametrize
`β(t)=γ/t^p`. With `p>1` the spring dies too early, the flow is barely contracted, slow; with `p<1` it
dies too late, keeps dragging toward `z0`, slow (and biased — this is R-SEG's disease in the limit
`p→0`). The sweet spot is `p=1`, `β(t)=1/t`, where contracting and vanishing speeds are matched.

Check `β(t)=1/t` actually accelerates on the rotation, because that is where R-SEG bit the dust and
where I need the bias gone. For `f=xy` the anchored flow is `ẋ=−y+(1/t)(x0−x)`, `ẏ=x+(1/t)(y0−y)`.
Multiply by `t`: `d/dt(tx)=−ty+x0`, `d/dt(ty)=tx+y0`. Differentiate again to get forced harmonic
oscillators in `tx` and `ty`, whose solution gives `x(t)=(y0 cos t + x0 sin t − y0)/t` and
`y(t)=(y0 sin t − x0 cos t + x0)/t`. The iterate decays like `1/t`, so `‖z(t)‖²∼1/t²` — the squared
gradient norm goes like `1/t²`. The anchor with a `1/t` weight converts the rotation's neutral
circling into a clean polynomial `1/t²` decay, *and* the `z0` dependence sits inside bounded
oscillating numerators divided by `t`, so its influence vanishes — no permanent bias toward `[10,10]`.
That is the property I need: the contraction that R-SEG had, but with the bias decaying away rather
than fixed. The discrete shadow of `β(t)=1/t` is an anchoring coefficient that decays like `1/k`.

Now the concrete discrete step. Plant the decaying anchor inside both half-steps of extragradient,
with the offset relative to the *current* point `z` (not the look-ahead `w`) in both lines, and the
gradient step size kept constant:

  predictor: `w = z − τF(z) + c_k(z0 − z) + noise`,
  corrector: `z_next = z − τF(w) + c_k(z0 − z) + noise`,

where `c_k` is the decaying anchor coefficient. When `c_k=0` this is exactly the SEG of the previous
rung; the decaying anchor is the only new thing. The discrete schedule that realizes `β(t)=1/t`: I
use `c_k = 1/(k+3)` with `k` the zero-based step index. The `+3` offset keeps the very first
coefficient sensible (`c_0=1/3 < 1`, a contraction, not an overshoot) and avoids the singularity a
bare `1/k` would have at `k=0`; the tail still decays as `1/k`, so the `1/t²` acceleration is intact.
Note this is a *pure* convex combination toward `z0` — the offset is `c_k(z0−z)` with no separate
`τλ` factor, unlike R-SEG's `τλ(z0−z)`. That matters: R-SEG's pull was `O(τλ)` and constant; here the
pull is `O(1/k)` and decaying, so it can be order-one early (when `c_0=1/3` it moves a third of the
way toward `z0`, a real contraction) yet die to nothing by iteration 900 or 6000. The strength is in
the schedule, not in a fixed `λ`.

Let me sanity-check the schedule produces the right structural growth, because that is the mechanism
behind the `1/k²`. The anchoring extragradient with `c_k=1/(k+3)` admits a Lyapunov function
`V_k = A_k‖F(z_k)‖² + B_k⟨F(z_k), z_k−z0⟩`, and the recurrence the schedule forces is
`B_{k+1}=B_k/(1−c_k)`, which telescopes to `B_k` growing *linearly* in `k`, and `A_k ∝ c_k^{-1}B_k`
growing *quadratically*. A quadratically growing weight on `‖F(z_k)‖²` is exactly what makes `V_k`
bounded force `‖F(z_k)‖²=O(1/k²)` on the *last* iterate — no averaging, no best-so-far tracking. The
look-ahead gradient earns a sum-of-squares in the Lyapunov decrease (this is what extragradient buys
that a plain anchored gradient step cannot, and why I plant the anchor *inside* extragradient rather
than alongside it), and monotonicity plus Lipschitzness, weighted by exactly these coefficients, make
`V_{k+1}≤V_k`. The upshot: deterministically this is the optimal `O(L²‖z0−z*‖²/k²)` last-iterate
gradient-norm rate, which is faster than SEG's `O(1/k)` and carries no fixed bias.

But I have to be honest about what noise does to this acceleration, because the `delta_nu` high-noise
blow-up I just saw is the warning. The `1/k²` rests on `V_k` decreasing every step, and that decrease
was driven by monotonicity and Lipschitz identities for the *exact* operator. With additive update
noise, each step injects an error into those identities, and because the `‖F‖²` term is weighted by
`A_k∼k²`, the accumulated noise is *amplified* by the same quadratic factor that produced the
acceleration — exactly like stochastic Nesterov in convex minimization. So I should expect this rung
to show a fast transient followed by a noise-dominated floor: the gradient norm drops quickly while
the deterministic dynamics dominate, then flattens once the `k²`-amplified noise catches up. The
decaying anchor still helps `delta_nu` versus bare SEG — it gives the early contraction SEG lacked —
but it is not a variance-control mechanism, so the floor it reaches is still set by `σ`. On `bilinear`
the noise is tiny (`σ=0.001`) so the `1/t²` decay should mostly win and beat SEG's `0.173788`; on
`delta_nu` (`σ=0.02`) the early contraction should beat SEG's `0.190493`, but the `k²` amplification
over 6000 iterations means I should not expect it to reach the tiny values the deterministic rate
promises.

So the implementation: the state carries `z`, the fixed anchor `z0`, and the step index (the
coefficient needs `k`). Each step does two operator evaluations — predictor at `z`, corrector at the
look-ahead `w` — two noise draws, and the same pure decaying-anchor offset `c_k(z0−z)` in both lines,
with `c_k=1/(step_index+3)`. The step size is constant (`τ=0.1` bilinear, `τ=1.0` `(δ,ν)`); the
anchor decays. Two operator evaluations, two noise draws; the full module is in the answer.

Falsifiable expectations against the SEG numbers. Both halves should improve relative to SEG, because
the decaying anchor adds the early contraction SEG lacked without adding R-SEG's permanent bias.
`bilinear_fgn` should fall from `0.173788` — the `1/t²` decay on the near-noiseless rotation should
push it down, maybe to `~0.16` or below, though the constant-step decaying-anchor will not reach the
near-zero of a perfectly tuned schedule. `delta_nu_fgn` should fall most, from `0.190493`, because the
early contraction directly attacks the merely-monotone flatness that left SEG stranded there — I
expect it down to roughly `0.10–0.12`, with the noise floor preventing anything dramatically smaller.
So the mean should drop from `0.182141` to roughly the `0.13–0.14` range, with `delta_nu` no longer
the larger half. The cleanest signature to watch is the AUC: the `1/t²` transient should make
`auc_log_iteration_log_grad` markedly more negative than SEG's `−0.346938` (a steeper log-log
descent), even if the final value is only moderately better — that gap between a strongly negative
AUC and a not-dramatically-smaller final norm is exactly the fingerprint of "fast transient, then
noise floor." If the high-noise `delta_nu` still blows up the way SEG's did, that tells me the next
rung must add genuine variance control — re-anchoring with growing strength toward the *moving*
trajectory rather than a single fixed `z0`, so the contraction keeps tightening instead of stalling
at a σ-set floor.
