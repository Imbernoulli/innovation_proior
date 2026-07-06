The SEG numbers vindicate dropping the anchor and tell me precisely what the next move has to fix. The
mean `final_gradient_norm` fell from R-SEG's `0.751257` to `0.182141` — a `4.12×` improvement — and
`(0.173788 + 0.190493)/2 = 0.182140` confirms the mean is again the flat average, so the split is
where the mechanism lives. `bilinear_fgn` collapsed from `1.409909` to `0.173788`, a factor of `8.1`,
because with no pull toward `[10,10]ᵀ` the look-ahead contraction drove the iterate toward the *true*
origin over the 900 iterations. And the collapse landed almost exactly where I predicted: the
deterministic modulus `0.99504` over 900 steps gives `0.99504^{900}·√200 ≈ 0.161`, and the measured
`0.173788` sits `0.013` above that — the `σ=0.001` noise floor riding on the converged contraction,
about `8%`. The low-noise column is the clincher: `bilinear_fgn` there is `0.162024`, within `0.001`
of the pure deterministic `0.161`, so as the noise is turned down the bilinear half converges straight
onto the value the `−τ²I` contraction predicts. The bias floor is gone, exactly as the curvature story
said it would be.

But the other half of my prediction also came true, and it is now the binding constraint:
`delta_nu_fgn` *rose* from `0.092606` to `0.190493`, a factor of `2.06`. Removing R-SEG's stabilizing
`λ=0.01` cost me on the `(δ,ν)` instance, where the clipped field is flat and merely monotone — no
strong-monotonicity margin, so bare extragradient has no last-iterate contraction there and settles
wherever the `O(1/k)` ergodic behavior and the noise leave it. The two halves have crossed over:
`delta_nu` (`0.190`) is now slightly the *larger* of the two, and the mean is no longer dominated by
one instance but split evenly. The noise columns make the fragility blunt and quantify it. Across
low/default/high, `delta_nu_fgn` runs `0.071846 / 0.190493 / 0.936626` — a `13×` span, with
`0.936626/0.190493 = 4.9` from default to high and `0.190493/0.071846 = 2.65` from low to default, so
most of the default-noise `delta_nu` value is variance, not optimization error. Meanwhile
`bilinear_fgn` runs `0.162024 / 0.173788 / 0.227585`, only a `1.4×` span and `1.31×` from default to
high — near noise-immune, because `σ=0.001` keeps its `O(τσ²)` ball at `~3·10^{-4}` in every regime.
So the picture is clean: bilinear is deterministic-contraction-limited and solved; `delta_nu` is
variance-limited, with nothing to contract the noise it accumulates. The AUC agrees — SEG's
`−0.346938` is *shallower* than R-SEG's `−0.669446`, a less-steep log-log descent, which fits a method
that converges but only at the merely-monotone `O(1/k)` ergodic rate, with no acceleration. And the
aggregate flips the diagnosis cleanly relative to R-SEG: SEG's mean runs `0.116935 / 0.182141 /
0.582105` across low/default/high, a `5×` span, where R-SEG's mean spanned only `3%`. R-SEG was
bias-limited (score deaf to noise); SEG is variance-limited (score dominated by noise), and the entire
swing comes from the `delta_nu` half. That is the exact inversion I need to fix: I have to reinstate
contraction on `delta_nu` without reinstating bias on `bilinear`.

So the situation is symmetric to where I started, but inverted. R-SEG's anchor contracted everything
but biased the bilinear half into the floor. SEG dropped the anchor, fixed bilinear, but lost the
contraction on `delta_nu` and is now noise-limited there. What I want is the thing that sat between
these two failures: a form of anchoring that *contracts* the merely-monotone field — including under
noise — without paying R-SEG's fixed-`z0` bias. Let me lay out the moves honestly before I pick. The
first is to bring back a fixed anchor but with a small `λ`, hoping the bias stays tolerable. But I
already dismantled this at the previous rung: any fixed `λ` toward `z0` incurs a permanent bias
`λ‖z0−z*‖ = 14.14λ` on bilinear, so to keep the bias below the `0.16` I just won I would need
`λ<0.011`. And a `λ` that small barely regularizes: the noise floor it buys on `delta_nu` is
`~ητσ²/λ = 1·0.0004/0.011 ≈ 0.036`, radius `≈0.19` — essentially SEG's noise-limited `delta_nu`
unchanged. So a fixed weak anchor pays a fresh bilinear bias *and* fails to crush the `delta_nu` noise.
Strength and bias are the same stick, and a fixed anchor cannot escape it. The second move is iterate averaging: the
ergodic gap converges at `O(1/k)`, so I could report a running average instead of the last iterate.
But the metric is the *last-iterate* gradient norm, and on the rotation the trajectory circles the
origin, so its average washes out the angular motion without shrinking the radial gradient norm the
way I need — averaging attacks the wrong quantity. The third move is the one that threads the needle:
keep an anchor, but let its weight *decay* over the run. The defect in R-SEG was never anchoring per
se; it was that the anchor weight stayed *constant* forever, so the pull toward `z0` never died and the
bias `λ‖z0−z*‖` was permanent. If the anchor pull is strong early — enough to contract and kill the
rotation and stabilize the flat field — but vanishes as the iteration proceeds, then late iterates are
not dragged toward `z0` at all, the bilinear collapse survives, and the early contraction tames
`delta_nu`.

The rate of decay is the whole design, so I derive the right schedule rather than guess it. This is the
anchoring idea — pull each iterate back toward a fixed reference `z0`, the Halpern device — but planted
inside the extragradient step I now trust, with a time-decaying weight. The continuous picture is the
cleanest place to find the schedule. The anchored flow is `ż(t) = −F(z(t)) − β(t)(z(t) − z0)`: the
operator drives toward a zero of `F`, a decaying spring pulls back toward the start. There are two
competing speeds in `β(t)`. The contracting speed — the spring alone, `ż=−β(z−z0)`, contracts the
iterate and is what kills the rotation and stabilizes the flat `(δ,ν)` field. The vanishing speed — I
must not converge to `z0`, I want a zero of `F`, so the spring must eventually die, which is precisely
what R-SEG's constant weight failed to do. Parametrize `β(t)=γ/t^p`. With `p>1` the tail
`Σ β` converges, the spring's cumulative action is finite, it dies too early and barely contracts —
slow. With `p<1` the spring dies too late, keeps dragging toward `z0`, slow and biased — R-SEG's
disease in the limit `p→0`. The knife-edge is `p=1`, `β(t)=1/t`, where the two speeds match: the
harmonic tail `Σ 1/t` diverges, so the spring's cumulative contraction never runs out, yet `β(t)→0`
pointwise, so no permanent bias survives. This borderline is exactly the property I need, and it is
worth seeing it is genuinely the boundary: `Σ 1/t^p` converges for `p>1` and diverges for `p≤1`, so
`p=1` is the largest `p` at which the anchor's total pull is still unbounded.

Let me verify `β(t)=1/t` actually accelerates on the rotation, because that is where R-SEG bit the dust
and where I need the bias gone. For `f=xy` the anchored flow is `ẋ=−y+(1/t)(x0−x)`, `ẏ=x+(1/t)(y0−y)`.
Multiply through by `t` and set `u=tx`, `v=ty`; then `d/dt(tx)=−ty+x0` reads `u̇=−v+x0`, and
`d/dt(ty)=tx+y0` reads `v̇=u+y0`. This is a linear system `(u̇,v̇)=(−v+x0, u+y0)` whose equilibrium is
`(u*,v*)=(−y0,x0)` and whose homogeneous part `(u̇,v̇)=(−v,u)` is a pure unit-frequency rotation. So
`(u,v)` circles the point `(−y0,x0)` at radius equal to its initial distance from that center; the
initial condition is `u(0)=v(0)=0` (since `u=tx→0` as `t→0`), so the radius is
`√(y0²+x0²)=‖z0‖`. Therefore `u(t)` and `v(t)` are bounded oscillations of amplitude `‖z0‖` about a
fixed center, and `x=u/t`, `y=v/t` both decay like `1/t`. The initial condition survives the division:
near `t=0`, `u≈u̇(0)t=(−v(0)+x0)t=x0 t` since `v(0)=0`, so `x=u/t→x0` — the flow does start at `z0` as
it must, and only *later* does the `1/t` envelope take over. So there is no contradiction between
"decays like `1/t`" and "starts at the far `z0`"; the trajectory begins at `z0` and the anchor's own
`1/t` weighting is what peels it off toward the origin. Hence `‖z(t)‖² = (u²+v²)/t² = O(1/t²)` — the
squared gradient norm goes like `1/t²`. Two things fall out that I need: the rotation's neutral
circling is converted into a clean polynomial `1/t²` decay, *and* the `z0` dependence lives entirely
inside the bounded oscillating numerators `u,v` divided by `t`, so its influence vanishes — no
permanent bias toward `[10,10]`. That is the contraction R-SEG had, with the bias decaying away rather
than fixed. The discrete shadow of `β(t)=1/t` is an anchoring coefficient that decays like `1/k`.

Now the concrete discrete step. Plant the decaying anchor inside both half-steps of extragradient,
with the offset relative to the *current* point `z` (not the look-ahead `w`) in both lines, and the
gradient step size kept constant:

  predictor: `w = z − τF(z) + c_k(z0 − z) + noise`,
  corrector: `z_next = z − τF(w) + c_k(z0 − z) + noise`,

where `c_k` is the decaying anchor coefficient. When `c_k=0` this is exactly the SEG of the previous
rung; the decaying anchor is the only new thing. The discrete schedule that realizes `β(t)=1/t`: I
use `c_k = 1/(k+3)` with `k` the zero-based step index. The `+3` offset is not cosmetic, and I can see
what each part of it does. A bare `1/k` would be singular at `k=0`; and `1/k` at `k=1` equals `1`,
which would move the iterate *all* the way to `z0`, discarding the gradient progress entirely — an
overshoot to the anchor, not a contraction. With `+3`, the first coefficient is `c_0=1/3<1`, a genuine
one-third-of-the-way pull toward `z0`, a real contraction and not an overshoot, and there is no
singularity. The tail is unchanged: for large `k`, `1/(k+3)≈1/k`, so the asymptotic `1/t` schedule and
its `1/t²` acceleration survive; the offset only perturbs the first few steps. I can even size the
total anchoring budget over a run: `Σ_{k=0}^{n-1} 1/(k+3) ≈ ln((n+3)/3)`, which is `5.88` over the
900 bilinear steps and `7.78` over the 6000 `(δ,ν)` steps — the pull is cumulatively unbounded (it
keeps contracting, as the harmonic divergence demands) but each individual late pull is tiny, so
nothing is dragged toward `z0` at the end. One discrete check I want before trusting the schedule: what does the very first step do? At `k=0` the
iterate `z` is still `z0` (the anchor), so the offset `c_0(z0−z)=c_0·0=0` vanishes and the step is a
pure extragradient step — the anchor cannot contribute until the trajectory has left `z0`, which is
exactly right, and it matches the code, where the first step reduces to SEG. On the bilinear field that
first step is `z0=[10,10] → w=[9,11] → z_1=[8.9,10.9]`, `‖z_1‖=√198.02=14.072`, already contracted
from `14.142` by the `−τ²I` curvature alone; the decaying anchor only switches on from `k=1`, when
`z0−z` is finally nonzero, and adds rotation-damping on top. Contrast the alternatives to `+3`: a bare
`1/k` is singular at `k=0`, and `1/(k+1)` gives `c_1=1/2` and, worse, would have given `c_0=1` — a
coefficient of one means the offset `z0−z` moves the iterate *all the way* onto `z0`, throwing away the
gradient step; the `+3` caps the opening pull at `c_0=1/3`, safely a contraction. Note also this is a
*pure* convex combination toward `z0` — the offset is `c_k(z0−z)` with no separate `τλ` factor, unlike
R-SEG's `τλ(z0−z)`. That matters:
R-SEG's pull was `O(τλ)` and constant; here the pull is `O(1/k)` and decaying, so it can be
order-one early yet die to nothing by iteration 900 or 6000. The strength is in the schedule, not in a
fixed `λ`.

Let me check the schedule produces the right structural growth, because that is the mechanism behind
the `1/k²`. The anchoring extragradient with `c_k=1/(k+3)` admits a Lyapunov function
`V_k = A_k‖F(z_k)‖² + B_k⟨F(z_k), z_k−z0⟩`, and the recurrence the schedule forces is
`B_{k+1}=B_k/(1−c_k)`, which telescopes — `B_k = B_0 Π_{j<k}(1−c_j)^{-1}` and with `c_j=1/(j+3)` the
product `Π(1−1/(j+3))^{-1}=Π((j+3)/(j+2))` telescopes to `(k+2)/2`, so `B_k` grows *linearly* in `k`
— and `A_k ∝ c_k^{-1}B_k ∝ (k+3)(k+2)` grows *quadratically*. A quadratically growing weight on
`‖F(z_k)‖²` is exactly what makes `V_k` bounded force `‖F(z_k)‖²=O(1/k²)` on the *last* iterate — no
averaging, no best-so-far tracking. The look-ahead gradient earns a sum-of-squares in the Lyapunov
decrease, and this is the specific reason I plant the anchor *inside* extragradient rather than running
a plain anchored gradient step alongside it. A single-evaluation anchored step — Halpern without the
look-ahead — drives the anchored distance down but leaves a `‖F‖²` term in the Lyapunov drift
uncontrolled, so on the rotation it manages only `O(1/k)` on the gradient norm; the corrector supplies
the missing negative term. The two-evaluation structure contributes a `−(positive)·‖F(w_k)−F(z_k)‖²`
into the drift, and since `F(w_k)−F(z_k)` is `O(τ)`-close to `‖F‖`-scale that is exactly the `‖F‖²`
sink the plain step lacks — it is what lets the quadratically-growing `A_k∼k²` stay bounded and hence
forces `‖F‖²=O(1/k²)`. So the look-ahead is not decoration here; it is the difference between slope
`−1` and slope `−2`. Monotonicity plus Lipschitzness, weighted by exactly these coefficients, make
`V_{k+1}≤V_k`. Deterministically this is the optimal
`O(L²‖z0−z*‖²/k²)` last-iterate gradient-norm rate, faster than SEG's `O(1/k)` and carrying no fixed
bias.

But I have to be honest about what noise does to this acceleration, because the `delta_nu` high-noise
blow-up I just saw is the warning. The `1/k²` rests on `V_k` decreasing every step, and that decrease
was driven by monotonicity and Lipschitz identities for the *exact* operator. With additive update
noise, each step injects an error into those identities, and because the `‖F‖²` term is weighted by
`A_k∼k²`, the accumulated noise is *amplified* by the same quadratic factor that produced the
acceleration — exactly like stochastic Nesterov in convex minimization, where the momentum that
accelerates the mean also inflates the variance. So I should expect this rung to show a fast transient
followed by a noise-dominated floor: the gradient norm drops quickly while the deterministic dynamics
dominate, then flattens once the `k²`-amplified noise catches up. The decaying anchor still helps
`delta_nu` versus bare SEG — it gives the early contraction SEG lacked — but it is not a
variance-control mechanism, so the floor it reaches is still set by `σ`. On `bilinear` the noise is
tiny (`σ=0.001`, ball `~3·10^{-4}`) so the `1/t²` decay should mostly win and beat SEG's `0.173788`;
on `delta_nu` (`σ=0.02`) the early contraction should beat SEG's `0.190493`, but the `k²` amplification
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
the larger half. The cleanest signature to watch is the AUC, and I can say why quantitatively. The AUC integrates
`log‖F‖` against `log(iteration)`; a `1/k²` decay is `log‖F‖ ≈ −2 log k + const`, a straight line of
slope `−2` in log-log, whereas SEG's `O(1/k)` ergodic behavior is slope `−1`. Doubling the descent
slope over most of the trajectory drives the integrated area substantially more negative, so the
`1/t²` transient should make `auc_log_iteration_log_grad` markedly more negative than SEG's
`−0.346938`, even if the final value is only moderately better — that gap between a strongly negative
AUC and a not-dramatically-smaller final norm is exactly the fingerprint of "fast transient, then noise
floor": the trajectory plunges at slope `−2` until the `k²`-amplified noise arrests it, then flattens. There is a sharper falsifiable statement inside this, and it is the one that will hand off to the next
rung. Even if SEAG improves the *default*-noise `delta_nu` over SEG, the mark of a method that only
sped up the transient — rather than controlling variance — is that its `delta_nu` noise *span* stays
wide: low-noise `delta_nu` should be far below default, and high-noise `delta_nu` should still blow up
well above it, because the floor the transient stalls at is set by `σ` and the `k²` amplification does
not care how fast I got there. A genuinely variance-controlled method would instead show a *flat*
response across the three noise regimes. So I predict SEAG's `delta_nu` will improve but its noise span
will remain large — SEG's `delta_nu` ran `0.072 / 0.190 / 0.937` across low/default/high, and I expect
SEAG's to shift down but keep a comparable multiplicative spread. If the high-noise `delta_nu` still
blows up that way, that tells me the next rung must add genuine variance control — re-anchoring with
growing strength toward the *moving* trajectory rather than a single fixed `z0`, so the effective
regularization keeps rising and the contraction keeps tightening instead of stalling at a σ-set floor.
