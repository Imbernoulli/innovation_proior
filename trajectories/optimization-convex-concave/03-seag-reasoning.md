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
noise — without paying R-SEG's fixed-`z0` bias. Three moves. The first is a fixed anchor with small
`λ`, but I dismantled this already: to keep the bias `14.14λ` below the `0.16` I just won needs
`λ<0.011`, at which the `delta_nu` noise floor `~ητσ²/λ≈0.036` (radius `≈0.19`) is essentially SEG's,
unchanged — a fresh bilinear bias that still fails to crush the noise. The second is iterate
averaging, but the metric is the *last-iterate* norm, and on the rotation the trajectory circles the
origin so its average washes out the angular motion without shrinking the radial norm — it attacks the
wrong quantity. The third move is the one that threads the needle:
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

Does `β(t)=1/t` actually accelerate on the rotation? For `f=xy` the anchored flow is
`ẋ=−y+(1/t)(x0−x)`, `ẏ=x+(1/t)(y0−y)`. Set `u=tx`, `v=ty`; then `u̇=−v+x0`, `v̇=u+y0`, a linear
system whose homogeneous part `(u̇,v̇)=(−v,u)` is a pure unit-frequency rotation about the equilibrium
`(−y0,x0)`. With `u(0)=v(0)=0`, `(u,v)` circles that center at radius `‖z0‖`, so `u,v` are bounded
oscillations of amplitude `‖z0‖` and `x=u/t`, `y=v/t` both decay like `1/t`. Hence `‖z(t)‖²=O(1/t²)`.
Two things fall out: the rotation's neutral circling becomes a clean `1/t²` decay, and the `z0`
dependence lives entirely inside the bounded numerators divided by `t`, so its influence vanishes — no
permanent bias toward `[10,10]`. That is R-SEG's contraction with the bias decaying away rather than
fixed. The discrete shadow of `β(t)=1/t` is an anchoring coefficient that decays like `1/k`.

Now the concrete discrete step. Plant the decaying anchor inside both half-steps of extragradient,
with the offset relative to the *current* point `z` (not the look-ahead `w`) in both lines, and the
gradient step size kept constant:

  predictor: `w = z − τF(z) + c_k(z0 − z) + noise`,
  corrector: `z_next = z − τF(w) + c_k(z0 − z) + noise`,

where `c_k` is the decaying anchor coefficient. When `c_k=0` this is exactly the bare SEG step;
the decaying anchor is the only new thing. The discrete schedule that realizes `β(t)=1/t`: I
use `c_k = 1/(k+3)` with `k` the zero-based step index. The `+3` offset is not cosmetic, and I can see
what each part of it does. A bare `1/k` would be singular at `k=0`; and `1/k` at `k=1` equals `1`,
which would move the iterate *all* the way to `z0`, discarding the gradient progress entirely — an
overshoot to the anchor, not a contraction. With `+3`, the first coefficient is `c_0=1/3<1`, a genuine
one-third-of-the-way pull toward `z0`, a real contraction and not an overshoot, and there is no
singularity. The tail is unchanged: for large `k`, `1/(k+3)≈1/k`, so the asymptotic `1/t` schedule and
its `1/t²` acceleration survive; the offset only perturbs the first few steps. The total anchoring
budget over a run is `Σ_{k=0}^{n-1} 1/(k+3) ≈ ln((n+3)/3)` — `5.88` over the 900 bilinear steps,
`7.78` over the 6000 `(δ,ν)` steps — cumulatively unbounded (it keeps contracting, as the harmonic
divergence demands) but each late pull tiny, so nothing is dragged toward `z0` at the end. At `k=0`
the iterate is still `z0`, so the offset `c_0(z0−z)=0` vanishes and the first step reduces to SEG:
`z0=[10,10] → w=[9,11] → z_1=[8.9,10.9]`, `‖z_1‖=14.072`, contracted from `14.142` by the `−τ²I`
curvature alone; the anchor switches on from `k=1`. Note the offset `c_k(z0−z)` carries no separate
`τλ` factor, unlike R-SEG's `τλ(z0−z)`: there the pull was `O(τλ)` and constant, here it is `O(1/k)`
and decaying — order-one early, nothing by iteration 900 or 6000. The strength is in the schedule.

The mechanism behind the `1/k²` is the schedule's structural growth. The anchoring extragradient with
`c_k=1/(k+3)` admits a Lyapunov function
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
accelerates the mean also inflates the variance. So I should expect this method to show a fast transient
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

Falsifiable expectations against the SEG numbers. Both halves should improve, because the decaying
anchor adds the early contraction SEG lacked without R-SEG's permanent bias. `bilinear_fgn` should
fall from `0.173788` — the `1/t²` decay on the near-noiseless rotation pushing it to `~0.16` or below,
though the constant-step decaying anchor will not reach the near-zero of a perfectly tuned schedule.
`delta_nu_fgn` should fall most from `0.190493`, since the early contraction directly attacks the
merely-monotone flatness, held up only by the noise floor. So the mean should drop below SEG's
`0.182141`, with `delta_nu` no longer the larger half. The AUC is the clean signature: a `1/k²` decay
is a slope-`−2` line in log-log against SEG's `O(1/k)` slope `−1`, so `auc_log_iteration_log_grad`
should go markedly below SEG's `−0.346938` even if the final norm improves only moderately — that gap
is the fingerprint of "fast transient, then noise floor." The sharper handoff test: a method that only
sped up the transient without controlling variance keeps a *wide* `delta_nu` noise span — low-noise
far below default, high-noise still blowing up — because the floor is `σ`-set and the `k²`
amplification does not care how fast I reached it. SEG's `delta_nu` ran `0.072 / 0.190 / 0.937`; I
expect SEAG's to shift down but keep a comparable multiplicative spread. If high-noise `delta_nu`
still blows up, the next step must add genuine variance control — re-anchoring with growing strength
toward the *moving* trajectory, so the contraction keeps tightening instead of stalling at a `σ`-set floor.
