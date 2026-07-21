McAllester did exactly what I bet it would, and the KL numbers are the proof. Removing the `λ` knob
collapsed and stabilized the KL: on MNIST-FCN the mean fell from Catoni's 539 to 81, and it is now
*tight across seeds*, 71.90 / 80.28 / 91.83 — a high/low ratio of 1.28×, no bimodal blowup, right in the
"under 1.5×" band I hoped for. So the two-attractor system really did collapse to one basin the moment
the knob was gone. MNIST-CNN fell from 192 to 11 (10.0 / 12.1 / 12.1); FashionMNIST-CNN from 286 to 32.
The certificate followed the KL down almost one-for-one, as the separable-inversion argument predicted:
FCN 0.0558 → 0.0373, CNN 0.0250 → 0.0164, Fashion 0.1215 → 0.1002. So the parameter-free additive bound
is the right *stability* fix, and the diagnosis that the runaway KL was a `λ`-mechanism artifact, not a
weak-penalty problem, is confirmed.

But now look at where the additive bound is *stuck*. On MNIST-CNN the empirical 0-1 risk is 0.0119 and
the KL is only 11 nats — the posterior is essentially perfect and barely moved from the prior — yet the
certificate sits at 0.0164, a gap of ~0.0045 above the empirical risk, and it is no longer coming from a
large KL. Decompose it: at emp 0.012, KL 11, the budget is `c = (11+9.54)/30000 = 6.85·10⁻⁴`,
`inv_kl(0.012, 6.85·10⁻⁴) ≈ 0.0165`; drop KL all the way to *zero* and the floor from the confidence
term alone is `inv_kl(0.012, 9.54/30000) ≈ 0.0150`. So of the 0.0045 daylight, killing the entire
residual KL recovers at most 0.0015 — two-thirds of the gap is the irreducible `Λ/n` confidence budget
and the shape of the inversion, not complexity I can still squeeze. The `ce_bound` column says the same
in the training metric: at KL 11, `√((11+9.54)/60000) ≈ 0.0185` is essentially the whole thing, because
the empirical NLL is tiny. With KL this small, `kl_term = (KL+Λ)/(2n)` is around `10⁻³`, and the
additive bound adds `√(10⁻³) ≈ 0.0316` on top of a near-zero empirical risk. The square root is the
floor. McAllester fixed the KL; the residual looseness is now the `√` shape of the relaxation, not the
complexity it penalizes.

So the target is precise: small stable KL, penalized by the square-root shape rather than the complexity
itself. I need a bound whose complexity contribution is *linear* in `kl_term` at small empirical risk,
without reintroducing a free `λ` — the Catoni run showed what a free knob costs. Two of the obvious
routes I have already closed: training directly on the numerical `inv_kl` of the parent is the tightest
object per the parent but non-differentiable (a bisection loop I would have to unroll or
implicit-function-correct), trading the closed, convex, cheaply-differentiable objective I just
stabilized for exactly the optimization fragility I spent a rung removing — I keep `inv_kl` at
certificate time only; and pinning Catoni to its analytic optimum I killed on arithmetic last rung
(`λ*→1` is the point of minimum KL penalty, and it re-couples `λ` to the posterior). The route that
actually targets the *shape* is to go back to the parent and ask why Pinsker gave a square root, and
whether a different parameter-free lower bound on the binary KL is tighter in my regime.

Pinsker lower-bounds the binary KL by `2(p−q)²`, a parabola *symmetric* in `p` around `q`; inverting
that parabola gives the additive `√`. But the true binary KL is asymmetric: for `q < p` (empirical risk
below true risk, my case), as `p` shrinks toward `q` the true `kl` grows faster than the symmetric
parabola allows. The refined Pinsker inequality `kl(q‖p) ≥ (p−q)²/(2p)` is the sharper lower bound for
exactly that side. Comparing prefactors — refined `1/(2p)`, standard `2` — refined is tighter precisely
when `1/(2p) > 2`, i.e. `p < 1/4`. A true risk below 25% is exactly where a working network lives, and I
can check this is my regime from the McAllester certificates rather than assume it: they are all well
under 0.13 (CNN 0.0164, FCN 0.0373, Fashion 0.1002), so `p ≤ 0.10 < 0.25` with huge margin, and at
`p = 0.10` the refined prefactor `1/(2p) = 5` is 2.5× the standard `2` — on MNIST-CNN at `p ≈ 0.016`,
`1/(2p) ≈ 31`, more than fifteen times Pinsker's `2`. So I have been relaxing the parent with the wrong
inequality for my regime, and the refined one is sharp here by a factor that grows as the risk shrinks.

Substitute the refined bound into the parent. Let `C = (KL + log(2√n/δ))/n` be the bare parent budget.
Then `(E_Q[R] − E_Q[r])²/(2 E_Q[R]) ≤ C`, i.e. `(E_Q[R] − E_Q[r])² ≤ 2 E_Q[R] C`; taking the root in the
certificate case `E_Q[R] ≥ E_Q[r]`, `E_Q[R] ≤ E_Q[r] + √(2 E_Q[R] C)`. And there is the catch:
`E_Q[R]`, the thing I am bounding, sits on both sides, under the root. This is not explicit — but it is
a quadratic in `√(E_Q[R])`. Set `x = √(E_Q[R])`: `x² ≤ E_Q[r] + √(2C)·x`, i.e.
`x² − √(2C)·x − E_Q[r] ≤ 0`, non-positive between its roots, so `x ≤ (√(2C) + √(2C + 4 E_Q[r]))/2`.
Squaring back and pulling the `1/2` inside the roots (`√(2C)/2 = √(C/2)`),
```
E_Q[R] ≤ ( √(E_Q[r] + C/2) + √(C/2) )²,
```
with `C/2 = (KL + log(2√n/δ))/(2n)` — the *same* `kl_term` as the additive bound, the `2n` re-explained
(the `n` from the parent, the `2` from the refined `(p−q)²/(2p)`). `E_Q[R]` is gone from the right side:
the quadratic-in-`√(E_Q[R])` trick simultaneously used the inequality tight in my low-risk regime and
eliminated the implicit risk, with no free parameter. This is the fquad bound.

The "square back" step is where a factor of two likes to hide, so I check it on the MNIST-CNN operating
point `C/2 = kl_term = 10⁻³`, `E_Q[r] = 0.012`: the closed form gives
`(√0.013 + √0.001)² = (0.114018 + 0.031623)² = 0.021211`, and solving the quadratic in `x` independently
gives the same 0.02121 — no dropped or doubled `2`. Two shape checks: as `kl_term → 0` the bound is
`(√E_Q[r])² = E_Q[r]`, collapsing to the empirical risk with zero complexity, correct; and
`(√(emp+k)+√k)² = emp + 2k + 2√(k(emp+k)) ≥ emp`, so it never dips below the empirical risk — a genuine
upper relaxation, monotone increasing in both `emp` and `k`, the two-lever monotonicity I rely on at
inversion time.

Does it actually attack the square-root floor? Compare head to head at zero empirical risk, McAllester's
stuck regime. At `E_Q[r] = 0` the additive bound gives `√(kl_term)`; fquad gives
`(2√(kl_term))² = 4·kl_term`. With `kl_term ≈ 10⁻³` that is `4·10⁻³` versus `3.16·10⁻²` — roughly eight
times tighter. And the advantage widens as `kl_term` shrinks, since `4k/√k = 4√k → 0`: at the CNN's
smaller `kl_term` fquad pulls further ahead, which is the right direction, since the CNN is where the KL
is smallest and the `√` floor bites hardest. The structural reason is the wall diagnosis exactly:
fquad's complexity contribution is `O(kl_term)`, *linear*, while McAllester's is `O(√(kl_term))` — the
fast-rate realizable `1/n` behavior instead of `1/√n`, the direct payoff of refined Pinsker being sharp
below true risk 1/4. I keep the parameter-free property *and* fix the square-root floor.

I should not overclaim: fquad is not universally tighter, only in this regime. Setting the two equal at
`E_Q[r] = 0` gives `4·kl_term = √(kl_term)`, i.e. `kl_term = 1/16 = 0.0625`; below that fquad wins, above
it the additive bound is tighter. My measured `kl_term` values are `~10⁻³`, two orders of magnitude below
the crossover — it would take a KL of order `0.0625·2n ≈ 3750` nats to reach it, which the stability fix
keeps me nowhere near. So the shape improvement is not a fragile knife-edge; I would have to inflate KL
by more than three orders of magnitude to lose it.

I should also test the regime claim where it is *weakest*, not only where it is strongest. The hardest
setting is FashionMNIST-CNN: McAllester certifies it at 0.1002 with empirical 0-1 risk ~0.085, the
highest true-risk of the three and therefore the closest to the `p = 1/4` crossover above which refined
Pinsker stops helping. At `p ≈ 0.10` the refined prefactor is `1/(2p) = 5` versus `2` — still a genuine
improvement, but nothing like the CNN's 15×+ margin. Its `kl_term` is larger too: at KL 32,
`kl_term = (32+9.54)/60000 = 6.9·10⁻⁴`, so on the bounded CE bound McAllester's
`0.085 + √(6.9·10⁻⁴) = 0.111` runs against fquad's
`(√(0.085+6.9·10⁻⁴)+√(6.9·10⁻⁴))² = 0.1018` — fquad still tighter, but by ~0.009 rather than the CNN's
2× factor. That is the honest shape of the win: fquad helps *most* where McAllester was most stuck (tiny
risk, tiny KL, the `√` floor dominating) and *least* where the empirical risk already dominates the
certificate. So my headline expectation should be a large relative improvement on the MNIST settings and
a smaller absolute one on FashionMNIST; the reverse would mean the mechanism story is wrong.

One implementation detail this rung introduces that the previous two omitted is load-bearing for fquad
specifically. The earlier rungs fed the NLL surrogate *unrescaled*; fquad will not tolerate that, and I
can see why from the formula. `(√(E_Q[r] + C/2) + √(C/2))²` assumes `E_Q[r] ∈ [0,1]`, but the clamped
NLL ranges up to `log(1/pmin) = ln(10⁵) = 11.51`, more than an order of magnitude above 1. Feed that raw
and the empirical term inside the root is mis-scaled against the `C/2 ≈ 10⁻³` complexity term by a factor
of thousands, and the outer `(·)²` *amplifies* the miscalibration — the objective stops upper-bounding
anything the parent guarantees, and the posterior, chasing a grossly inflated empirical term, drifts far
from the prior and reinflates the KL I worked to suppress. In the additive bound an over-1 NLL was only
an additive offset; inside fquad's square it becomes a multiplicative distortion of the fit-versus-
complexity balance. So for fquad I add the bounded-loss rescaling: clamp `log_softmax` at `log(pmin)`,
then multiply by `_loss_scale = 1/log(1/pmin) ≈ 0.0869`, landing the surrogate in `[0,1]`. That is the
calibration that lets the linear-in-KL form be tighter in practice rather than only on paper, which is
why this rung must change the surrogate even though the last deliberately did not. I also clamp both
arguments under the square roots at zero so a floating-point negative from a tiny `kl_term` does not
crash `sqrt`.

The certificate stays separate and stays the tighter object. I train against fquad — convex,
parameter-free, tight at low risk — but report the PAC-Bayes-kl inversion: `compute_risk_certificate`
MC-samples the empirical 0-1 risk, reads the KL, forms the bare parent budget
`c = (KL + log(2√n/δ))/n` (`/n`, not `/(2n)`: the `2` lived only in the fquad relaxation), and returns
`inv_kl(emp_risk_01, c)`. The fquad shape acts on the certificate only *indirectly* — the reported
number is still the parent inversion, identical machinery to the two prior rungs, so fquad can change it
only by changing what the run *lands at*: a smaller KL and a low empirical 0-1 risk. As before, a single
uncorrected inversion; the empirical NLL fed into `ce_bound` uses the same `_loss_scale` as training so
the two are consistent — which is also why I expect the `ce_bound` column, which wears the fquad shape
directly, to move far more than the `risk_certificate` column, which only inherits it through the
learned KL.

My falsifiable expectation against McAllester's numbers. The KL should stay small and stable — fquad is
parameter-free, so the stability win holds — and the bounded-loss rescaling should if anything let KL
drop *further*, since the objective no longer over-weights an unrescaled empirical term against the
complexity. So I predict the KL collapses by roughly another order of magnitude, from McAllester's 81 on
MNIST-FCN into the single digits, and its already-small CNN KLs to low single digits. The certificate
should improve modestly but unmistakably: at these small KLs the shape change from `√(kl_term)` to
`4·kl_term` is exactly what lowers the floor, so each of McAllester's certificates (0.0373, 0.0164,
0.1002) should tick down, most visibly on the MNIST settings where the `√` floor was proportionally
largest and least on FashionMNIST where the empirical risk already dominates. The improvement is bounded
below by the floor arithmetic — even at KL 0 the MNIST-CNN floor is ~0.0150, so getting close means
crushing the residual KL against the irreducible confidence budget. The sharpest signature of the
mechanism is the `ce_bound` column: it should fall by a large factor — McAllester's MNIST-CNN 0.0805 to
well under 0.01 — because it wears the shape change directly, whereas the reported certificate only
inherits it through the KL. What would falsify me: if fquad's certificate matches or exceeds
McAllester's, either the rescaling destabilized the posterior (I would see a KL that fails to drop or
grows) or the low-risk regime is not actually below the 1/4 crossover — but the measured true risks are
all well under 0.13, so I am confident this is the regime fquad is built for.
