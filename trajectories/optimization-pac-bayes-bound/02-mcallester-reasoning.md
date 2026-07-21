The Catoni run confirmed exactly the failure mode I worried about, and `kl_divergence` tells the story.
On MNIST-FCN the KL means 539 across seeds — but the mean hides the shape: seed 42 landed at 131.92
while seeds 123 and 456 blew up to 731.99 and 754.10, a 5.7× high/low ratio. That is not noise around a
stable operating point; it is a bimodal outcome where two of three runs let the posterior flee the
prior. The certificate tracks it monotonically — seed 42's KL 132 gives `risk_certificate` 0.0404, the
754-KL seed gives 0.0647, 60% looser from the same architecture and data, only a different `λ`/posterior
corner. And the tell that this is *entirely* a KL effect, not a fitting effect: the empirical 0-1 risk
barely moved (0.0228, 0.0166, 0.0175), and the two high-KL seeds fit *better* empirically yet certify
*worse*. A posterior that fits marginally harder and pays with a much larger KL is precisely the
small-`λ`/weak-penalty trade I sketched — the free clamped `λ` co-adapted with the posterior into a
high-KL basin, and the unrescaled NLL let the empirical term carry too much weight to counter it. The
same two-of-three blowup repeats on the CNNs: MNIST-CNN 42.20 / 268.77 / 266.59 (6.4×), FashionMNIST-CNN
36.71 / 401.32 / 420.77 (11.5×). The CNNs are tighter in absolute terms — fewer probabilistic weights,
as expected — but the instability is the same shape.

There is a sharper reading. Seed 42 is the *low* basin in every single setting (131.92, 42.20, 36.71)
and seeds 123 and 456 are the high basin in every single setting. That coherence across three
architectures and two datasets is the fingerprint of an optimization-basin effect, not an architecture
effect: the same random seed that seeds the prior-training ERM run and the posterior init lands the
joint `(λ, Q)` trajectory in the same qualitative corner regardless of the model. If the blowup were
architecture-driven the seed ordering would reshuffle between FCN and CNN; it does not. So the disease
is a bistable dynamical system — a low-KL attractor and a high-KL one — and the free `λ` is what makes
the high-KL attractor reachable at all. That reframes the fix from "tune the penalty harder" to "delete
the second attractor," a structural change, not a numerical one. The `ce_bound` column corroborates,
riding the KL in lockstep (0.1598 / 0.2315 / 0.2324 against KL 132 / 732 / 754).

I want the mechanism exactly right, because the fix depends on it: the reported certificate is a
deterministic monotone function of *only* `emp_01` and `KL`, so the entire seed spread is the KL column
feeding a fixed inversion. The certificate is `inv_kl(emp_01, (KL+Λ)/n)` with `Λ ≈ 9.54`, `n ≈ 30000`.
Take seed 456: `c = (754.10+9.54)/30000 = 0.025455`, and `inv_kl(0.017533, 0.025455) ≈ 0.06471` —
reported 0.064714, to five digits from the two columns alone, no hidden variable. Since the inversion is
monotone increasing in the budget `c`, hence in `KL`, everything from here is a fight to *shrink and
stabilize the KL column*. And the two levers are cleanly separable — `emp_01` and `KL` enter the
inversion independently — so if I hold `emp_01` roughly fixed while collapsing `KL`, the certificate
follows the KL collapse almost one-for-one, as the 5.7× KL swing at essentially fixed `emp_01` moving
the certificate 60% already showed.

So I remove the unstable knob. A free trade-off parameter optimized by SGD with no analytic pinning is a
liability on this surface — a degree of freedom the optimizer uses to relax the KL penalty and pay in
divergence. The alternatives, weighed honestly: I could keep Catoni but *pin* `λ` to its analytic
optimum each step, but I settled last rung that for a well-fitting posterior `λ*→1`, which is the point
of *least* KL defense (prefactor `2/n = 6.67·10⁻⁵`, weaker than even the additive bound's KL gradient
`≈ 2.3·10⁻⁴` at `KL≈80`), and recomputing `λ*` from the current `L̂, KL` every step just re-couples `λ`
to the posterior — the very coupling that produced the feedback loop. Pinning drives straight to the
corner and does not even buy a stronger penalty. I could fix the calibration by rescaling the NLL into
`[0,1]` — a genuine fix, but *orthogonal* to the knob: it reweights the empirical term, it does not
remove the `λ` for SGD to abuse. I deliberately keep the *same unrescaled NLL* here so whatever I
measure is attributable to the bound functional and the removal of `λ`, not to a surrogate change made
simultaneously — a one-variable experiment, since the thesis is that the knob is the disease. Attacking
KL at its source (variance floor, prior sigma) is outside the editable region. That leaves the clean
option: the parameter-free McAllester additive square-root form, which has no `λ` at all. It is also the
scaffold default, so this rung asks: does removing the trade-off knob, at the same surrogate, beat the
unstable Catoni run?

I relax the same parent PAC-Bayes-kl I derived last rung — `kl(E_Q[r]‖E_Q[R]) ≤ (KL + log(2√n/δ))/n` —
but this time through Pinsker's inequality `kl(p‖q) ≥ 2(p−q)²` instead of Catoni's tilt. Catoni and
McAllester are not two theorems; they are two one-sided relaxations of one binary-KL inequality, which
is exactly why the reported certificate — the inversion of that shared parent — is identical machinery
across both rungs and only the training objective changes. Applying Pinsker to the left of the parent:
`2(E_Q[R] − E_Q[r])² ≤ (KL+Λ)/n`, so `(E_Q[R] − E_Q[r])² ≤ (KL+Λ)/(2n)`, and taking the upper root,
`E_Q[R] ≤ E_Q[r] + √((KL + log(2√n/δ))/(2n))`. The `2` from Pinsker lands in the denominator inside the
root, turning the parent's `/n` into `/(2n)` — a fingerprint of the Pinsker relaxation that lives only in
the additive training bound; the certificate I report later inverts the *bare* parent and so uses `/n`.
This is the McAllester/Maurer additive certificate, and its decisive property is that it has *no free
parameter*: the trade-off between fit and complexity is fixed by the functional form, with nothing for
SGD to co-adapt into a bad corner.

I now have to confront the very thing that made me leave this bound last rung: the square-root shape.
When `E_Q[r] → 0` it collapses to `√((KL+Λ)/(2n))`, whose KL gradient `1/(2√(2n(KL+Λ)))` *shrinks* as KL
grows — the same sublinear penalty I complained about, the opposite of Catoni's linear one. But the
Catoni run taught the dual lesson, read precisely: the runaway KL was *not* driven by the KL penalty
being too weak, it was driven by the `λ` mechanism — `λ` sliding toward the weak-penalty region while
the empirical discount `1/(1−λ/2)` and the `KL → b → λ* → weaker-penalty` feedback let the posterior buy
a large KL cheaply. Remove `λ` and the escape route is gone: the empirical term is always weighted
exactly 1, so the posterior cannot discount its way into a high-KL configuration, and there is no
prefactor to shrink. The weak KL gradient is enough as long as nothing actively pushes KL up, and
removing `λ` removes the pusher — the high-KL attractor existed *because* `λ` could slide small and
rebate the empirical term; with `λ` gone the system is monostable, and a weak restoring gradient around
a single stable point is fine. So the bet is: a weak-but-honest sublinear penalty with no knob beats a
strong-but-abusable linear penalty with a free knob, because the instability, not the penalty strength,
was the disease.

The implementation is the literal scaffold default, and the details are the fix made concrete.
`compute_bound` is `empirical_risk + √((kl + Λ)/(2n))`; `train_step` does a stochastic forward pass,
computes the NLL surrogate (`F.nll_loss` on `log_softmax` clamped below at `log(pmin)`), reads the KL,
and returns the bound — no second optimizer, no detached side-update, nothing but the one scalar. The
contrast with the Catoni `train_step` is the whole point: there I carried `self._lambda_param`, an
optimizer over it, a detached side-step, and a device-migration dance; here all of that evaporates, so
the posterior parameters are the only thing the outer optimizer sees and no co-adapted scalar can exist.
I keep the NLL *unrescaled* (surrogate can exceed 1), both to match the default and to keep the
comparison with Catoni clean. Because the additive form *adds* `√(kl_term)` rather than *dividing* by
`1−λ/2`, an over-1 NLL is far less damaging here: it is an additive offset shifting the loss uniformly,
which does not move the argmin the way a multiplicative rebate on one term does — so I can leave the
calibration for later without fearing it destabilizes this rung.

The certificate stays separate and stays the tighter object: `compute_risk_certificate` MC-samples the
empirical 0-1 risk via `compute_01_risk`, reads the KL, forms `c = (KL + log(2√n/δ))/n` — `/n`, not
`/(2n)`, because the inversion uses the bare parent budget — and returns `inv_kl(emp_risk_01, c)`. The
additive `√` is an upper relaxation of the parent, so reporting the parent inversion directly can only
help and costs nothing (a post-hoc bisection over the learned `(emp_01, KL)`). As before, a single
uncorrected inversion, matching the scaffold style, plus the additive `ce_bound` from empirical NLL and
KL.

My falsifiable expectation against the Catoni numbers: the decisive metric is `kl_divergence`, and I
predict McAllester collapses it. On MNIST-FCN, where Catoni's KL means 539 and spiked to 754, I expect
the KL roughly an order of magnitude lower — tens to low hundreds — and, just as important, *stable
across seeds*, the 5.7× ratio shrinking toward something under 1.5×, because removing `λ` removes the
bimodal corner. A post-collapse KL in the tens, with the empirical 0-1 risk ticking up slightly
(McAllester does not discount the empirical term, so the posterior fits a touch less hard), drops the
certificate well below Catoni's 0.0558 mean, into the high-0.03s — the small empirical increase far
outweighed by the KL collapse, since the certificate's KL-sensitivity is steepest exactly in the range
Catoni was blowing through. The CNNs, whose Catoni KLs were already smaller, should carry both
certificates comfortably under their Catoni means (0.0250, 0.1215). What would falsify me: if
McAllester's KL stays in the hundreds, the problem was never the `λ` knob but something in the shared
substrate, and removing the knob bought nothing.

And what I am *not* claiming: McAllester is not the tightest possible. Once the KL is small the
certificate sits near its `inv_kl` floor — even at `KL=0` the floor from the confidence term alone is
`inv_kl(0.012, 9.54/30000) ≈ 0.0150`, so most of the daylight above the empirical risk is the
irreducible `Λ/n` confidence budget, not squeezable KL. The remaining slack lives not in the certificate
arithmetic — already the tight parent inversion — but in the *training objective*, which still wears the
additive `√`. That `√` is the Pinsker relaxation, and Pinsker's symmetric parabola `2(p−q)²` is loosest
precisely when the true risk is small: it under-rewards driving the empirical risk down in exactly the
realizable regime I now sit in. A training bound tight at small risk should drive KL smaller still and,
through the same monotone inversion, tighten the reported number — that is the wall I aim at next. For
now: kill the knob, collapse and stabilize the KL.
