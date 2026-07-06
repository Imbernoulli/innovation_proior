The Catoni run confirmed exactly the failure mode I was worried about, and the number that tells the
story is `kl_divergence`. On MNIST-FCN the KL means 539 across seeds — but the mean hides the shape.
Seed 42 landed at 131.92 while seeds 123 and 456 blew up to 731.99 and 754.10, a high-to-low ratio of
5.7×. That is not noise around a stable operating point; it is a bimodal outcome where two of three runs
let the posterior flee the prior. And the certificate tracks it monotonically: seed 42's KL of 132 gives
`risk_certificate` 0.0404, while the 754-KL seed gives 0.0647 — a 60% looser certificate (ratio 1.60)
from the same architecture, same data, only a different `λ`/posterior corner. The tell that this is
*entirely* a KL effect and not a fitting effect: the empirical 0-1 risk barely moved across seeds
(0.0228, 0.0166, 0.0175), and — pointedly — the two high-KL seeds fit *better* empirically (0.0166,
0.0175 versus seed 42's 0.0228) yet certify *worse*. A posterior that fits the bound split marginally
harder and pays for it with a much larger KL is precisely the small-`λ`/weak-penalty trade I sketched:
the free, clamped `λ` co-adapted with the posterior into a high-KL basin, and because I fed an
*unrescaled* NLL the empirical term carried too much weight to act as a counterforce on KL. The same
two-of-three blowup repeats on the CNNs: MNIST-CNN KL 42.20 / 268.77 / 266.59 (high/low 6.4×, certificate
0.0181 / 0.0287 / 0.0282), FashionMNIST-CNN 36.71 / 401.32 / 420.77 (high/low 11.5×, certificate up to
0.1318). The CNN settings are tighter in absolute terms — as I predicted, fewer probabilistic weights,
less room for KL to accumulate — but the instability is the same shape.

There is a sharper reading of that table that I do not want to miss, because it tells me *what kind* of
instability this is. Line the low-KL seed up against the two high-KL seeds in each of the three settings:
MNIST-FCN puts seed 42 at 131.92 against 731.99/754.10; MNIST-CNN puts seed 42 at 42.20 against
268.77/266.59; FashionMNIST-CNN puts seed 42 at 36.71 against 401.32/420.77. Seed 42 is the low basin in
*every single setting*, and seeds 123 and 456 are the high basin in every single setting. That coherence
across three different architectures and two datasets is the fingerprint of an optimization-basin effect,
not an architecture effect — the same random seed that seeds the prior-training ERM run and the posterior
init lands the joint `(λ, Q)` trajectory in the same qualitative corner regardless of whether the model is
a 784-600-600-600-10 FCN or a 2-conv-2-fc CNN. If the blowup were architecture-driven I would expect the
ordering of seeds to reshuffle between FCN and CNN; it does not. So the disease is a bistable dynamical
system — two attractors, a low-KL one and a high-KL one — and the free `λ` is what makes the high-KL
attractor reachable at all. That reframes the fix from "tune the penalty harder" to "delete the second
attractor," which is a structural change, not a numerical one. The `ce_bound` column corroborates: it
rides the KL in lockstep (MNIST-FCN 0.1598 / 0.2315 / 0.2324 against KL 132 / 732 / 754), so both the
training-time bound and the reported certificate are being set by the same runaway column. The diagnosis is
clean: the Catoni bound is valid but its trade-off knob is unstable, and the certificate is being set by
runaway KL, not by empirical risk.

Before I act on that, let me make sure I have the mechanism exactly right, because the fix depends on it.
My claim is that the reported certificate is a *deterministic monotone function of only* `emp_01` and
`KL` — that nothing else in the run leaks into it — so that the entire seed spread is the KL column
feeding a fixed inversion. I can check that on paper. The certificate is `inv_kl(emp_01, (KL+Λ)/n)` with
`Λ = log(2√n/δ) ≈ 9.54` and `n ≈ 30000`. Take seed 42: `c = (131.92+9.54)/30000 = 0.004715`, and
inverting, the largest `p` with `kl(0.0228‖p) ≤ 0.004715` is `p ≈ 0.04040` — the reported `risk_certificate`
is 0.040395. Take seed 456: `c = (754.10+9.54)/30000 = 0.025455`, and `inv_kl(0.017533, 0.025455) ≈
0.06471` — reported 0.064714. Both reproduce to five digits from the two columns alone. And to be sure
this is not an FCN-only coincidence, take a CNN seed: MNIST-CNN seed 42 has emp 0.010867, KL 42.20, so
`c = (42.20+9.54)/30000 = 0.0017247`, and `inv_kl(0.010867, 0.0017247) ≈ 0.01812` — reported 0.018122.
Three settings, three exact hits. So there is no hidden variable; the certificate is a pure readout of
`(emp_01, KL)`, and since it is monotone increasing in the budget `c`, hence in `KL`, everything I do from
here is really a fight to *shrink and stabilize the KL column*. That reframing is the whole rung, and it
also tells me the two levers are cleanly separable: `emp_01` and `KL` enter the inversion independently, so
if I can hold `emp_01` roughly fixed while collapsing `KL`, the certificate follows the `KL` collapse
almost one-for-one. The seed-42-versus-456 pair proves the sensitivity is real — a 5.7× swing in `KL` at
essentially fixed `emp_01` moved the certificate 60%.

So I want to remove the unstable knob. The lesson is that a free trade-off parameter, optimized by SGD
with no analytic pinning, is a liability on this surface — it gives the optimizer a degree of freedom it
uses to relax the KL penalty and pay for it in divergence. Let me weigh the alternatives honestly rather
than jump. One option is to keep Catoni but *pin* `λ` to its analytic optimum each step instead of
letting SGD wander it. But I derived last rung that for a well-fitting posterior (`L̂ → 0`) the optimal
`λ*` tends to `1`, and I should check what `λ = 1` actually buys before committing. The complexity
prefactor is `1/(nλ(1−λ/2))`; its denominator `λ(1−λ/2)` is a downward parabola in `λ`, maximized at
`λ = 1` where it equals `1·(1−0.5) = 0.5`, so the prefactor is *minimized* at `λ = 1`, giving
`1/(n·0.5) = 2/n = 6.67·10⁻⁵`. Compare that to the additive bound's own KL gradient at the operating
point I will land at, `1/(2√(2n·KL))` — at `KL ≈ 80`, `n = 30000` that is `1/(2√(4.8·10⁶)) ≈ 2.3·10⁻⁴`,
about 3.4× *stronger* than a perfectly tuned Catoni prefactor. So even a *perfectly* tuned `λ` sits at the
point of least KL defense, weaker than the parameter-free additive bound I am about to adopt; and
recomputing `λ*` from the current `L̂, KL` every step just re-couples `λ` to the posterior, which is the
coupling that produced the feedback loop in the first place. Pinning the knob does not remove the corner;
it drives straight to it, and it does not even trade the instability for a stronger penalty. That kills the
"tune it better" option on arithmetic, not on feel.

A second option is to fix the calibration problem directly by rescaling the NLL into `[0,1]` with
`1/log(1/pmin)` — that is a genuine fix, but it is *orthogonal* to the knob: it changes how the empirical
term is weighted, not whether there is a `λ` for SGD to abuse. I will hold it in reserve and, deliberately,
keep the *same unrescaled NLL* as the Catoni run here, so that whatever difference I measure this rung is
attributable to the bound functional and the removal of `λ`, not to a surrogate change I made at the same
time. Changing two things at once would leave me unable to say which one collapsed the KL, and given the
whole thesis is that the *knob* is the disease, I want a controlled one-variable experiment. A third
option would be to attack `KL` at its source — shrink the posterior variance floor or the prior sigma — but
those are outside the editable region (the substrate fixes `prior_sigma=0.03` and the layer
parameterization), so they are not mine to move; the only surface I own is the bound functional itself.
That leaves the clean option: go back to the *parameter-free* bound, McAllester's additive square-root
form, which has no `λ` at all. It is also the scaffold default, so this rung is, in effect, the question
"does removing the trade-off knob, at the same surrogate, beat the unstable Catoni run?"

Let me derive the bound from scratch so I know precisely what I am committing to and why it should be
more stable. I start where every PAC-Bayes bound starts: I have a learning algorithm that returns a
distribution `Q` over weights, I predict by drawing `h ~ Q`, and I want a high-probability upper bound on
the `Q`-averaged true risk `E_Q[R(h)]` from the `Q`-averaged empirical risk `E_Q[r(h)]`. The obstruction
is that `Q` is chosen after and because of the data, so a fixed-hypothesis concentration statement does
not transfer. The escape is to certify the distribution, not the hypothesis: fix a data-free reference
`P`, let `Q` be anything, and charge complexity as `KL(Q‖P)` — zero when `Q=P` (nothing learned),
growing as `Q` flees the prior. The change-of-measure inequality `E_Q[φ] ≤ KL(Q‖P) + log E_P[e^φ]`
transports a per-hypothesis exponential moment under the fixed `P` onto the data-dependent `Q`; it is
just `KL ≥ 0` applied to the gap between `Q` and the Gibbs tilt of `P` by `φ`. Choosing
`φ(h) = n·kl(r(h)‖R(h))`, the binary KL between empirical and true Bernoulli risk, and using Maurer's
sharp moment control `E_S[e^{n·kl(r‖R)}] ≤ 2√n` for `n ≥ 8`, plus Markov and Jensen, gives the parent:
with probability `1−δ`, simultaneously for all `Q`, `kl(E_Q[r]‖E_Q[R]) ≤ (KL(Q‖P) + log(2√n/δ))/n`. The
`2√n` is the sharp constant — Maurer's halving of `log(2n)` to `log(2√n)` — and it is the same parent the
Catoni bound relaxed; I am now relaxing it differently. It is worth pausing on *why* it is the same
parent: Catoni and McAllester are not two different theorems, they are two different one-sided relaxations
of one binary-KL inequality, which is exactly why the reported certificate — the numerical inversion of
that shared parent — is identical machinery across both rungs and only the *training* objective changes.
That is what lets me compare them cleanly.

The relaxation that removes the trade-off parameter is Pinsker's inequality, `kl(p‖q) ≥ 2(p−q)²`. Apply
it to the left of the parent: `2(E_Q[R] − E_Q[r])² ≤ (KL + log(2√n/δ))/n`, so
`(E_Q[R] − E_Q[r])² ≤ (KL + log(2√n/δ))/(2n)`, and taking the upper root,
`E_Q[R] ≤ E_Q[r] + √((KL(Q‖P) + log(2√n/δ))/(2n))`. The `2` from Pinsker's `2(p−q)²` lands in the
denominator inside the root, turning the parent's `/n` into `/(2n)` — I should keep that accounting
straight, because the certificate I report later inverts the *bare* parent and so uses `/n`, not `/(2n)`;
the `2` is a fingerprint of the Pinsker relaxation and lives only in the additive training bound. Let me
sanity-check the units and the two limits, because a sign or factor error here would poison the whole rung.
Dimensionally `KL` and `Λ` are in nats, `/(2n)` makes the ratio dimensionless, the square root of a
dimensionless risk-squared is a risk, and it is added to a risk `E_Q[r]` — consistent. In the limit
`Q = P` we have `KL = 0` and the bound is `E_Q[r] + √(Λ/(2n)) = E_Q[r] + √(9.54/60000) ≈ E_Q[r] + 0.0126`,
a small irreducible confidence gap, exactly what a bound at zero complexity should give — not zero (there
is always sampling uncertainty) and not vacuous. In the opposite limit `KL → ∞` the root grows without
bound and the certificate saturates at 1, as it must for a bound on a `[0,1]` risk. Both limits behave, so
the functional is the McAllester/Maurer additive certificate, and it has exactly the property I want for
stability: it is closed, additive, and — crucially — it has *no free parameter*. There is nothing for SGD
to co-adapt into a bad corner. The trade-off between empirical fit and complexity is fixed by the
functional form, not by a knob.

Now I have to confront the very thing that made me leave this bound in the first place: the square-root
shape. When `E_Q[r] → 0`, the bound collapses to `√((KL+Λ)/(2n))`, and its gradient with respect to KL
is `1/(2√(2n(KL+Λ)))`, which *shrinks* as KL grows — halving from `1.95·10⁻⁴` at `KL=100` to `0.90·10⁻⁴`
at `KL=500`, the same sublinear penalty I complained about. So the additive bound penalizes KL weakly,
the opposite of Catoni's linear penalty. But the Catoni run just taught me the dual lesson, and I have to
read it precisely: the runaway KL was *not* driven by the KL penalty being too weak — it was driven by
the `λ` mechanism, `λ` drifting toward the weak-penalty region while the empirical discount `1/(1−λ/2)`
and the `KL → b → λ* → weaker-penalty` feedback loop let the posterior buy a large KL cheaply. Remove `λ`
and that escape route is gone: the empirical term is always weighted exactly 1, so the posterior cannot
discount its way into a high-KL configuration, and there is no prefactor for the optimizer to shrink. The
additive bound's weak KL gradient is *enough as long as nothing is actively pushing KL up*, and removing
`λ` removes the pusher. Put the two attractors from the feedback table next to this: the high-KL attractor
existed because `λ` could slide small and rebate the empirical term; with `λ` gone the system is
monostable, and a weak restoring gradient around a single stable point is fine — it does not need to fight
a competing basin. So the bet is: a weak-but-honest sublinear penalty with no knob beats a
strong-but-abusable linear penalty with a free knob, because the instability, not the penalty strength,
was the disease. I expect McAllester to sit at a far smaller, far more *stable* KL than Catoni.

Let me be concrete about the implementation, because this rung is the literal scaffold default and the
details are load-bearing. `compute_bound` is the additive formula: `empirical_risk + √((kl + Λ)/(2n))`
with `Λ = log(2√n/δ)`. `train_step` does a stochastic forward pass, computes the NLL surrogate for the
0-1 loss — `F.nll_loss` on `log_softmax` clamped below at `log(pmin)` — reads the KL from
`get_total_kl`, and returns the bound; no second optimizer, no detached side-update, nothing but the one
scalar. The contrast with the Catoni `train_step` is the whole point of the rung made concrete: there I
carried `self._lambda_param`, an `SGD` optimizer over it, a detached `lam_bound.backward()` side-step, and
a device-migration dance to keep `λ` on the right device — an entire second control loop. Here all of that
evaporates; the posterior parameters are the *only* thing the outer optimizer sees, so there is no way for
a co-adapted scalar to exist. That is not a code simplification for its own sake, it is the mechanism of
the fix: the reason the KL cannot run is that there is nothing left in the objective for it to bargain
with. I note again the one deliberate choice that matches this edit surface and departs from the textbook
recipe: the NLL is *not* rescaled by `1/log(1/pmin)`, so the surrogate can exceed 1. I keep it unrescaled
both to match the default and to make the comparison with Catoni clean. And because the additive form
*adds* `√(kl_term)` rather than *dividing* by `1−λ/2`, an unrescaled NLL that occasionally exceeds 1 is far
less damaging here than in Catoni — there is no empirical discount to exploit, so the mis-scaling inflates
the loss uniformly instead of opening a KL-for-cheap trade. An additive constant offset on the objective
does not move the argmin the way a multiplicative rebate on one term does; that is why I can afford to
leave the calibration for later without fearing it destabilizes this rung.

The certificate is the same separate-from-training story. I train against the additive bound but report
the tighter PAC-Bayes-kl inversion: `compute_risk_certificate` MC-samples the empirical 0-1 risk via
`compute_01_risk`, reads the KL, forms `c = (KL + log(2√n/δ))/n` — the `/n`, not `/(2n)`, exactly because
the inversion uses the bare parent budget and the `2` only appeared in the additive relaxation — and
returns `inv_kl(emp_risk_01, c)`. This is deliberately the *tighter* object than what I trained on: the
additive `√` is an upper relaxation of the parent, so reporting the parent inversion directly can only
help, and it costs nothing because the inversion is a post-hoc bisection over the already-learned
`(emp_01, KL)`. As in the Catoni rung I keep this single-inversion and uncorrected, no inner Monte-Carlo
correction for posterior-sampling error, matching the scaffold's style. I also report the additive
`ce_bound` by feeding empirical NLL and KL through `compute_bound`.

Here is my falsifiable expectation against the Catoni numbers, and I can compute the predictions rather
than gesture at them. The decisive metric is `kl_divergence`, and I predict McAllester collapses it. On
MNIST-FCN, where Catoni's KL means 539 and spiked to 754, I expect McAllester's KL to land roughly an
order of magnitude lower — in the tens-to-low-hundreds — and, just as important, to be *stable across
seeds*, because removing `λ` removes the bimodal corner; concretely I expect the seed-42/123/456 spread to
shrink from the 5.7× Catoni ratio toward something under 1.5×. To see what such a collapse does to the
certificate, take a representative post-collapse KL in the low tens with the empirical 0-1 risk ticking up
slightly to ~0.0235 (McAllester does not discount the empirical term, so the posterior fits a touch less
hard): at a hypothetical KL around 80, `c = (80+9.54)/30000 = 0.00298` and `inv_kl(0.0235, 0.00298) ≈ 0.037`,
so I expect MNIST-FCN's `risk_certificate` to fall from Catoni's 0.0558 mean into the high-0.03s. The CNN
settings should track the same collapse — their Catoni KLs were already smaller, so an order-of-magnitude
drop should carry both certificates comfortably under their Catoni means (0.0250 and 0.1215) as well. The
small empirical increase is far outweighed by the KL collapse in the inversion, because the certificate's
KL-sensitivity is steepest exactly in the range Catoni's KL was blowing through. The thing that would
falsify me: if McAllester's KL stays in the hundreds, then the problem was never the `λ` knob but
something in the shared substrate (the prior split, the unrescaled NLL), and removing the knob bought
nothing.

And the thing I am explicitly *not* claiming: I do not expect McAllester to be the tightest possible
bound. Once the KL is stabilized small, the reported certificate sits near its `inv_kl` floor — at
emp ~0.012 and KL ~11 the inversion gives ~0.0165, and even at `KL = 0` the floor from the confidence
term alone is `inv_kl(0.012, 9.54/30000) ≈ 0.0150`, so of the ~0.005 daylight above the empirical risk,
about three-fifths (0.0150 − 0.012 = 0.003) is the irreducible `Λ/n` confidence budget and only the
remaining ~0.0015 is the residual KL — the runaway is gone and what is left is mostly the confidence term
plus the shape of the relaxation. That decomposition matters because it tells me where the *next* lever is.
The remaining slack is not in the certificate arithmetic — that is already the tight parent inversion — but
in the *training objective*, which still wears the additive `√`. That `√` is the Pinsker relaxation, and
Pinsker's symmetric parabola `2(p−q)²` is loosest precisely when the true risk is small: it under-rewards
the posterior for driving the empirical risk down in exactly the realizable regime I now sit in, so it
does not press KL as low as a relaxation tuned to small risk would. A training bound that is
tight at small risk should drive KL smaller still and, through the same monotone inversion, tighten the
reported number — that is the wall I will aim at next. For now the goal is narrower and clean: kill the
knob, collapse and stabilize the KL. The full scaffold module — additive `compute_bound`, the plain
`train_step`, and the single-`inv_kl` `compute_risk_certificate` — is in the answer.
