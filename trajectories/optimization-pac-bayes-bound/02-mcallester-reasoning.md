The Catoni run confirmed exactly the failure mode I was worried about, and the number that tells the
story is `kl_divergence`. On MNIST-FCN the KL means 539 across seeds — but look at the spread: seed 42
landed at 131.92 while seeds 123 and 456 blew up to 731.99 and 754.10. That is not noise around a stable
operating point; that is a bimodal outcome where two of three runs let the posterior flee the prior. And
the certificate tracks it monotonically: seed 42's KL of 132 gives `risk_certificate` 0.0404, while the
754-KL seed gives 0.0647 — a 60% looser certificate from the same architecture, same data, only a
different `λ`/posterior corner. The empirical 0-1 risk barely moved across seeds (0.0228, 0.0166,
0.0175), so the posterior is fitting the bound split fine; the entire gap in the certificate is KL. The
free, clamped `λ` co-adapted with the posterior into the small-`λ`/large-KL corner I sketched, and
because I fed an *unrescaled* NLL the empirical term carried too much weight to act as a counterforce on
KL. The CNN settings are tighter (KL 42–269 on MNIST-CNN, certificate 0.018–0.029) precisely because
they have fewer probabilistic weights for KL to accumulate in, exactly as I predicted — but even there
the same two-of-three blowup appears (seed 42 KL 42, seeds 123/456 KL ~267). FashionMNIST-CNN is the
worst, certificate mean 0.1215 with KL up to 421. So the diagnosis is clean: the Catoni bound is valid
but its trade-off knob is unstable, and the certificate is being set by runaway KL, not by empirical
risk.

So I want to remove the unstable knob entirely. The lesson is that a free trade-off parameter, optimized
by SGD with no analytic pinning, is a liability on this surface — it gives the optimizer a degree of
freedom it uses to discount the empirical term and pay for it in KL. The cleanest way to kill that
liability is to go back to the *parameter-free* bound: McAllester's additive square-root form, which has
no `λ` at all. That is also the scaffold default, so this rung is, in effect, the question "does removing
the trade-off knob and accepting the square-root shape beat the unstable Catoni run?" Let me derive the
bound from scratch so I know precisely what I am committing to and why it should be more stable.

I start where every PAC-Bayes bound starts: I have a learning algorithm that returns a distribution `Q`
over weights, I predict by drawing `h ~ Q`, and I want a high-probability upper bound on the
`Q`-averaged true risk `E_Q[R(h)]` from the `Q`-averaged empirical risk `E_Q[r(h)]`. The obstruction is
that `Q` is chosen after and because of the data, so a fixed-hypothesis concentration statement does not
transfer. The escape is to certify the distribution, not the hypothesis: fix a data-free reference `P`,
let `Q` be anything, and charge complexity as `KL(Q‖P)` — zero when `Q=P` (nothing learned), growing as
`Q` flees the prior. The change-of-measure inequality, `E_Q[φ] ≤ KL(Q‖P) + log E_P[e^φ]`, transports a
per-hypothesis exponential moment under the fixed `P` onto the data-dependent `Q`; it is just `KL ≥ 0`
applied to the gap between `Q` and the Gibbs tilt of `P` by `φ`. Choosing `φ(h) = n·kl(r(h)‖R(h))`, the
binary KL between empirical and true Bernoulli risk, and using Maurer's sharp moment control
`E_S[e^{n·kl(r‖R)}] ≤ 2√n` for `n ≥ 8`, plus Markov and Jensen, gives the parent: with probability
`1−δ`, simultaneously for all `Q`, `kl(E_Q[r]‖E_Q[R]) ≤ (KL(Q‖P) + log(2√n/δ))/n`. The `2√n` is the
sharp constant — Maurer's halving of `log(2n)` to `log(2√n)` — and it is the same parent the Catoni
bound relaxed; I am now relaxing it differently.

The relaxation that removes the trade-off parameter is Pinsker's inequality, `kl(p‖q) ≥ 2(p−q)²`. Apply
it to the left of the parent: `2(E_Q[R] − E_Q[r])² ≤ (KL + log(2√n/δ))/n`, so
`(E_Q[R] − E_Q[r])² ≤ (KL + log(2√n/δ))/(2n)`, and taking the upper root,
`E_Q[R] ≤ E_Q[r] + √((KL(Q‖P) + log(2√n/δ))/(2n))`. The `2` from Pinsker's `2(p−q)²` lands in the
denominator inside the root, turning the parent's `/n` into `/(2n)`. This is the McAllester/Maurer
additive certificate, and it has exactly the property I want for stability: it is closed, additive,
convex in `Q` (linear `E_Q[r]` plus the square root of a convex KL, and `√` of a convex function
composed this way is still convex here because the KL enters affinely under the root), and — crucially —
it has *no free parameter*. There is nothing for SGD to co-adapt into a bad corner. The trade-off between
empirical fit and complexity is fixed by the functional form, not by a knob.

Now I have to confront the very thing that made me leave this bound in the first place: the square-root
shape. When `E_Q[r] → 0`, the bound collapses to `√(KL/(2n))`, and its gradient with respect to KL is
`1/(2√(2n·KL))`, which *shrinks* as KL grows. So the additive bound penalizes KL sublinearly — the
opposite of Catoni's linear penalty. I argued before that this is a weak incentive to keep KL down. But
the Catoni run just taught me the dual lesson: a *strong* (linear) KL penalty with an *unstable* knob is
worse than a *weak* (sublinear) penalty with *no* knob, because the instability dominates. The question
is which effect wins numerically. Here is the reasoning that makes me bet on McAllester. The runaway KL
in Catoni was not driven by the KL penalty being too weak — it was driven by `λ` drifting small, which
*simultaneously* discounted the empirical term and inflated the complexity prefactor, letting the
posterior buy a large KL cheaply through the `1/(1−λ/2)` empirical discount. Remove `λ` and that escape
route is gone: the empirical term is always weighted exactly 1, so the posterior cannot discount its way
into a high-KL configuration. The additive bound's weak KL gradient is enough *as long as there is no
mechanism actively pushing KL up*, and removing `λ` removes that mechanism. So I expect McAllester to
sit at a far smaller, far more *stable* KL than Catoni, even though its per-nat KL penalty is weaker.

Let me be concrete about the implementation, because this rung is the literal scaffold default and the
details are load-bearing. `compute_bound` is the additive formula:
`empirical_risk + √((kl + log(2√n/δ))/(2n))`. `train_step` does a stochastic forward pass, computes the
NLL surrogate for the 0-1 loss — `F.nll_loss` on `log_softmax` clamped below at `log(pmin)` — reads the
KL from `get_total_kl`, and returns the bound. I note one deliberate choice that matches this task's edit
surface and departs from the textbook recipe: the NLL is *not* rescaled by `1/log(1/pmin)`. The clean
PAC-Bayes derivation requires the loss in `[0,1]`, and the rescaling is what enforces that; feeding the
raw clamped NLL means the surrogate can exceed 1. I keep it unrescaled here both to match the default
and to make the comparison with Catoni clean — both rungs feed the same unrescaled NLL, so any
difference between them is attributable to the bound functional and the `λ`, not to the surrogate
scaling. (I am holding the rescaling in reserve; it is the kind of calibration fix that a later, tighter
bound will want.) Because the additive surrogate adds `√(kl_term)` rather than dividing by `1−λ/2`, an
unrescaled NLL that occasionally exceeds 1 is less damaging here than in Catoni — there is no empirical
discount to exploit.

The certificate is the same separate-from-training story. I train against the additive bound but report
the tighter PAC-Bayes-kl inversion: `compute_risk_certificate` MC-samples the empirical 0-1 risk via
`compute_01_risk`, reads the KL, forms `c = (KL + log(2√n/δ))/n` — note the `/n`, not `/(2n)`, because
the inversion uses the bare parent budget, the `2` only appeared in the additive *relaxation* — and
returns `inv_kl(emp_risk_01, c)`. As in the Catoni rung I keep this single-inversion and uncorrected,
with no inner Monte-Carlo correction for posterior-sampling error, matching the scaffold's style. I also
report the additive `ce_bound` by feeding empirical NLL and KL through `compute_bound`.

Here is my falsifiable expectation against the Catoni numbers. The decisive metric is `kl_divergence`,
and I predict McAllester collapses it. Concretely: on MNIST-FCN, where Catoni's KL means 539 and spiked
to 754, I expect McAllester's KL to land roughly an order of magnitude lower — in the tens-to-low-hundreds
— and, just as important, to be *stable across seeds*, because removing `λ` removes the bimodal corner.
If the additive bound holds KL near 70–90 across all three seeds while Catoni swung 132–754, that is the
stability win. The certificate should follow: a KL of ~80 versus ~539 over the same `n` shrinks the
inversion budget `c` substantially, so I expect `risk_certificate` on MNIST-FCN to drop from Catoni's
0.0558 mean toward the high-0.03s, and the MNIST-CNN certificate from 0.0250 toward the mid-0.01s, with
FashionMNIST-CNN improving from 0.1215 toward ~0.10. The empirical 0-1 risk may tick *up* slightly
relative to Catoni — McAllester does not discount the empirical term, so the posterior fits a touch less
aggressively — but that small empirical increase should be far outweighed by the KL collapse in the
inverted certificate. The thing that would falsify me: if McAllester's KL stays in the hundreds, then the
problem was never the `λ` knob but something in the shared substrate (the prior split, the unrescaled
NLL), and removing the knob bought nothing. And the thing I am explicitly *not* claiming: I do not expect
McAllester to be the tightest possible bound. Its square-root shape still caps the certificate at
`~√(KL/2n)` even with the empirical risk at zero — so once KL is stabilized small, the remaining looseness
is the square root itself, which is the next thing to attack. The full scaffold module — additive
`compute_bound`, the plain `train_step`, and the single-`inv_kl` `compute_risk_certificate` — is in the
answer.
