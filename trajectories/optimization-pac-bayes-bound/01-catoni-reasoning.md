I start from the default scaffold, which fills `compute_bound` with the McAllester/Maurer additive
square-root form `L̂ + √((KL + log(2√n/δ))/(2n))`, and I want to ask whether that is the right shape
to be *training against* in the first place. The whole certificate, in this substrate, is decided by
the bound functional: the stochastic layers, the data-dependent prior split, and the SGD loop are all
fixed, so the only lever I have is which inequality I push the posterior down. The additive bound has a
property I should look at hard before I commit to it. When the posterior fits the bound split — which it
will, because I initialize it at the ERM minimizer of the prior — the empirical risk `L̂(Q)` goes toward
zero, and the bound collapses to `√((KL + log(2√n/δ))/(2n))`. The complexity enters through a *square
root*. That means the certificate is governed by `√(KL)`, and the gradient of `√(KL)` with respect to
the posterior parameters is `1/(2√(KL))` times the gradient of `KL` — it *flattens* as `KL` grows. The
additive objective, in other words, only weakly penalizes a posterior that drifts away from the prior,
because the marginal cost of one more nat of KL shrinks as the KL accumulates. I suspect that is the
wrong incentive: I want the bound to fight KL growth proportionally, not sublinearly.

So let me look for a bound whose complexity term is *linear* in KL near the operating point, and which I
can still differentiate and minimize directly. The standard family that buys this is the localized,
trade-off-parameter bounds, and Catoni's is the canonical one. Let me derive it from the same root the
default came from, the PAC-Bayes-kl inequality, so I understand exactly what I am relaxing. The change
of measure plus Maurer's sharp moment bound `E_S[e^{n·kl(L̂‖L)}] ≤ 2√n` gives, with probability `1−δ`,
simultaneously for all `Q`, `kl(L̂(Q)‖L(Q)) ≤ (KL(Q‖P) + log(2√n/δ))/n`. This is the parent. The
additive default is the Pinsker relaxation `kl(p‖q) ≥ 2(p−q)²` of it; that relaxation is symmetric in
its argument and loose precisely when the true risk is small, which is my regime. The Catoni route
relaxes the *same* parent differently — through a tilt engineered to give a linear-in-`L̂`,
linear-in-KL trade-off — and the price is a free parameter `λ`.

Let me write the Catoni/lambda functional and check its structure. The bound is
`L(Q) ≤ L̂(Q)/(1−λ/2) + (KL(Q‖P) + log(2√n/δ))/(nλ(1−λ/2))`, valid for `λ ∈ (0,2)`. Two things to
verify. First, that it is a genuine certificate — it is, for any *fixed* `λ` chosen before the data;
this is exactly Catoni's localized bound. Second, that it is convex in `Q` for fixed `λ`: the numerator
`L̂(Q)` is linear in `Q`, `KL(Q‖P)` is convex, and the denominators `1−λ/2` and `nλ(1−λ/2)` are
positive constants once `λ ∈ (0,2)` is fixed — so the right-hand side is a positive-weighted sum of a
linear and a convex functional, convex in `Q`. Good: this is differentiable and well-posed as a
training objective, and the complexity term `(KL+const)/(nλ(1−λ/2))` is *linear* in KL, which is exactly
the proportional penalty the additive bound lacked. Near `L̂ = 0` the Catoni bound is roughly
`(KL + log(2√n/δ))/(nλ(1−λ/2))` — order `KL/n`, linear — versus the additive `√(KL/(2n))`. For small
`KL/n` the additive `√` is actually *smaller numerically*, so I should be honest that the linear form
does not automatically win as a certificate; what it wins is the *training dynamics* — a posterior
trained against a linear-in-KL objective is pushed harder to keep KL down, and whether that nets out
tighter depends on the `λ` it finds.

That is the crux: `λ`. Catoni's bound holds for one fixed `λ`. If I want the trade-off to suit the data
— and I do, because the right balance between fitting and staying near the prior depends on how much the
posterior must move — I have to tune `λ`. The textbook-clean way is a uniform-in-`λ` bound that holds
for all `λ` at once via a deterministic AM-GM identity, which would let me optimize `λ` continuously for
free. But I have to be careful about what the *task's* edit surface actually lets me build, because the
contract is just `compute_bound`/`train_step`/`compute_risk_certificate` over the fixed substrate, and
the substrate's `model(x, sample=…)` and `get_total_kl` give me a Gaussian posterior with an analytic
KL — not an explicit Gibbs measure I can form, and not a recurrent or sigmoid-scaled `λ` head wired
into the loop. So the cleanest thing that fits the contract is to carry `λ` as my *own* learnable scalar
inside the `BoundOptimizer`, give it its own optimizer, and update it by gradient descent on the same
bound — a numerical alternating minimization rather than the closed-form one. That is the design I will
land.

Now the mechanics, and there is a real trap here that I need to reason through, because it is the
difference between this baseline working and failing. The outer SGD loop in the substrate constructs
*one* optimizer over `model.parameters()` and steps it after `train_step` returns the loss. My `λ` is
not a model parameter — it lives in the `BoundOptimizer` — so the loop's `optimizer.step()` will never
touch it. If I do nothing special, `λ` stays frozen at its initialization for the entire run. So I must
give `λ` its *own* optimizer and step it myself inside `train_step`. And I must step it on a *detached*
copy of the loss: the posterior's gradient flows through `optimizer.step()` in the loop using the bound
I return, so if I also backprop the `λ`-update through the live `nll`/`kl` graph I would either
double-count or corrupt the posterior gradient. The right structure is: compute the bound for the
posterior (the value I return, with live graph), then *separately* form a detached scalar
`L̂.detach()/(1−λ/2) + (KL.detach()+log(2√n/δ))/(nλ(1−λ/2))`, backprop *that* into `λ` only, and step
`λ`'s optimizer. The detach makes the `λ` step see `L̂` and `KL` as constants — which is correct,
because at the alternating-minimization `λ`-substep the posterior is held fixed.

The second trap is the range of `λ`. The bound has a singularity at `λ = 2` (the `1−λ/2` denominator
hits zero) and degenerates at `λ = 0` (the `1/λ` blows up). I argued the optimal `λ` sits well inside
`(0,2)` — for `L̂ = 0` the closed-form optimum is `λ = 1`, and it only decreases as `L̂` grows, with a
floor around `1/√n`. But a free SGD scalar can wander anywhere, and if it ever reaches `λ ≥ 2` the
denominator flips sign and the bound becomes negative and meaningless — the optimizer would then happily
drive `λ` to the singularity to "minimize" a bound that is no longer a bound. So I clamp `λ` into a safe
open interval, `(0.01, 1.99)`, every time I read it. The clamp at `0.01` keeps the `1/λ` finite; the
clamp at `1.99` keeps `1−λ/2 > 0`. This is the literal guard the edit surface needs — not the elegant
sigmoid-into-`[1/√n,1]` reparameterization, just a hard clamp on a raw learnable scalar, initialized at
`0.5`.

Let me reason about *where* I expect this to land relative to the default, because that is the prediction
I am making. The concern that nags me is the interaction between the linear-in-KL Catoni objective and
the SGD it is trained with. The Catoni bound's complexity term has `1/(nλ(1−λ/2))` in front of KL; with
`λ` starting at `0.5`, that prefactor is `1/(n·0.5·0.75) = 1/(0.375n)`, roughly `2.67/n`. Compare the
additive bound, whose KL contribution near `L̂=0` is `√(KL/(2n))` with derivative `1/(2√(2n·KL))`. For
the KL values a Gaussian posterior over a 600-wide net reaches — easily tens to hundreds of nats — the
Catoni objective's *gradient* with respect to KL is the constant `2.67/n`, while the additive's gradient
*shrinks* as `1/√KL`. That cuts both ways: the constant gradient keeps pressing on KL, which is good,
but if `λ` drifts small (toward its `0.01` floor), the prefactor `1/(nλ(1−λ/2))` *explodes*, and a tiny
`λ` makes the complexity term enormous, which paradoxically can let the posterior trade a large KL for a
small `λ`-discounted empirical term — `L̂/(1−λ/2)` with `λ→0` approaches `L̂`, the cheapest possible
empirical weighting. A free, clamped `λ` with no closed-form pinning is exactly the kind of knob that can
settle in a bad corner: small `λ`, large KL, a bound that is technically valid but loose because KL ran
away. I think this is the most likely failure mode of *this* simpler implementation versus the
closed-form alternating one — the `λ` step is just SGD on a detached scalar, with no guarantee it tracks
the analytic optimum, so the posterior and `λ` can co-adapt into a high-KL configuration.

The final certificate is separate from the training objective and I should be deliberate about it. I
train against the Catoni functional, but for the *reported* number I want the tightest valid bound on the
learned posterior, which is PAC-Bayes-kl itself, inverted. So `compute_risk_certificate` MC-samples the
stochastic predictor's empirical 0-1 risk on the bound set via `compute_01_risk`, reads off the KL from
one forward pass, forms `c = (KL + log(2√n/δ))/n`, and calls `inv_kl(emp_risk_01, c)` for the reported
`risk_certificate`. For this baseline I keep it simple — a single `inv_kl` on the raw MC estimate, with
no inner Monte-Carlo correction — matching the unrescaled, uncorrected style of the scaffold default; the
nested inner-`inv_kl` correction for posterior-sampling error is a refinement I am not adding here. I also
report the Catoni `ce_bound` by feeding the empirical NLL and KL through `compute_bound`, and the `λ` I
converged to, as a diagnostic. The empirical risk surrogate during training is the plain clamped NLL,
`F.nll_loss` on `log_softmax` floored at `log(pmin)` — *not* rescaled by `1/log(1/pmin)`. I note that
choice deliberately: the rescaling that maps NLL into `[0,1]` is what calibrates the empirical-risk
term against the KL term, and leaving it out means the Catoni objective is being fed an empirical term
that can exceed 1, which over-weights the empirical side relative to KL early in training. I expect this
unrescaled feed, combined with the free `λ`, to be part of why the KL is not held down.

So here is my falsifiable expectation for this rung. The Catoni/lambda bound, with a clamped free `λ`
and an unrescaled NLL surrogate, is a legitimate convex-in-`Q` certificate and a reasonable first thing
to try, and it should produce a non-vacuous `risk_certificate` well below 1 — the data-dependent prior
guarantees the certificate is in the few-percent range, not the vacuous range. But I predict it will be
the *loosest* of the bound formulations I can build on this surface, dominated by a KL that runs large.
The signature I am watching for is `kl_divergence`: if the free `λ` settles small and the posterior
exploits the discounted empirical term, the KL should balloon into the hundreds, and because the
certificate `inv_kl(L̂_01, (KL+log(2√n/δ))/n)` grows monotonically in KL, a KL of a few hundred over
`n ≈ 30000` gives `c` on the order of `0.01` and inflates the inverted certificate noticeably above the
empirical 0-1 risk. I expect, concretely, that the FCN — with its three 600-wide hidden layers and the
largest parameter count, hence the most room for KL to grow — will show the worst certificate and by far
the largest KL, with the CNN settings tighter simply because they have fewer probabilistic weights. If
instead the KL stays small (tens of nats) and the certificate comes in near the additive default, then my
diagnosis of the free-`λ` failure mode is wrong and I will have to look elsewhere. The full scaffold
module — `compute_bound`, the dual-optimizer `train_step`, and the single-`inv_kl`
`compute_risk_certificate` — is in the answer.
