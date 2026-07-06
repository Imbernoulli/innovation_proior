I start from the default scaffold, which fills `compute_bound` with the McAllester/Maurer additive
square-root form `L̂ + √((KL + log(2√n/δ))/(2n))`, and I want to ask whether that is the right shape
to be *training against* in the first place. The whole certificate, in this substrate, is decided by
the bound functional: the stochastic layers, the data-dependent prior split, and the SGD loop are all
fixed, so the only lever I have is which inequality I push the posterior down. The additive bound has a
property I should look at hard before I commit to it. When the posterior fits the bound split — which it
will, because I initialize it at the ERM minimizer of the prior — the empirical risk `L̂(Q)` goes toward
zero, and the bound collapses to `√((KL + log(2√n/δ))/(2n))`. The complexity enters through a *square
root*. Let me put numbers on the shape, because "square root" alone is not an argument. The bound split
is half of the 60k training set, so `n ≈ 30000`, and `log(2√n/δ) = log(2·173.2/0.025) ≈ 9.54`; call that
constant `Λ`. Then the marginal cost of one more nat of divergence is `d/dKL √((KL+Λ)/(2n)) =
1/(2√(2n(KL+Λ)))`. At `KL = 100` that derivative is `1/(2√(2·30000·109.5)) ≈ 1.95·10⁻⁴`; at `KL = 500`
it has fallen to `≈ 0.90·10⁻⁴`. So going from a hundred to five hundred nats of divergence, the marginal
penalty *halves*. The additive objective, in other words, only weakly penalizes a posterior that drifts
away from the prior, because the marginal cost of one more nat of KL shrinks as the KL accumulates. That
is the wrong incentive: I want the bound to fight KL growth proportionally, not sublinearly.

So let me look for a bound whose complexity term is *linear* in KL near the operating point, and which I
can still differentiate and minimize directly. The standard family that buys this is the localized,
trade-off-parameter bounds, and Catoni's is the canonical one. Let me derive it from the same root the
default came from, the PAC-Bayes-kl inequality, so I understand exactly what I am relaxing. The change-of-measure inequality `E_Q[φ] ≤ KL(Q‖P) + log E_P[e^φ]` — which is nothing but
`KL ≥ 0` applied to the gap between `Q` and the exponential tilt of `P` by `φ` — transports a
per-hypothesis exponential moment computed under the *data-free* `P` onto the data-dependent `Q`.
Choosing `φ(h) = n·kl(L̂(h)‖L(h))`, the binary KL between empirical and true Bernoulli risk, and
controlling its moment with Maurer's sharp bound `E_S[e^{n·kl(L̂‖L)}] ≤ 2√n` (for `n ≥ 8`), then applying
Markov and Jensen, gives, with probability `1−δ`, simultaneously for all `Q`,
`kl(L̂(Q)‖L(Q)) ≤ (KL(Q‖P) + log(2√n/δ))/n`. This is the parent, and its right-hand side is exactly the
`Λ/n`-plus-`KL/n` budget I will be spending in every one of these bounds. The `2√n` is Maurer's halving
of the cruder `log(2n)` constant, and it is what makes `Λ = log(2√n/δ) ≈ 9.54` rather than something a
nat or so larger — small, but it sets the certificate's floor when `KL → 0`, so it is worth carrying
exactly. The additive default is the Pinsker relaxation `kl(p‖q) ≥ 2(p−q)²` of it; that
relaxation is a symmetric parabola in its first argument, and it is loose precisely when the true risk is
small, which is my regime. The Catoni route relaxes the *same* parent differently — through a tilt
engineered to give a linear-in-`L̂`, linear-in-KL trade-off — and the price is a free parameter `λ`.

Before I commit to Catoni let me lay out what is actually on the menu, because the parent admits more
than one descendant and I should reject the alternatives for stated reasons, not by reflex. Option one is
to keep the additive default and do nothing — but I have just argued its marginal KL cost halves from
`1.95·10⁻⁴` to `0.90·10⁻⁴` as KL runs `100 → 500`, so it under-defends the divergence, and there is no
free lunch in leaving it. Option two is the most tempting on paper: train directly against the *inverted*
parent, `inv_kl(L̂, (KL+Λ)/n)`, since that is literally the tightest certificate and it is what I will
report anyway. The obstruction is mechanical. `inv_kl` is defined by bisection — I solve for the largest
`p` with `kl(L̂‖p) ≤ c` — and that root is not a clean differentiable function I can hand the substrate's
autograd to push the Gaussian posterior through. I would be backpropagating through a bisection loop, and
the substrate's `train_step` contract wants a single differentiable scalar assembled from `nll` and
`get_total_kl`, not an implicit-function derivative. So the inverted parent is the right *evaluation* but
the wrong *training* object; I need an explicit, closed-form relaxation to descend. That leaves the two
explicit relaxations of the parent's left-hand `kl`: the symmetric Pinsker parabola, which gives the
additive default, and Catoni's tilt, which gives the localized linear-in-KL form. Since the whole reason
I am moving is to replace a sublinear KL penalty with a proportional one, Catoni is the option that
matches the diagnosis. So the choice is made for a reason: linear complexity, still explicit and
convex, at the cost of one knob.

Let me write the Catoni/lambda functional and check its structure. The bound is
`L(Q) ≤ L̂(Q)/(1−λ/2) + (KL(Q‖P) + log(2√n/δ))/(nλ(1−λ/2))`, valid for `λ ∈ (0,2)`. Two things to
verify. First, that it is a genuine certificate — it is, for any *fixed* `λ` chosen before the data;
this is exactly Catoni's localized bound. Second, that it is convex in `Q` for fixed `λ`: the numerator
`L̂(Q)` is linear in `Q` (an expectation of a per-hypothesis loss under `Q`), `KL(Q‖P)` is convex in `Q`,
and the denominators `1−λ/2` and `nλ(1−λ/2)` are positive constants once `λ ∈ (0,2)` is fixed — so the
right-hand side is a positive-weighted sum of a linear and a convex functional, convex in `Q`. Good:
this is differentiable and well-posed as a training objective, and the complexity term
`(KL+Λ)/(nλ(1−λ/2))` is *linear* in KL, which is the proportional penalty the additive bound lacked.

Now I have to be honest about whether the linear form is actually a tighter *number*, because it is
tempting to assume "linear penalty" means "tighter bound" and that is not automatic. The naive read is
that near `L̂ = 0` the Catoni bound is order `KL/n` while the additive is `√(KL/(2n))`, and since `√u > u`
for small `u`, the additive square root looks smaller. But that read cheats by dropping the `Λ` constant
from the additive root while keeping it in Catoni's numerator; if I keep `Λ` in both, the comparison
flips. Write `w = (KL+Λ)/n`. The additive value is `√(w/2)`; the Catoni value at `L̂ = 0` is
`w/(λ(1−λ/2))`. Setting them equal, `√(w/2) = w/(λ(1−λ/2))` gives `λ(1−λ/2) = √(2w)`, i.e. a crossover at
`w* = ½[λ(1−λ/2)]²`. At the natural starting `λ = 0.5` that is `w* = 0.0703`, meaning `KL+Λ ≈ 2100`; at
`λ = 1` it is `w* = 0.125`, `KL ≈ 3740`. For every KL I realistically expect — tens to a few hundred
nats — `w` sits far below `w*`, so the Catoni *value* is actually the smaller one. Let me sharpen that
with the optimal-`λ` value, since a well-tuned `λ` is what I am promising to find. At `L̂ = 0` the
optimum is `λ* = 1`, and the bound there is `2(KL+Λ)/n`; at `KL = 100` that is `2·109.5/30000 ≈ 0.0073`,
against the additive `√(109.5/60000) ≈ 0.0427` — the Catoni value is roughly six times tighter. So the
linear form is not weaker as a raw bound; if anything it is numerically tighter in this regime, which is
exactly what makes it tempting. The reason I cannot bank
that as a win is subtler, and it is the thing I have to keep straight through this whole trajectory: the
number I *report* is never the training bound. The reported certificate is `inv_kl(L̂_01, (KL+Λ)/n)`, the
kl-inversion of the parent, and that functional is identical for every bound I could train against — it
depends only on the empirical 0-1 risk and the KL that training converges to, and it is monotone
increasing in KL. So Catoni "winning the value comparison" buys nothing directly; what decides the
certificate is which KL the training dynamics settle at. The linear-in-KL objective is a claim about
those dynamics, not about the arithmetic of the bound at a fixed operating point, and it lives or dies on
the `λ` it finds.

That is the crux, `λ`. Catoni's bound holds for one fixed `λ`. If I want the trade-off to suit the data
— and I do, because the right balance between fitting and staying near the prior depends on how much the
posterior must move — I have to tune `λ`. The textbook-clean way is a uniform-in-`λ` bound that holds
for all `λ` at once via a deterministic AM-GM / union-over-a-grid device, which would let me optimize `λ`
continuously and read off its optimum in closed form. Let me actually consider building that, because it
is the principled option. The closed-form `λ*` for fixed `L̂, KL` minimizes
`g(λ) = L̂/(1−λ/2) + (KL+Λ)/(nλ(1−λ/2))`; differentiating and clearing the positive `(1−λ/2)²`, the
stationarity condition is `(L̂/2)λ² + bλ − b = 0` with `b = (KL+Λ)/n`, so `λ* = [−b + √(b(b+2L̂))]/L̂`.
That is clean, but building it into *this* task means I need either the Gibbs measure it localizes around
or a bound that already carries the union over `λ`, and the edit surface does not give me that. The
contract is just `compute_bound`/`train_step`/`compute_risk_certificate` over a fixed substrate whose
`model(x, sample=…)` and `get_total_kl` hand me a diagonal-Gaussian posterior with an analytic KL — not
an explicit Gibbs posterior I can normalize, and not a `λ`-head wired into the loop. So the cleanest
thing that fits the contract is to carry `λ` as my *own* learnable scalar inside the `BoundOptimizer`,
give it its own optimizer, and update it by gradient descent on the same bound — a numerical alternating
minimization rather than the closed-form one. That is the design I will land, and I should be clear-eyed
that swapping the closed-form `λ*` for an SGD scalar is exactly where the risk enters.

Now the mechanics, and there is a real trap here that I need to reason through, because it is the
difference between this baseline working and failing. The outer SGD loop in the substrate constructs
*one* optimizer over `model.parameters()` and steps it after `train_step` returns the loss. My `λ` is
not a model parameter — it lives in the `BoundOptimizer` — so the loop's `optimizer.step()` will never
touch it. If I do nothing special, `λ` stays frozen at its initialization for the entire run, and the
"tuning" is a fiction. So I must give `λ` its *own* optimizer and step it myself inside `train_step`. And
I must step it on a *detached* copy of the loss: the posterior's gradient flows through the loop's
`optimizer.step()` using the bound I return with its live `nll`/`kl` graph, so if I also backprop the
`λ`-update through that same live graph I would either double-count the posterior gradient or corrupt it.
The right structure is: compute the bound for the posterior (the value I return, live graph), then
*separately* form a detached scalar `L̂.detach()/(1−λ/2) + (KL.detach()+Λ)/(nλ(1−λ/2))`, backprop *that*
into `λ` only, and step `λ`'s optimizer. The detach makes the `λ`-step see `L̂` and `KL` as constants —
which is correct, because at the alternating-minimization `λ`-substep the posterior is held fixed. If I
skipped the detach, `λ.backward()` would deposit gradients on every `μ` and `ρ` in the model as a side
effect, and then the loop's `optimizer.step()` would apply a posterior update assembled partly from that
`λ`-flavored objective on top of the correct update from the returned bound — the two would fight and the
posterior would descend a direction that is neither. There is also a device subtlety worth pinning down:
`λ` is created as a bare CPU tensor at construction, but the model runs on the training device, so on the
first `train_step` I move `λ` onto the device, re-mark it `requires_grad_`, and rebuild its optimizer;
otherwise the first `λ.backward()` mixes a CPU scalar with device tensors and the update silently fails.

The second trap is the range of `λ`. The bound has a singularity at `λ = 2` (the `1−λ/2` denominator hits
zero) and degenerates at `λ = 0` (the `1/λ` blows up). A free SGD scalar can wander anywhere, and if it
ever reaches `λ ≥ 2` the denominator flips sign and the bound becomes negative and meaningless — the
optimizer would then happily drive `λ` to the singularity to "minimize" a bound that is no longer a
bound. So I clamp `λ` into a safe open interval, `(0.01, 1.99)`, every time I read it. Both ends are
pathological in their own way, which the numbers make vivid: the complexity prefactor `1/(nλ(1−λ/2))` is
`3.35·10⁻³` at `λ = 0.01` (the `1/λ` blowing up) and *also* `3.35·10⁻³` at `λ = 1.99` (the `1−λ/2`
vanishing), against `6.67·10⁻⁵` at the middle `λ = 1`; and the empirical weight `1/(1−λ/2)` reaches `200`
at `λ = 1.99`. The clamp keeps the bound finite and positive at both walls; I initialize at `0.5`, well
inside, not the elegant sigmoid-into-`[1/√n,1]` reparameterization, just a hard clamp on a raw learnable
scalar.

Let me reason about *where* I expect this to land relative to the default, because that is the prediction
I am making, and I can actually compute the failure mode rather than assert it. The complexity prefactor
`P(λ) = 1/(nλ(1−λ/2))` is governed by the denominator `λ(1−λ/2) = λ − λ²/2`, which is maximized at
`λ = 1` (its derivative `1 − λ` vanishes there), value `0.5`. So `P(λ)` is *minimized* at `λ = 1`, where
it equals `2/n = 6.67·10⁻⁵`. Now combine that with the `λ*` I derived: as `L̂ → 0`, the stationarity
condition `(L̂/2)λ² + bλ − b = 0` sends `λ* → 1`. So a well-fitting posterior drives `λ` toward exactly
the value where the KL penalty is *weakest possible* — and `6.67·10⁻⁵` is weaker than the additive
bound's own marginal KL cost at the same operating point (`1.95·10⁻⁴` at `KL = 100`). The free `λ`,
seeking its optimum, threatens to relax the linear penalty below even the square-root penalty I was
trying to improve on. Worse, the effect is self-reinforcing: as `KL` grows, `b = (KL+Λ)/n` grows, which
pushes `λ*` *up* toward `1`, which lowers `P(λ)` further, which lets `KL` grow more — a positive feedback
that saturates only when the fitting gradient balances the `2/n` penalty, i.e. at a large equilibrium KL.
There is a competing pull I should name honestly: because I feed the *unrescaled* NLL, `L̂` can be large
(order `0.1`–`0.2` rather than the `~0.01` a `[0,1]` rescaling would give), and a large `L̂` pulls `λ*`
*down* — plugging `L̂ = 0.15, KL = 100` into the closed form gives `λ* ≈ 0.20`, on the strong-penalty
side. So the joint landscape has two competing basins — small `λ` where the empirical term is cheapest
and the KL penalty strong, versus `λ → 1` where the KL penalty collapses — and nothing pins the SGD
scalar (lr `0.01`, detached, no analytic anchor) to either. That is exactly the configuration that
produces seed-dependent, potentially bimodal outcomes: whichever basin a run drifts into early, the
feedback loop entrenches.

The final certificate is separate from the training objective and I should be deliberate about it. I
train against the Catoni functional, but for the *reported* number I want the tightest valid bound on the
learned posterior, which is PAC-Bayes-kl itself, inverted. So `compute_risk_certificate` MC-samples the
stochastic predictor's empirical 0-1 risk on the bound set via `compute_01_risk`, reads off the KL from
one forward pass, forms `c = (KL + log(2√n/δ))/n`, and calls `inv_kl(emp_risk_01, c)` for the reported
`risk_certificate`. Let me trace what that inversion does to a large KL so I can see the stakes, using a
plausible operating point — say the posterior fits to `L̂_01 ≈ 0.02` (a few percent, which the
data-dependent prior guarantees). If `KL` lands at `500`, then `c = (500+9.54)/30000 ≈ 0.0170`, and
`inv_kl(0.02, 0.0170) ≈ 0.057` — the certificate is inflated to nearly three times the empirical risk,
entirely by KL. If instead `KL` were held to `80`, `c ≈ 0.0030` and `inv_kl(0.02, 0.0030) ≈ 0.033`; at
`KL = 0` the floor is `≈ 0.024`. So the inversion is exquisitely sensitive to KL in exactly the range my
failure-mode analysis says the free `λ` can drive it. For this baseline I keep the inversion simple — a
single `inv_kl` on the raw MC estimate, no inner Monte-Carlo correction for posterior-sampling error,
matching the unrescaled, uncorrected style of the scaffold default. I also report the Catoni `ce_bound`
by feeding the empirical NLL and KL through `compute_bound`, and the converged `λ`, as diagnostics. The
empirical-risk surrogate during training is the plain clamped NLL, `F.nll_loss` on `log_softmax` floored
at `log(pmin)` — *not* rescaled by `1/log(1/pmin)`. I note that choice deliberately: the rescaling that
maps NLL into `[0,1]` is what calibrates the empirical-risk term against the KL term, and leaving it out
means the Catoni objective is fed an empirical term that can exceed 1 and, as I just computed, pulls `λ*`
around; I am holding that rescaling in reserve.

So here is my falsifiable expectation for this rung. The Catoni/lambda bound, with a clamped free `λ`
and an unrescaled NLL surrogate, is a legitimate convex-in-`Q` certificate and a reasonable first thing
to try, and it should produce a non-vacuous `risk_certificate` well below 1 — the data-dependent prior
puts it in the few-percent range, not the vacuous range. But I predict it will be the *loosest* of the
bound formulations I can build on this surface, dominated by a KL that runs large and, because the free
`λ` has no analytic pin and two competing basins, is *unstable across seeds* — I would not be surprised
to see a bimodal `kl_divergence` where some seeds sit modestly and others blow up into the hundreds. The
certificate must track it, since `inv_kl(L̂_01, (KL+Λ)/n)` is monotone in KL: a KL of a few hundred over
`n ≈ 30000` gives `c` on the order of `0.02` and inflates the inverted certificate well above the
empirical 0-1 risk, exactly as the trace above shows. Concretely I expect the FCN — three 600-wide hidden
layers, the largest count of probabilistic weights, hence the most room for KL to accumulate — to show
the worst certificate and by far the largest, most seed-variable KL, with the CNN settings tighter simply
because they have fewer probabilistic weights. Let me put a number on "most room." The FCN carries
`784·600 + 600·600 + 600·600 + 600·10 ≈ 1.196` million Gaussian weights (plus ~1810 biases), each with
its own `μ` and `ρ` and its own additive contribution to `get_total_kl`. The KL is a *sum* of per-weight
terms `log(σ_p/σ_q) + (σ_q² + (μ_q−μ_p)²)/(2σ_p²) − ½`, so with a million-plus independent summands, even
a minuscule average per-weight divergence aggregates: an average of `5·10⁻⁴` nats per weight already puts
the FCN near `600` nats, whereas the same per-weight average over a convolutional model — whose kernels
*share* parameters and number in the tens of thousands, not the millions — lands one to two orders of
magnitude lower. So the architecture with the most independent probabilistic weights is precisely the one
where a weak KL penalty and a runaway `λ` have the most surface to inflate, which is why I expect the FCN
to be the worst case and the CNNs to look comparatively tame regardless of what `λ` does. If instead the KL stays small and stable (tens of nats
across all seeds) and the certificate comes in near the additive default, then my diagnosis of the
free-`λ` failure mode is wrong and the instability lives somewhere in the shared substrate, and I will
look elsewhere; the `kl_divergence` column is what will tell me. The full scaffold module —
`compute_bound`, the dual-optimizer `train_step`, and the single-`inv_kl` `compute_risk_certificate` —
is in the answer.
