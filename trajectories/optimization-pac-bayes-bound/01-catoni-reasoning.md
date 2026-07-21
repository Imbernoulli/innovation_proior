I start from the default scaffold, which fills `compute_bound` with the McAllester/Maurer additive
square-root form `L̂ + √((KL + log(2√n/δ))/(2n))`, and I want to ask whether that is the right shape
to be *training against* in the first place. The whole certificate, in this substrate, is decided by
the bound functional: the stochastic layers, the data-dependent prior split, and the SGD loop are all
fixed, so the only lever I have is which inequality I push the posterior down. When the posterior fits
the bound split — which it will, because I initialize it at the ERM minimizer of the prior — the
empirical risk `L̂(Q)` goes toward zero, and the bound collapses to `√((KL + log(2√n/δ))/(2n))`. The
complexity enters through a *square root*, and "square root" alone is not an argument, so I put numbers
on the marginal cost. The bound split is half of the 60k training set, so `n ≈ 30000`, and
`log(2√n/δ) = log(2·173.2/0.025) ≈ 9.54`; call that constant `Λ`. The marginal cost of one more nat of
divergence is `d/dKL √((KL+Λ)/(2n)) = 1/(2√(2n(KL+Λ)))`. At `KL = 100` that is `≈ 1.95·10⁻⁴`; at
`KL = 500` it has fallen to `≈ 0.90·10⁻⁴`. So over a hundred to five hundred nats the marginal penalty
*halves*: the additive objective only weakly penalizes a posterior that drifts from the prior, because
the marginal cost of one more nat shrinks as KL accumulates. That is the wrong incentive — I want the
bound to fight KL growth proportionally, not sublinearly.

So I look for a bound whose complexity term is *linear* in KL near the operating point and which I can
still differentiate and minimize directly. The standard family that buys this is the localized,
trade-off-parameter bounds, and Catoni's is the canonical one. I derive it from the same root the
default came from, the PAC-Bayes-kl inequality, so I know exactly what I am relaxing. The
change-of-measure inequality `E_Q[φ] ≤ KL(Q‖P) + log E_P[e^φ]` — nothing but `KL ≥ 0` applied to the gap
between `Q` and the exponential tilt of `P` by `φ` — transports a per-hypothesis exponential moment
computed under the *data-free* `P` onto the data-dependent `Q`. Choosing `φ(h) = n·kl(L̂(h)‖L(h))`, the
binary KL between empirical and true Bernoulli risk, controlling its moment with Maurer's sharp bound
`E_S[e^{n·kl(L̂‖L)}] ≤ 2√n` (for `n ≥ 8`), then Markov and Jensen, gives with probability `1−δ`,
simultaneously for all `Q`, `kl(L̂(Q)‖L(Q)) ≤ (KL(Q‖P) + log(2√n/δ))/n`. This is the parent, and its
right-hand side is exactly the `Λ/n`-plus-`KL/n` budget every one of these bounds spends. The `2√n` is
Maurer's halving of the cruder `log(2n)`, which is what makes `Λ ≈ 9.54` rather than a nat or so larger
— small, but it sets the certificate's floor when `KL → 0`, so it is worth carrying exactly. The
additive default is the Pinsker relaxation `kl(p‖q) ≥ 2(p−q)²` of this parent — a symmetric parabola in
its first argument, loose precisely when the true risk is small, which is my regime. Catoni relaxes the
*same* parent through a tilt engineered to give a linear-in-`L̂`, linear-in-KL trade-off, and the price
is a free parameter `λ`.

The parent admits more than one descendant, so I reject the alternatives for stated reasons. Keeping the
additive default under-defends KL, as the halving marginal cost just showed. Training directly against
the *inverted* parent `inv_kl(L̂, (KL+Λ)/n)` is tempting — it is literally the tightest certificate and
what I report anyway — but the obstruction is mechanical: `inv_kl` is defined by bisection (the largest
`p` with `kl(L̂‖p) ≤ c`), and that root is not a clean differentiable function to hand the substrate's
autograd; the `train_step` contract wants a single differentiable scalar assembled from `nll` and
`get_total_kl`, not backprop through a bisection loop. So the inverted parent is the right *evaluation*
but the wrong *training* object. That leaves the two explicit relaxations of the parent's left-hand
`kl`: the symmetric Pinsker parabola (the additive default) and Catoni's tilt (localized, linear-in-KL).
Since the whole reason I am moving is to replace a sublinear KL penalty with a proportional one, Catoni
is the option that matches the diagnosis: linear complexity, still explicit and convex, at the cost of
one knob.

The Catoni/lambda functional is `L(Q) ≤ L̂(Q)/(1−λ/2) + (KL(Q‖P) + log(2√n/δ))/(nλ(1−λ/2))`, valid for
`λ ∈ (0,2)`. It is a genuine certificate for any *fixed* `λ` chosen before the data, and for fixed `λ`
it is convex in `Q`: `L̂(Q)` is linear in `Q`, `KL(Q‖P)` is convex, and the denominators `1−λ/2`,
`nλ(1−λ/2)` are positive constants — a positive-weighted sum of a linear and a convex functional. So it
is differentiable and well-posed as a training objective, and its complexity term
`(KL+Λ)/(nλ(1−λ/2))` is linear in KL, the proportional penalty the additive bound lacked.

I should not assume "linear penalty" means "tighter bound," because the naive read cheats. Near `L̂ = 0`
Catoni is order `KL/n` while the additive is `√(KL/(2n))`, and `√u > u` for small `u`, so the additive
root looks smaller — but that drops the `Λ` constant from the root while keeping it in Catoni's
numerator. Keep `Λ` in both: with `w = (KL+Λ)/n`, the additive value is `√(w/2)` and Catoni at `L̂=0` is
`w/(λ(1−λ/2))`; they cross at `w* = ½[λ(1−λ/2)]²`, which at `λ=0.5` is `w*=0.0703`, i.e. `KL+Λ ≈ 2100`.
For every KL I realistically expect — tens to a few hundred nats — `w` sits far below `w*`, so the
Catoni *value* is the smaller one; at the optimal `λ*=1` (`L̂=0`) the bound is `2(KL+Λ)/n`, which at
`KL=100` is `≈ 0.0073` against the additive `≈ 0.0427`, six times tighter. But I cannot bank that, and
this is the thing to keep straight through the whole trajectory: the number I *report* is never the
training bound. The reported certificate is `inv_kl(L̂_01, (KL+Λ)/n)`, identical for every bound I could
train against — it depends only on the empirical 0-1 risk and the KL that training converges to, and it
is monotone increasing in KL. So Catoni "winning the value comparison" buys nothing directly; what
decides the certificate is which KL the training dynamics settle at. The linear-in-KL objective is a
claim about those dynamics, and it lives or dies on the `λ` it finds.

That is the crux. Catoni holds for one fixed `λ`, but I want the trade-off to suit the data, so I have
to tune `λ`. The textbook-clean way is a uniform-in-`λ` bound holding for all `λ` at once, which would
let me read the optimum off in closed form: minimizing `g(λ) = L̂/(1−λ/2) + (KL+Λ)/(nλ(1−λ/2))` gives
stationarity `(L̂/2)λ² + bλ − b = 0` with `b = (KL+Λ)/n`, so `λ* = [−b + √(b(b+2L̂))]/L̂`. Clean, but
building it needs either the Gibbs measure it localizes around or a bound already carrying the union
over `λ`, and the edit surface gives me neither — just `compute_bound`/`train_step`/
`compute_risk_certificate` over a substrate whose `model(x, sample=…)` and `get_total_kl` hand me a
diagonal-Gaussian posterior, not an explicit Gibbs posterior or a `λ`-head wired into the loop. So the
thing that fits the contract is to carry `λ` as my *own* learnable scalar inside `BoundOptimizer`, give
it its own optimizer, and update it by gradient descent on the same bound — numerical alternating
minimization rather than the closed form. That is the design I land, and swapping the closed-form `λ*`
for an SGD scalar is exactly where the risk enters.

The mechanics carry a real trap. The outer SGD loop constructs *one* optimizer over `model.parameters()`
and steps it after `train_step` returns; my `λ` is not a model parameter, so `optimizer.step()` will
never touch it — without special handling `λ` stays frozen at init and the "tuning" is a fiction. So I
give `λ` its own optimizer and step it myself inside `train_step`, on a *detached* copy of the loss. The
posterior's gradient flows through the loop's step using the bound I return with its live `nll`/`kl`
graph; if I also backprop the `λ`-update through that same graph I corrupt the posterior gradient. The
right structure: compute the bound for the posterior (returned, live graph), then separately form a
detached scalar `L̂.detach()/(1−λ/2) + (KL.detach()+Λ)/(nλ(1−λ/2))`, backprop *that* into `λ` only, and
step `λ`'s optimizer. The detach makes the `λ`-step see `L̂` and `KL` as constants — correct, since at
the `λ`-substep the posterior is held fixed; without it, `λ.backward()` would deposit gradients on every
`μ` and `ρ` and the loop's step would then apply a posterior update assembled partly from the
`λ`-objective, fighting the correct update. One device subtlety: `λ` is created as a bare CPU tensor at
construction but the model runs on the training device, so on the first `train_step` I move `λ` onto the
device, re-mark it `requires_grad_`, and rebuild its optimizer; otherwise the first `λ.backward()` mixes
a CPU scalar with device tensors and the update silently fails.

The second trap is `λ`'s range. The bound has a singularity at `λ=2` (`1−λ/2 → 0`) and degenerates at
`λ=0` (`1/λ` blows up). A free SGD scalar can wander anywhere, and if it ever reaches `λ ≥ 2` the
denominator flips sign, the bound goes negative and meaningless, and the optimizer happily drives `λ` to
the singularity to "minimize" a non-bound. So I clamp `λ` into `(0.01, 1.99)` every read. Both ends are
pathological: the prefactor `1/(nλ(1−λ/2))` is `3.35·10⁻³` at `λ=0.01` and again at `λ=1.99`, against
`6.67·10⁻⁵` at the middle `λ=1`; the empirical weight `1/(1−λ/2)` reaches 200 at `λ=1.99`. The clamp
keeps the bound finite and positive at both walls; I initialize at `0.5`, well inside — a hard clamp on
a raw learnable scalar, not the sigmoid reparameterization.

Now where I expect this to land. The prefactor `P(λ) = 1/(nλ(1−λ/2))` is governed by
`λ(1−λ/2) = λ − λ²/2`, maximized at `λ=1`, so `P(λ)` is *minimized* at `λ=1`, where it equals
`2/n = 6.67·10⁻⁵`. And my `λ*` sends `λ* → 1` as `L̂ → 0`. So a well-fitting posterior drives `λ` toward
exactly the value where the KL penalty is weakest — `6.67·10⁻⁵`, weaker even than the additive bound's
marginal cost at the same point (`1.95·10⁻⁴` at `KL=100`). Worse, it is self-reinforcing: as `KL` grows,
`b` grows, pushing `λ*` up toward `1`, lowering `P(λ)`, letting `KL` grow more — positive feedback that
saturates only at a large equilibrium KL. There is a competing pull: because I feed the *unrescaled*
NLL, `L̂` can be large (order `0.1`–`0.2` rather than the `~0.01` a `[0,1]` rescaling gives), and a large
`L̂` pulls `λ*` down — `L̂=0.15, KL=100` gives `λ* ≈ 0.20`, the strong-penalty side. So the joint
landscape has two competing basins — small `λ` (empirical term cheap, KL penalty strong) versus `λ → 1`
(KL penalty collapses) — and nothing pins the detached SGD scalar (lr `0.01`, no analytic anchor) to
either. That is the configuration that produces seed-dependent, potentially bimodal outcomes: whichever
basin a run drifts into early, the feedback entrenches. I expect the effect worst on the FCN, which
carries the most probabilistic weights — `784·600 + 600·600 + 600·600 + 600·10 ≈ 1.2` million Gaussian
weights, each contributing a per-weight term `log(σ_p/σ_q) + (σ_q² + (μ_q−μ_p)²)/(2σ_p²) − ½` to
`get_total_kl`. With a million-plus independent summands even a minuscule average per-weight divergence
aggregates into the many hundreds of nats, whereas a convolutional model, whose kernels *share*
parameters and number in the tens of thousands, lands one to two orders of magnitude lower. So the
architecture with the most independent probabilistic weights is where a weak KL penalty and a runaway
`λ` have the most surface to inflate.

The certificate is separate from the training objective. I train against Catoni, but report the tightest
valid bound on the learned posterior, PAC-Bayes-kl inverted: `compute_risk_certificate` MC-samples the
stochastic predictor's empirical 0-1 risk via `compute_01_risk`, reads the KL from one forward pass,
forms `c = (KL + log(2√n/δ))/n`, and calls `inv_kl(emp_risk_01, c)`. The stakes of a large KL show in
that inversion: at `L̂_01 ≈ 0.02` (a few percent, which the data-dependent prior guarantees), a `KL` of
`500` gives `c ≈ 0.0170` and `inv_kl(0.02, 0.0170) ≈ 0.057` — nearly three times the empirical risk,
entirely from KL; holding `KL` to `80` gives `c ≈ 0.0030` and `inv_kl(0.02, 0.0030) ≈ 0.033`; at `KL=0`
the floor is `≈ 0.024`. So the inversion is exquisitely sensitive to KL in exactly the range the free
`λ` can drive it. I keep the inversion simple — a single `inv_kl` on the raw MC estimate, no inner
Monte-Carlo correction for posterior-sampling error, matching the unrescaled, uncorrected scaffold style
— and report the Catoni `ce_bound` and converged `λ` as diagnostics. The surrogate stays the plain
clamped NLL, `F.nll_loss` on `log_softmax` floored at `log(pmin)`, *not* rescaled by `1/log(1/pmin)`:
that rescaling is what calibrates the empirical-risk term against the KL term, and leaving it out means
the objective is fed an empirical term that can exceed 1 and pulls `λ*` around. I hold that rescaling in
reserve.

So my expectation: the Catoni/lambda bound is a legitimate convex-in-`Q` certificate and should produce
a non-vacuous `risk_certificate` in the few-percent range. But I expect it to be the *loosest* of the
formulations I can build here, dominated by a KL that runs large and, because the free `λ` has no
analytic pin and two competing basins, unstable across seeds — plausibly a bimodal `kl_divergence` with
some seeds modest and others in the hundreds — with the FCN the worst case and by far the largest, most
seed-variable KL, the CNNs comparatively tame. The certificate must track it, since
`inv_kl(L̂_01, (KL+Λ)/n)` is monotone in KL. If instead the KL stays small and stable across all seeds
and the certificate comes in near the additive default, then the free-`λ` diagnosis is wrong and the
instability lives in the shared substrate; the `kl_divergence` column is what will tell me.
