Let me get the problem dead clear before I reach for any machinery. I have a network of stochastic units. Each unit reads some inputs, computes a number from its weights — a firing probability, say — and then *flips a coin* against that number to decide its output. The whole net does this forward, produces some behavior, and the environment hands back a single scalar `r`, a reward judging what the net just did. That `r` gets broadcast to every unit. And that is the entire feedback signal. No target output. No per-unit error. No idea how `r` depends on what the net did — the environment is a black box, and the reward could be noisy or non-differentiable.

So the thing I'd reflexively reach for, backpropagation, is just unavailable. Backprop is the chain rule made into a computation: form a differentiable error between output and target, push partial derivatives backward through deterministic units, descend. It needs a target to make an error from, and it needs the path from weight to error to be a differentiable chain. Here the path from a weight to `r` runs straight through a coin flip and then through an unknown environment. There is no error to send back, and even if there were, the randomness sits *inside* the units, breaking the chain. The chain rule has nothing to bite on.

What do I actually want to maximize? Not `r` on any single trial — that's a roll of the dice. I want the *expected* reward under the net's own randomness, `E{r|W}`, as a function of the weight matrix `W`. Under stationarity — the environment picks inputs from a fixed distribution, picks rewards for an input–output pair from a fixed distribution — this `E{r|W}` is a perfectly well-defined deterministic function of `W`. It's just that I can't write it down, because writing it down needs the environment's response, which is the one thing I don't have. So the objective is a hill `E{r|W}` whose shape is hidden, and I want to climb it from samples, locally, with each unit using only its own input, its own output, and the broadcast `r`.

Let me stare at what climbing actually requires. To do gradient ascent I'd need `∂E{r|W}/∂w_ij` for every weight. Write the expectation out for a single unit `i` with input `x^i`. Its output `y_i` is drawn from `g_i(ξ, w^i, x^i) = Pr{y_i = ξ | w^i, x^i}`, a probability mass function controlled by the unit's weights. Condition on the output value:

  E{r | W, x^i} = Σ_ξ E{r | W, x^i, y_i = ξ} · Pr{y_i = ξ | w^i, x^i}
               = Σ_ξ E{r | W, x^i, y_i = ξ} · g_i(ξ, w^i, x^i).

Now differentiate with respect to `w_ij`. Here's the thing worth pausing on: once I've *fixed* the output value `y_i = ξ`, the weight `w_ij` no longer has any say in what reward comes back — `w_ij` only ever influenced `r` *through* the output it produced, and I've nailed that output down. So `E{r | W, x^i, y_i = ξ}` does not depend on `w_ij` at all. Every bit of the `w_ij`-dependence lives in the `g_i` factor. So

  ∂E{r | W, x^i}/∂w_ij = Σ_ξ E{r | W, x^i, y_i = ξ} · ∂g_i/∂w_ij.

Good — that's exact. But it's not yet something I can sample, because `∂g_i/∂w_ij` is a derivative of a probability and the update I take at trial time has to be built from the *one* output I actually drew, not a sum over all possible outputs weighted by their derivatives. I need to turn that sum-over-outcomes into an expectation, so a single sample is an unbiased stand-in for it.

The obstacle is that the parameter I'm differentiating with respect to is buried *in the sampling distribution itself*, not in some integrand sitting on top of a fixed distribution. If it were `∂/∂w E_p[f(y)]` with `p` fixed and `f` carrying the weight, I'd just differentiate `f`. But it's the opposite: `f` here is the unknown reward, and `w` lives in `p = g_i`. I can't differentiate the reward — I don't have it.

So how do I move a derivative of a probability into the *inside* of an expectation taken over that same probability? Multiply and divide by the probability:

  ∂g_i/∂w_ij = g_i · (1/g_i) · ∂g_i/∂w_ij = g_i · ∂(ln g_i)/∂w_ij.

There it is. The factor `(1/g_i) ∂g_i/∂w_ij` is exactly the derivative of the *log* of the probability. So

  ∂E{r | W, x^i}/∂w_ij = Σ_ξ E{r | W, x^i, y_i = ξ} · g_i(ξ) · ∂ln g_i/∂w_ij
                       = E{ r · ∂ln g_i/∂w_ij | W, x^i }.

Read the right-hand side: it's an *expectation*, over the unit's own output distribution, of the reward times the gradient of the log-probability of the output that occurred. Which means: draw one output `y_i`, observe `r`, and form `r · ∂ln g_i/∂w_ij` evaluated at the output you got — that single quantity is an *unbiased estimate of the gradient of expected reward* with respect to `w_ij`. No model of the environment. No differentiating `r`. No chain rule through the coin flip. The reward just sits there as a scalar multiplier; all the differentiating happens on the log-probability of my own action, which I know in closed form.

I want to make sure I believe *why* the log is the magic factor and not an accident. The reason is that the `g_i` produced by the multiply-and-divide is precisely the weight a sample carries: when I sample `y_i ~ g_i`, outcome `ξ` shows up with frequency `g_i(ξ)`. So `E[ r · ∂ln g_i/∂w_ij ] = Σ_ξ g_i(ξ) · r(ξ) · (1/g_i(ξ)) ∂g_i/∂w_ij = Σ_ξ r(ξ) ∂g_i/∂w_ij` — the `g_i` cancels against the sampling frequency, leaving the bare `Σ r ∂g_i/∂w_ij`, which is the gradient I was after. The log derivative is exactly the thing that, multiplied by the sampling frequency, reconstitutes `∂g_i/∂w_ij`. It's not a trick pulled from a hat; it's the unique factor that makes a frequency-weighted average equal a probability-derivative.

This `∂ln g_i/∂w_ij` deserves a name of its own, because it's the only place the unit's *own* structure enters — it's the sensitivity of the log-likelihood of the chosen output to the weight, a per-trial quantity the unit can always compute locally. Call it the *characteristic eligibility* `e_ij` of the weight. It tells the unit, "given the output you happened to produce, here's the direction in weight space that would have made that output more likely." Multiply it by the reward and you have a learning signal: make likely-what-paid-off.

Now a worry. The reward `r` in `r · e_ij` is sitting there raw. If `r` is always large and positive — say payoffs all live in `[5, 10]` — then every update reinforces whatever just happened, strongly, regardless of whether it was good *relative to the alternatives*. The estimator is still unbiased, but its variance is awful and its short-run behavior is a near-blind random walk biased only weakly toward better actions. I've seen this exact pathology described for the two-action reward-inaction automaton `L_{R-I}` (Narendra & Thathatchar): the expected motion is always toward the better action, yet because the walk only ever pushes in the direction of whatever was sampled, there's a nonzero probability of being absorbed at the *inferior* action — like a biased random walk on the integers with absorbing barriers, which can still get absorbed at the wrong end. The cure people use empirically is reinforcement comparison (Sutton): keep a running prediction `r̄` of upcoming reward and learn from the *centered* signal `r − r̄`, so an action only gets reinforced if it beat expectation. That clearly helps. But why am I *allowed* to subtract `r̄`? Doesn't tampering with `r` corrupt the gradient I just proved I was estimating?

Let me check. Suppose I subtract some baseline `b` and form `(r − b) · e_ij`. The expectation splits:

  E{ (r − b) e_ij } = E{ r e_ij } − E{ b e_ij }.

The first term is the true gradient, established. The second term I need to vanish. Take `b` to be conditionally independent of the output `y_i` given the unit's weights and input — `r̄`, a prediction made from *past* experience, qualifies, as long as it doesn't peek at the current `y_i` or current `r`. Then

  E{ b e_ij | W, x^i } = E{ b | W, x^i } · Σ_ξ g_i(ξ) · (1/g_i(ξ)) ∂g_i/∂w_ij
                       = E{ b | W, x^i } · Σ_ξ ∂g_i/∂w_ij.

And now look at that last sum:

  Σ_ξ ∂g_i/∂w_ij = ∂/∂w_ij ( Σ_ξ g_i(ξ) ) = ∂/∂w_ij ( 1 ) = 0.

Because `g_i` is a probability distribution — its outputs always sum to one, no matter the weights — its total has derivative zero. So the *expected eligibility is exactly zero*, and therefore `E{ b e_ij } = 0` for *any* admissible baseline. Subtracting a baseline that doesn't depend on the current outcome changes the estimator's variance but leaves its mean — the true gradient — untouched. That's why reinforcement comparison was never cheating: `r̄` is free precisely because the eligibility integrates to zero. The baseline is a knob I can set to anything (independent of `y_i`) to shrink variance, and the natural setting is `b ≈ E{r}`, the running mean, so the multiplier `(r − b)` is the *surprise* in the reward rather than its raw magnitude.

I want to underline the identity that did all the work, because it's the same fact twice. The eligibility has mean zero because `Σ g = 1` is constant in the weights: `Σ ∂g = ∂Σ g = ∂(1) = 0`. That single fact — the probabilities sum to one always, so the gradient of their sum is zero — is simultaneously what makes the baseline harmless and what guarantees the un-baselined estimator is centered on the true gradient. The whole method hangs on `∇∫ = ∇1 = 0`.

So I have a candidate update for one weight: `Δw_ij = α_ij (r − b_ij) e_ij`, with `e_ij = ∂ln g_i/∂w_ij` the characteristic eligibility, `b_ij` any baseline that doesn't depend on the current output, and `α_ij ≥ 0` a rate factor that may depend on the unit's weights or the trial count but not, by itself, flip the sign. Reward increment equals a nonnegative factor times the offset reinforcement times the characteristic eligibility — `REINFORCE`. Any rule of this exact shape I'll call a REINFORCE algorithm, and now I want to know what's provably true about the *whole network* running it, not just one weight.

The per-weight fact was `E{Δw_ij | W, x^i} = α_ij ∂E{r|W, x^i}/∂w_ij` — the average update points along the input-conditioned gradient. But I want the unconditional statement, averaged over the environment's choice of input too, because that's the surface I actually care about. Condition on the input pattern:

  E{r | W} = Σ_x E{r | W, x^i = x} · Pr{x^i = x | W}.

Differentiate by `w_ij`. The weight `w_ij` sits *downstream* of whatever computation produced this unit's input `x^i` — the input arrived before this unit acted — so `Pr{x^i = x | W}` does not depend on `w_ij`. Hence

  ∂E{r|W}/∂w_ij = Σ_x ∂E{r | W, x^i = x}/∂w_ij · Pr{x^i = x | W}.

Now average the update the same way: `E{Δw_ij | W} = Σ_x E{Δw_ij | W, x^i = x} Pr{x^i = x | W}`, substitute the per-input result `E{Δw_ij | W, x^i=x} = α_ij ∂E{r|W,x^i=x}/∂w_ij`, pull the constant `α_ij` out (it doesn't depend on the input), and match term-for-term:

  E{Δw_ij | W} = α_ij Σ_x ∂E{r|W,x^i=x}/∂w_ij Pr{x^i=x|W} = α_ij ∂E{r|W}/∂w_ij.

The reason this step matters is that the messy quantity `Pr{x^i = x | W}` — which in a deep net could be a horrible function of all the other weights — drops out of the *derivative* entirely, just because the input is upstream. So the average update of a single weight equals the rate factor times the true unconditional gradient component, with none of the network's internal complexity intruding.

Stack the weights into vectors and take the inner product of the average update with the true gradient:

  E{ΔW | W} · ∇_W E{r|W} = Σ_{ij} E{Δw_ij | W} · ∂E{r|W}/∂w_ij
                         = Σ_{ij} α_ij ( ∂E{r|W}/∂w_ij )².

Every term is a nonnegative `α_ij` times a square. So the inner product is `≥ 0`: the average weight change *always* lies in a direction that does not decrease expected reward. If every `α_ij > 0`, the inner product is zero only when the gradient itself is zero — i.e. the algorithm sits still in expectation only at a stationary point of `E{r|W}`. And if all the rate factors are a common constant `α`, the average update is *literally* `α ∇_W E{r|W}` — exact gradient ascent, in expectation, on a hill I could never write down. That is the whole prize: a local, sample-driven, model-free rule whose mean step is the gradient of expected reinforcement.

Let me now see what falls out for the actual units I have, because if known rules are special cases then this theorem covers them for free, and the cases tell me whether I've got the algebra right.

Start with the barest unit: a single Bernoulli with parameter `p = Pr{y=1}`, no input — a two-action automaton. Its mass function is `g(y, p) = 1−p` if `y=0`, `p` if `y=1`. The eligibility with respect to `p`:

  ∂ln g/∂p = −1/(1−p) if `y=0`,   = +1/p if `y=1`.

Let me fold both branches into one expression. When `y=1`: `1/p = (1−p)/(p(1−p)) = (y−p)/(p(1−p))` since `y−p = 1−p`. When `y=0`: `−1/(1−p) = −p/(p(1−p)) = (y−p)/(p(1−p))` since `y−p = −p`. Both branches collapse to

  ∂ln g/∂p = (y − p) / ( p(1−p) ).

Now pick the baseline `b=0` and the rate factor `α = ρ p(1−p)` with `0 < ρ < 1` — choosing `α` to cancel the `p(1−p)` in the denominator. Then `Δp = ρ p(1−p) · r · (y−p)/(p(1−p)) = ρ r (y−p)`. And if the reward is binary, `r ∈ {0,1}`, this `Δp = ρ r (y−p)` is exactly the two-action linear reward-inaction `L_{R-I}` automaton — it only moves on reward, and toward the action taken. So `L_{R-I}` is a REINFORCE algorithm, and Theorem 1 hands me, for free, the statement that its expected motion is along the gradient of expected reward — which is precisely the "biased toward the better action yet can be absorbed at the worse one" picture I noted earlier, now explained: the mean is right, but nothing here controls the variance.

Now give the Bernoulli unit an input — the semilinear unit, `p = f(s)`, `s = Σ_j w_ij x_j`. The eligibility of a weight `w_ij` is a chain-rule walk: `∂ln g/∂w_ij = (∂ln g/∂p)(dp/ds)(ds/∂w_ij)`. I have `∂ln g/∂p = (y−p)/(p(1−p))`; `dp/ds = f'(s)`; `ds/∂w_ij = x_j`. So

  ∂ln g/∂w_ij = (y − p)/(p(1−p)) · f'(s) · x_j.

If `f` is the logistic, `f'(s) = p(1−p)`, and the `p(1−p)` cancels beautifully:

  ∂ln g/∂w_ij = (y − p) x_j.

So for a whole network of Bernoulli-logistic units with constant rate `α` and zero baseline, `Δw_ij = α r (y_i − p_i) x_j`. Stare at that. It's the associative reward-inaction `A_{R-I}` rule — the `λ=0` case of Barto & Anandan's associative reward-penalty,

  Δw_ij = α [ r (y_i − p_i) + λ (1 − r)(1 − y_i − p_i) ] x_j.

So `A_{R-I}`, a rule that had been written down directly and hand-justified for Bernoulli-logistic units, is *also* a REINFORCE algorithm, and Theorem 1 now tells me it follows the expected-reward gradient — a property it was never derived to have. That's the moment it clicks that the eligibility-times-reward form isn't one more heuristic in the pile; it's the principle the pile was approximating. (The full `A_{R-P}` with the penalty term `λ(1−r)(1−y−p)` is *not* of REINFORCE form — that extra term doesn't arise from any eligibility — so the theorem doesn't cover it, which is itself informative: the penalty term is a separate design choice, not a gradient-follower.)

And reinforcement comparison drops in with no extra work: take `b = r̄`, the running prediction, with `r̄(t) = γ r(t−1) + (1−γ) r̄(t−1)`. As long as `r̄` is built from past trials and doesn't look at the current `y_i` or `r`, it's an admissible baseline, the unbiasedness proof goes through unchanged, and the rule for Bernoulli-logistic units is `Δw_ij = α (r − r̄)(y_i − p_i) x_j`. Now I understand why reinforcement comparison helped: it's the variance-reducing baseline, free because the eligibility integrates to zero, centering the multiplier on the reward's surprise.

Let me push to the continuous case, because that's where the framework earns its generality. A Gaussian unit computes a mean `μ` and a standard deviation `σ` and draws `y ~ N(μ, σ²)`:

  g(y, μ, σ) = 1/((2π)^{1/2} σ) · exp( −(y−μ)² / (2σ²) ).

I just need the eligibilities of `μ` and `σ`. Take the log: `ln g = −½ ln(2π) − ln σ − (y−μ)²/(2σ²)`. Then

  ∂ln g/∂μ = (y − μ)/σ²,

and

  ∂ln g/∂σ = −1/σ + (y−μ)²/σ³ = ( (y−μ)² − σ² ) / σ³.

(Both checked by hand: differentiating `−(y−μ)²/(2σ²)` in `μ` gives `(y−μ)/σ²`; in `σ` the `−lnσ` gives `−1/σ` and the exponent term gives `+(y−μ)²/σ³`.) So a REINFORCE rule for the Gaussian unit is

  Δμ = α_μ (r − b_μ) (y − μ)/σ²,    Δσ = α_σ (r − b_σ) ( (y−μ)² − σ² )/σ³.

Read what these *do*, because it tells me the framework is buying something single-parameter units can't. The `μ` update moves the mean toward `y` when the reward (relative to baseline) was good and away when bad — sensible. The `σ` update is the interesting one: when a rewarded sample landed *close* to the mean (`|y−μ| < σ`), `(y−μ)²−σ² < 0`, so `σ` *shrinks* — narrow the search, the good stuff is near here; when a rewarded sample landed *far* (`|y−μ| > σ`), `σ` *grows* — widen the search, good stuff is out there. So `σ` is a controllable exploration width: a multiparameter distribution lets the unit decide *where* to explore (`μ`) and *how widely* (`σ`) separately, which a one-number Bernoulli unit simply cannot do. A reasonable choice is `α_μ = α_σ = α σ²`, which tidies the `σ²` denominators, with `b_μ = b_σ` a reinforcement-comparison baseline. (One caveat I should flag to myself: nothing stops the `σ` update from driving `σ` negative; in practice the tails get truncated, or better, adapt `λ = ln σ` instead of `σ` so the parameter stays positive.)

Now I notice something. The mean-eligibility of the Bernoulli was `(y−p)/(p(1−p))`, and `p(1−p)` is exactly the *variance* of a Bernoulli. The mean-eligibility of the Gaussian is `(y−μ)/σ²`, and `σ²` is exactly its variance. Same shape: `∂ln g/∂(\text{mean}) = (y − \text{mean})/\text{variance}`. That can't be a coincidence between two distributions. Let me see how wide it goes. Suppose the mass (or density) function can be written

  g(y, μ, θ₂,…,θ_k) = exp[ Q(μ, θ₂,…) y + D(μ, θ₂,…) + S(y) ],

where `μ` is the mean — an exponential family, linear in `y` inside the exponent. Then `ln g = Q y + D + S(y)`, and differentiating in `μ`:

  ∂ln g/∂μ = (∂Q/∂μ) y + (∂D/∂μ) =: α y + β,

writing `α = ∂Q/∂μ`, `β = ∂D/∂μ` (a different `α` from the learning rate — local shorthand). I claim `∂ln g/∂μ = (y−μ)/σ²` always. Two facts pin down `α` and `β`. First, the eligibility has mean zero — the same `Σ ∂g = ∂Σ g = ∂1 = 0` as before — so `E[αy + β] = α μ + β = 0`. Second, take the `(y−μ)`-weighted average of the eligibility:

  Σ_y (y − μ) g · ∂ln g/∂μ = Σ_y (y−μ) ∂g/∂μ = ∂/∂μ Σ_y (y−μ) g + Σ_y g
  — let me do that more carefully. `Σ_y (y−μ) ∂g/∂μ = Σ_y y ∂g/∂μ − μ Σ_y ∂g/∂μ = ∂/∂μ(Σ_y y g) − μ·0 = ∂μ/∂μ = 1`,

since `Σ_y y g = μ` by definition of the mean and its `μ`-derivative is `1`. On the other hand, computing the same weighted average directly from `∂ln g/∂μ = αy + β`:

  Σ_y (y−μ)(αy + β) g = α Σ_y (y−μ)² g + (αμ+β) Σ_y (y−μ) g = α σ² + 0 = α σ²,

using `α μ + β = 0` and `Σ(y−μ)g = 0`. Equate the two evaluations: `α σ² = 1`, so `α = 1/σ²`, and then `β = −μ/σ²`. Therefore

  ∂ln g/∂μ = α y + β = y/σ² − μ/σ² = (y − μ)/σ².

So the mean-parameter eligibility is `(y−μ)/variance` for the *entire* exponential family — Bernoulli, Gaussian, Poisson, exponential, all of them — which is exactly why the Bernoulli and Gaussian came out matching. The framework didn't get two lucky special cases; it has one general law and they're instances.

Two loose ends from the immediate-reward story, then the temporal extension.

First, how does this live alongside backpropagation, in case the net has deterministic hidden units feeding stochastic outputs? Put all the randomness in the output units `O`. Because the outputs are conditionally independent given their inputs, the network's overall output probability factors: `g(ξ, W, x) = ∏_{k∈O} g_k(ξ_k, w^k, x^k)`, so `ln g = Σ_{k∈O} ln g_k`, and therefore for any weight `w_ij` anywhere in the net,

  ∂ln g/∂w_ij = Σ_{k∈O} ∂ln g_k/∂w_ij.

That sum over output units of a derivative of a log-likelihood is *exactly* the kind of thing backprop computes. So the recipe is: at each stochastic output unit, "inject" its characteristic eligibility — for a Bernoulli-logistic output, `(y_k − p_k)/(p_k(1−p_k))` just after the squasher — and then run the ordinary backward pass to spread it to the upstream weights. The correlation-style REINFORCE estimate sits at the stochastic units; backprop carries it through the deterministic ones. Unbiasedness of the gradient estimate survives because backprop is just the chain rule and preserves it.

Second loose end: could I instead backpropagate *through the coin flip itself*? For the Gaussian I can — write `y = μ + σ z` with `z` a standard normal draw, and then `∂y/∂μ = 1`, `∂y/∂σ = z = (y−μ)/σ`, so `y` is a differentiable function of the parameters plus an auxiliary noise variable, and if I had `∂J/∂y` I could push it back to `μ, σ` deterministically. But this only works when the output is a differentiable function of the parameters and some external noise. For a generic Bernoulli unit it fails: if the downstream objective `J` is a nonlinear function of the binary output, there's no relation that lets `∂E{J|p}/∂p` be recovered from `E{∂J/∂y | p}` — the discrete coin flip has no reparameterization. So the eligibility route is the general one; the reparameterized route is a bonus available only for nicely-shaped continuous units.

Now the temporal problem, which is where a single reward must be shared across many decisions. Suppose the net runs for `k` steps and one reward `r` arrives at the end. Loops and delay seem to break the clean single-trial analysis. The way through is to *unfold in time*: duplicate the net once per step into an acyclic network `N*`, where each weight `w_ij` of the real net `N` becomes a family of tied copies `w_ij^t` in `N*`, one per time step, all constrained equal to the shared `w_ij`. The unfolded `N*` has no cycles, and it faces an ordinary associative reinforcement problem — so everything I proved applies to `N*` directly.

I just need to relate the real weight's gradient to the tied copies'. By the chain rule, since `w_ij^t = w_ij` for every `t`,

  ∂E{r|W}/∂w_ij = Σ_{t=1}^k (∂E{r|W*}/∂w_ij^t)(∂w_ij^t/∂w_ij) = Σ_{t=1}^k ∂E{r|W*}/∂w_ij^t,

because each `∂w_ij^t/∂w_ij = 1`. So the real gradient is the *sum* over time of the unfolded gradients. Now apply the single-trial REINFORCE result inside `N*`: for the `t`-th copy, the update `α_ij (r − b_ij) e_ij(t)` has expectation `α_ij ∂E{r|W*}/∂w_ij^t`, where `e_ij(t)` is the characteristic eligibility evaluated at step `t`. Summing over the copies — which physically means accumulating the per-step eligibilities into one number per weight — and using the chain-rule identity,

  E{Δw_ij | W} = E{ Σ_t α_ij (r − b_ij) e_ij(t) } = α_ij Σ_t ∂E{r|W*}/∂w_ij^t = α_ij ∂E{r|W}/∂w_ij.

Same conclusion as the single-trial lemma. So the *episodic* rule is

  Δw_ij = α_ij (r − b_ij) Σ_{t=1}^k e_ij(t),

and the very same inner-product argument gives the episodic theorem: the average episodic update has nonnegative inner product with `∇_W E{r|W}`, zero only at a stationary point, and equals the true gradient when the rate factors are a common constant. What makes this *implementable* online is that the eligibility sum `Σ_t e_ij(t)` is built up purely from the unit's own operation as the episode runs — each term depends only on that step's input and output — and it owes nothing to the reward, which only arrives at the end. One accumulator per weight; add `e_ij(t)` each step; when `r` comes, multiply the whole accumulated sum by `(r − b_ij) α_ij`. For a recurrent net of Bernoulli-logistic units that's `Δw_ij = α_ij (r − b_ij) Σ_t (y_i(t) − p_i(t)) x_j(t−1)`. The price of this clean online form is that it spreads credit *uniformly* over every step of the episode — it doesn't try to figure out which step mattered — so it's correct but slow; the temporal credit assignment is done by brute summation.

Let me close the loop on the whole chain. I wanted to climb expected reward `E{r|W}` in a network whose only feedback is a broadcast scalar, where backprop can't run because the path to the reward goes through a coin flip and an unknown environment. I found I could write the gradient of expected reward as `Σ_ξ E{r|y=ξ} ∂g/∂w`, and by multiplying and dividing by the probability — `∂g/∂w = g · ∂ln g/∂w` — I turned that into an expectation `E{ r · ∂ln g/∂w }`, so a single sampled `(output, reward)` pair gives an unbiased estimate of the gradient with no need to differentiate the reward. The log-probability gradient — the characteristic eligibility — is the only place the unit's structure enters, and it has mean zero because probabilities sum to one (`Σ ∂g = ∂1 = 0`); that single fact makes any output-independent baseline subtractable without bias, so the raw reward can be replaced by its surprise `r − b` to cut variance. Averaging over inputs (which drop out of the derivative because they're upstream of the weight) gives, for the whole network, an average update whose inner product with the true gradient is `Σ α_ij (∂E{r|W}/∂w_ij)² ≥ 0` — provable hill-climbing — and exact gradient ascent when the rates are constant. The Bernoulli automaton `L_{R-I}` and the associative `A_{R-I}` fall out as special cases; the Gaussian unit and the whole exponential family obey the same `(y−mean)/variance` mean-eligibility; deterministic sub-nets compose with backprop; and unfolding in time turns delayed reward into a sum of eligibilities, giving an online episodic rule with one accumulator per weight. The landing form, for a network of stochastic units, is one rule:

```python
import numpy as np

def logistic(s):
    return 1.0 / (1.0 + np.exp(-s))

class BernoulliLogisticUnit:
    """Stochastic unit: p = logistic(w·x), y ~ Bernoulli(p). REINFORCE-trained."""
    def __init__(self, n_in, alpha=0.1, gamma=0.1):
        self.w = np.zeros(n_in)
        self.alpha = alpha            # constant rate factor -> exact gradient ascent in mean
        self.gamma = gamma            # reinforcement-comparison averaging constant
        self.rbar = 0.0               # adaptive baseline (running reward prediction)

    def forward(self, x):
        self.p = logistic(self.w @ x) # the unit's distribution parameter
        self.y = float(np.random.rand() < self.p)   # sample the stochastic output
        self.x = x
        return self.y

    def eligibility(self):
        # characteristic eligibility of w_ij for a Bernoulli-logistic unit:
        # (y - p) * x   (the (y-p)/(p(1-p)) log-prob gradient times f'(s)=p(1-p) cancels)
        return (self.y - self.p) * self.x

    def learn(self, r):
        # Delta w = alpha * (r - baseline) * characteristic_eligibility
        dw = self.alpha * (r - self.rbar) * self.eligibility()   # unbiased grad-ascent step
        self.w += dw
        # update the running baseline after the current weight update
        self.rbar = self.gamma * r + (1 - self.gamma) * self.rbar
        return dw


class GaussianUnit:
    """Continuous stochastic unit: y ~ N(mu, sigma^2); mu, sigma adapted by REINFORCE."""
    def __init__(self, alpha=0.01, gamma=0.1):
        self.mu, self.sigma = 0.0, 1.0
        self.alpha, self.gamma, self.rbar = alpha, gamma, 0.0

    def forward(self):
        self.z = np.random.randn()
        self.y = self.mu + self.sigma * self.z       # y = mu + sigma * z
        return self.y

    def learn(self, r):
        b = self.rbar
        mu, sigma = self.mu, self.sigma
        # eligibilities: d ln g / d mu = (y-mu)/sigma^2 ; d ln g / d sigma = ((y-mu)^2 - sigma^2)/sigma^3
        # with rate alpha_mu = alpha_sigma = alpha * sigma^2 the sigma^2 denominators tidy up:
        self.mu    += self.alpha * (sigma**2) * (r - b) * (self.y - mu) / sigma**2
        self.sigma += self.alpha * (sigma**2) * (r - b) * ((self.y - mu)**2 - sigma**2) / sigma**3
        self.sigma = max(self.sigma, 1e-3)           # keep sigma positive in practice
        self.rbar = self.gamma * r + (1 - self.gamma) * self.rbar


def episodic_update(units_steps, r, alpha=0.1, baseline=0.0):
    # units_steps: per weight, the list of per-step eligibilities e_ij(t) over the episode.
    # Delta w_ij = alpha (r - b) * sum_t e_ij(t)  -- one accumulator per weight, built online,
    # multiplied by (r - b) only when the single episode-end reward arrives.
    return alpha * (r - baseline) * np.sum(units_steps, axis=0)
```
