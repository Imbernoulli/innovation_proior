Let me start from the thing that keeps biting me. I have a network of stochastic units. Each unit doesn't commit to an output, it samples one from a distribution it controls — a Gaussian unit draws `y ~ N(μ, σ²)`, a Bernoulli unit flips a coin with `Pr{y=1}=p`. I score the whole thing by a scalar performance signal — call it reward `r`, though in a supervised setting "reward" is just the log-likelihood the model assigns to the right answer — and I want to push the parameters up that signal. The tool I trust for that is the gradient. For expected reward `E{r|θ}` I have a clean way to get it from samples: the score-function identity, `d/dθ E{r|θ} = E{ r · ∂ ln g/∂θ }`, where `∂ ln g/∂θ` is the characteristic eligibility of the parameter. So I run REINFORCE: `Δθ = α (r − b) ∂ ln g/∂θ`, reward-increment equals a nonnegative rate times the offset reward `(r−b)` times the eligibility, and the baseline `b` just cuts variance without biasing the direction. The unbiasedness is the whole appeal — on average I climb `∇_θ E{r|θ}` exactly. For a Gaussian unit the eligibilities are `∂ ln g/∂μ = (y−μ)/σ²` and `∂ ln g/∂σ = ((y−μ)²−σ²)/σ³`, so the rule nudges `μ` toward sampled actions that beat the baseline and grows or shrinks `σ` depending on whether the realized squared error `(y−μ)²` ran above or below `σ²`. All correct. All principled.

And it keeps converging to junk. Not because the gradient is wrong — it's unbiased, I checked — but because it's a *gradient*, and gradient ascent finds local optima, and worse, the search distribution itself is one of the things the gradient is allowed to move. Watch what happens to `σ`. Early on the unit samples widely, occasionally lands on something good, and `μ` drifts toward it. But the same dynamics that pull `μ` toward a decent region also, through the `σ` eligibility, start shrinking `σ` the moment the unit is doing "well enough" — `(y−μ)²` is small near a comfortable `μ`, so the `σ` update is negative, `σ` falls, the samples cluster tighter, and now the unit can only sample near where it already is. The distribution sharpens toward a near-deterministic output. The instant it does, exploration is dead: there's no spread left to stumble onto anything better, and the eligibility variance collapses too, so even the learning signal fades. The agent has committed — early, confidently — to whatever local optimum it happened to be near when the spread ran out. On the hierarchical tasks this is brutal, because there I need the agent to keep *several* options live at once for a while, and the collapse kills them one by one before the coordination has a chance to pay off. The empirical fact is stark: reward-following alone reliably collapses, reliably too soon.

So what do I actually want? I want to keep climbing reward but I want to *resist the collapse* — to keep the distribution spread out until the reward has genuinely earned the right to sharpen it. The reward-penalty algorithm from Barto's group does fight back, with that extra `λ(1−r)(1−y−p)` term that on failure shoves the Bernoulli unit toward the opposite of what it just did, and empirically it does escape some traps REINFORCE falls into. But it's a hand-built correction welded to the `0/1` reward-penalty form; it isn't telling me, at the level of the distribution, *how much spread to keep*, and I can't carry it over to a Gaussian unit or to a likelihood objective. And the ε-greedy style fix — keep the policy greedy but act random with small probability — is exploration bolted on *outside* the objective: the noise level is a fixed knob, applied the same whether the agent is wildly over-confident or appropriately uncertain, and the policy I'm optimizing is still free to collapse underneath the noise. None of these is a statement about the policy's own shape.

Let me try to name the quantity I actually care about. "Spread of the distribution," "how non-committal it is," "how much it's hedging." That's a measured thing. For a discrete output it's `−Σ_y p(y) log p(y)`; for a density it's `−∫ g log g = E[−log g(y)]`. Entropy. And entropy has exactly the monotonicity I keep complaining about: it is *maximal* when the distribution is most spread, and it falls toward its floor as the distribution concentrates. For a Gaussian, `H = ½ log(2π e σ²)`, which rises with `σ` — so "shrinking `σ`" and "destroying entropy" are the same act. For a Bernoulli, `H = −p log p − (1−p) log(1−p)`, peaked at `p=½` and zero at `p∈{0,1}` — so "driving `p` deterministic" is again "killing entropy." The collapse I keep suffering *is* an entropy collapse. That reframes the whole problem: the failure mode is the optimizer driving entropy to zero too fast, so the cure should be a force that pushes entropy back up — something that makes low entropy *cost* the optimizer something.

And there's an old principle that says preferring high entropy isn't a hack, it's the right default. Jaynes: among all distributions consistent with what you actually know, choose the one of maximum entropy — the least-committal, the one that assumes no structure beyond what the evidence forces. My un-regularized agent is doing the opposite: it commits to a near-delta on the strength of a handful of early samples, asserting far more certainty than the evidence supports. So I don't want to *replace* the reward objective with entropy — I still need to fit — but I want to bias it toward the maximum-entropy solution among those that fit. The natural way to express "do well *and* stay spread" is to add the two: maximize

  J(π) = E_{a~π}[ r(a) ] + τ · H(π),

with `τ > 0` a small weight. The first term still climbs reward; the second penalizes collapse, with `τ` setting how hard. Write `H(π) = E_{a~π}[ −log π(a) ]` and it folds into one expectation,

  J(π) = E_{a~π}[ r(a) − τ log π(a) ].

I should make sure this `τ` is doing what I think before I commit to it, so let me solve the idealized problem exactly: forget the parametric network for a second and ask what distribution `π` *maximizes* `J` over all distributions, subject only to `Σ_a π(a) = 1`. Lagrangian,

  L = Σ_a π(a) r(a) − τ Σ_a π(a) log π(a) + λ ( Σ_a π(a) − 1 ).

Differentiate w.r.t. a single `π(a)`:

  ∂L/∂π(a) = r(a) − τ ( log π(a) + 1 ) + λ = 0,

so `log π(a) = (r(a) + λ)/τ − 1`, i.e. `π(a) ∝ exp( r(a)/τ )`. Normalize: `π*_τ(a) = exp(r(a)/τ) / Z`, `Z = Σ_a exp(r(a)/τ)`. The Boltzmann distribution. So the entropy bonus doesn't just "encourage randomness" vaguely — it makes the *target* of the optimization a temperature-`τ` softmax over reward. Now read off the two limits, because they tell me whether `τ` is the right knob. As `τ → 0`, `exp(r/τ)` is dominated entirely by the largest `r`, so `π*_τ → ` a delta on `argmax_a r(a)` — pure greedy, deterministic, exactly the collapse I get with no bonus (`τ=0` recovers plain REINFORCE, good, the family contains the baseline as a special case). As `τ → ∞`, `r(a)/τ → 0` for all `a`, so `π*_τ →` uniform — pure exploration, ignores the reward entirely. In between, `τ` slides continuously from exploit to explore. That's precisely the dial I wanted: a single small `τ` that says "keep this much spread regardless of how confident the reward gradient wants you to get." Too small and I'm back to collapsing; too large and the target is near-uniform and I stop fitting the data. I want just enough to defeat premature collapse, which says `τ` should be *small but nonzero*.

There's a second reading of the same objective that makes the "prevents collapse" claim sharp rather than hand-wavy. Plug `π*_τ` back in. With `π ∝ exp(r/τ)`,

  J(π) = Σ_a π(a)[ r(a) − τ log π(a) ].

Take any `π` and compare it to the optimal `π*_τ`. Substitute `r(a) = τ log π*_τ(a) + τ log Z` (from the Boltzmann form, `r = τ log(Z π*)`):

  J(π) = Σ_a π(a)[ τ log π*_τ(a) + τ log Z − τ log π(a) ]
       = τ log Z − τ Σ_a π(a) log( π(a)/π*_τ(a) )
       = τ log Z − τ · D_KL( π ‖ π*_τ ).

So `J(π) = −τ · D_KL(π ‖ π*_τ) + τ log Z`, and since KL is nonnegative and zero only at `π = π*_τ`, maximizing the regularized objective is *exactly* minimizing the KL divergence to the spread-out Boltzmann target. Without the `τH` term the implicit target would be the delta on argmax; *with* it, the target is the soft, full-support `π*_τ`, and pushing `π` toward a full-support target is structurally incapable of driving it to a premature delta. The entropy bonus is what turns the optimization's destination from "a point" into "a distribution with mass everywhere reward is non-tiny." That's the mechanism, derived, not asserted.

Now I have to actually optimize `J` over my parametric `π_θ`, not the idealized free `π`, so I need `∇_θ J`. The reward part is the usual score-function gradient, `∇_θ E_{a~π_θ}[r(a)] = E[ r(a) ∇_θ log π_θ(a) ]` — REINFORCE, unchanged. The new part is `τ ∇_θ H(π_θ)`, and I need to differentiate it cleanly because `H` depends on `θ` both through the sampling distribution and through the `log π_θ` inside. `H(π_θ) = −E_{a~π_θ}[ log π_θ(a) ] = −Σ_a π_θ(a) log π_θ(a)`. Differentiate:

  ∇_θ H = −Σ_a [ (∇_θ π_θ(a)) log π_θ(a) + π_θ(a) ∇_θ log π_θ(a) ].

The second sum is `−Σ_a π_θ(a) ∇_θ log π_θ(a) = −Σ_a ∇_θ π_θ(a) = −∇_θ Σ_a π_θ(a) = −∇_θ 1 = 0` — the score has zero mean. The first sum, using `∇_θ π_θ = π_θ ∇_θ log π_θ`, is `−Σ_a π_θ(a) (∇_θ log π_θ(a)) log π_θ(a) = −E[ log π_θ(a) ∇_θ log π_θ(a) ]`. So

  ∇_θ H(π_θ) = −E_{a~π_θ}[ ( 1 + log π_θ(a) ) ∇_θ log π_θ(a) ] = −E[ log π_θ(a) ∇_θ log π_θ(a) ],

the last step dropping the `1·∇log` term because it's the zero-mean score again. Folding everything together, the regularized gradient is

  ∇_θ J = E_{a~π_θ}[ ( r(a) − τ log π_θ(a) ) ∇_θ log π_θ(a) ].

If I keep the undropped form it is `E[(r(a) − τ log π_θ(a) − τ)∇log π_θ(a)]`, but the extra `−τ` is only a constant baseline because `E[∇log π_θ]=0`; it changes variance, not the expected gradient. So the useful view is the same REINFORCE shape with the reward replaced by `r − τ log π_θ`: the optimizer is rewarded both for high `r` and for sampling events that the current policy has not made too probable. That `−τ log π_θ` bonus is large for low-probability actions and disappears only when the distribution has enough spread, so it pushes back precisely against premature over-confidence.

Let me make the "pushes back hardest at collapse" quantitative, because that's the real reason this beats a fixed noise floor. Take a Gaussian unit and look at the entropy term as `σ` shrinks toward zero — the collapse direction. `H = ½ log(2π e σ²) = log σ + const`, so `∂H/∂σ = 1/σ`. As `σ → 0` that derivative *blows up*: the entropy gradient pushing `σ` back up grows without bound exactly as the distribution approaches a delta. Meanwhile the reward part near a comfortable local optimum is *flat* — that's what "local optimum" means, small `∂E{r}/∂σ`. So in the dangerous regime the reward gradient is tiny and the entropy gradient is enormous; their sum points firmly back toward spread, and the optimizer simply cannot drive `σ` to zero — it gets held off at the `σ` where `τ/σ` balances the (small) reward gradient. A fixed external noise floor can't do this: it doesn't know the policy is collapsing, it adds the same noise everywhere; the entropy bonus is *adaptive* because its own gradient steepens precisely as the thing it's preventing approaches. That's the payoff of regularizing the policy's shape from *inside* the objective rather than bolting exploration on outside.

Now the practical wrinkle: I need `H(π_θ)` (and its gradient) for whatever distribution my model emits. For the easy cases I have closed forms — Gaussian `H = ½ log(2π e σ²)` per dimension, Bernoulli `H = −p log p − (1−p) log(1−p)` — and if my output is a single Gaussian I'd just write that down and differentiate it analytically. But the distributions I actually care about are richer. If the model emits a *mixture* of Gaussians — several modes, each a Gaussian, weighted — there is no closed form for the entropy, because `H = −E[ log Σ_k w_k N_k(a) ]` has a log-of-a-sum inside the expectation that doesn't integrate in closed form. So I fall back on the one estimator that works for *any* distribution I can sample from and evaluate the density of: Monte-Carlo. Draw `a^(k) ~ π_θ`, and

  H(π_θ) = E_{a~π_θ}[ −log π_θ(a) ] ≈ (1/K) Σ_{k=1}^K −log π_θ( a^(k) ).

This is an unbiased estimate of the entropy value — it's just the sample mean of `−log π_θ` over draws from `π_θ` itself — and it's exactly the value I want to subtract from the loss. For a single draw `K=1` it's `−log π_θ(a)`, `a ~ π_θ`: sample one action from your own policy, take its negative log-prob, that's the entropy estimate.

I have to be careful about the gradient, because `MixtureSameFamily.sample()` is not reparameterized. If I draw a sample under `no_grad` and simply backpropagate through `−log π_θ(a)` at that fixed point, the expected gradient is `−E[∇log π_θ(a)] = 0`; that estimates the entropy value but not the entropy gradient. The gradient I derived is a score-function gradient, `−E[log π_θ(a) ∇log π_θ(a)]`, so the autograd surrogate has to carry `log π_θ(a)` as a detached multiplier on `∇log π_θ(a)`. Concretely: draw `a` with `dist.sample(...)` under `no_grad`, compute `logp = dist.log_prob(a)`, keep `entropy_value = −mean(logp)` for the scalar value, and use `entropy_score_grad = −mean(logp.detach() * logp)` for the gradient. The straight-through combination `entropy = entropy_value.detach() + entropy_score_grad − entropy_score_grad.detach()` has the forward value of the Monte-Carlo entropy estimate and the backward gradient `−E[logπ ∇logπ]`. That is the piece the loss needs.

Let me also pin down the sign, because the sign is the entire method and it's the easiest thing to get backwards. I want to *maximize* `J = E[r] + τ H`. In a likelihood/BC setting the "reward" I'm maximizing is the log-likelihood of the target, `log π_θ(target)`, so `E[r]` maximization is `log π_θ(target)` maximization, which as a *loss to minimize* is the negative log-likelihood, `NLL = −E[log π_θ(target)]`. Maximizing `J` then means minimizing `−J = NLL − τ H`. So the loss is

  loss = NLL − τ · H = −E[ log π_θ(target) ] − τ · ( −E_{a~π_θ}[ log π_θ(a) ] ).

The entropy enters the *minimized* loss with a *minus* sign — `−τH` — because minimizing `−τH` means maximizing `H`. If I wrote `+τH` in the loss I'd be *minimizing* entropy, i.e. actively driving the collapse I'm trying to prevent — the exact opposite. So: subtract the entropy bonus from the loss. The weight `τ` (I'll call it `alpha` in code) is small and positive, sized to escape collapse without dragging the fit toward the near-uniform target — a value on the order of `0.01` is the kind of small coefficient this calls for, big enough to keep the distribution honestly spread, small enough that the data-fit term still dominates wherever the data actually has something to say.

So the whole method, landed: keep the negative-log-likelihood fitting term, and *subtract* a Monte-Carlo estimate of the policy's entropy, weighted by a small `alpha`. The fitting term pulls the distribution toward the target; the entropy term resists letting it become a delta; `alpha` trades the two; the implied target is the Boltzmann-softened version of the pure-fit solution, which has full support and so cannot be a premature collapse. Let me fill the open slot in the loss module with exactly that, for the realistic case where the model emits a `torch.distributions.MixtureSameFamily` GMM: samples come from `.sample(sample_shape)`, `.log_prob(samples)` returns the matching sample/batch log probabilities, and the entropy has no closed form.

```python
import torch
import torch.nn as nn


class EntropyRegularizedLoss(nn.Module):
    """Negative log-likelihood of the target plus a maximum-entropy bonus on the
    predicted distribution. Minimizing this maximizes  log-likelihood + alpha * H(pi),
    so the fit term concentrates the distribution while the entropy term resists
    collapse to a near-deterministic (delta) solution -- the optimization's implicit
    target becomes the spread-out Boltzmann softening pi* ~ exp(reward/alpha) rather
    than a delta on the argmax."""

    def __init__(self, alpha=0.01, num_entropy_samples=1):
        super().__init__()
        self.alpha = alpha          # entropy weight tau: small => keep just enough spread
        self.num_entropy_samples = num_entropy_samples

    def forward(self, dist, target):
        # Fitting term: NLL of the target = -(log-likelihood). Minimizing it maximizes
        # the likelihood the predicted distribution assigns to the observed target.
        nll = -dist.log_prob(target).mean()

        # Entropy bonus H(pi) = E_{a~pi}[ -log pi(a) ].  MixtureSameFamily has no
        # closed-form entropy, so estimate the value by Monte-Carlo.  Its samples are
        # non-reparameterized, so use a score-function surrogate for the backward pass:
        # grad H = -E[log pi(a) * grad log pi(a)] after the zero-mean score term drops.
        with torch.no_grad():
            samples = dist.sample((self.num_entropy_samples,))
        log_prob = dist.log_prob(samples)
        entropy_value = -log_prob.mean()
        entropy_score_grad = -(log_prob.detach() * log_prob).mean()
        entropy = entropy_value.detach() + entropy_score_grad - entropy_score_grad.detach()

        # Subtract the entropy: minimizing (nll - alpha*H) maximizes H.  A PLUS sign here
        # would minimize entropy and *cause* the collapse we are preventing.
        return nll - self.alpha * entropy
```

Tracing the causal chain back: I started with REINFORCE climbing expected reward with an unbiased score-function gradient, and watched it reliably collapse the search distribution to a near-deterministic policy too early — because the same gradient that fits also shrinks the spread, and once the spread is gone so is exploration, leaving the agent stuck in whatever local optimum it first wandered near, worst of all on hierarchical tasks. The reward-penalty and ε-greedy fixes fight this from outside the objective and don't say how much spread to keep at the level of the distribution itself. Recognizing that "spread" is entropy, and that the collapse is an entropy collapse, pointed at making low entropy cost the optimizer: add `+τH` to the objective, a soft maximum-entropy bias (Jaynes) that prefers the least-committal fit. Solving the idealized objective by Lagrange multipliers showed the entropy bonus turns the optimization's target into the Boltzmann distribution `π* ∝ exp(r/τ)` — greedy as `τ→0` (recovering plain REINFORCE), uniform as `τ→∞`, a smooth dial in between — and rewriting the objective as `−τ D_KL(π ‖ π*) + τ log Z` proved that maximizing it is minimizing KL to a *full-support* target. Differentiating `H` gave `∇H = −E[logπ ∇logπ]` after the zero-mean score term vanished, and the Gaussian case showed the entropy gradient steepens (`H = ½ log(2πeσ²)`, `∂H/∂σ = 1/σ`) exactly as the distribution approaches collapse. Since rich mixture output distributions have no closed-form entropy, the entropy value is estimated by Monte-Carlo, `H ≈ −mean(log π_θ(a))` with `a ~ π_θ`, while a score-function surrogate supplies the same `−E[logπ ∇logπ]` gradient for PyTorch's non-reparameterized `MixtureSameFamily.sample()`. And in a likelihood/BC setting the reward is the log-likelihood, so the final loss is the negative log-likelihood minus `alpha` times that entropy term — fit the target, but keep the distribution honestly spread.
