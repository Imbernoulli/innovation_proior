Let me start from what my regression net actually gives me and where it stops being enough. I train a feed-forward net by minimizing the sum of squared errors between its output and the target, and I get one predicted number per input. I know — and I should hold onto this, because it's the whole reason squared error is the "right" loss for a point predictor — that at the minimum the net output equals the conditional average of the targets, ⟨d|x⟩. The clean way to see it: write the infinite-data error, add and subtract ⟨d|x⟩ inside the square, and it splits into ∫[f(x)−⟨d|x⟩]² p(x) dx plus ∫[⟨d²|x⟩−⟨d|x⟩²] p(x) dx. Only the first piece depends on the weights, and it's zero exactly when the net reproduces the conditional mean; the second piece doesn't touch the weights at all. So my net is hunting for the conditional mean, and it's doing it well. But look at that second, leftover piece — it's the average of the conditional variance of the targets around their mean, integrated over inputs. It's *in* the problem, it's the residual the loss can never drive to zero, and my training throws it away as a single scalar I read off at the end. The residual sum of squares divided by the number of scalar target observations is one global variance number. One error bar for the entire input space.

That's the thing that's killing me. My targets are not equally noisy everywhere. In the climate data some atmospheric regimes map almost deterministically to their tendencies and some are genuinely spread out — the intrinsic noise level is a function of where I am in input space, not a constant. A single residual variance says "your prediction is this uncertain, everywhere," which is wrong in both directions at once: too confident in the turbulent regions, too timid in the calm ones. I want an error bar that is itself a function of x. And not as a bolt-on computed after the fact — I want the net to *learn* it, the same way it learns the mean, from the same gradient descent.

So how do I make the network produce a second number per input that means "the variance of the target here"? The obvious move is to just add a second output. Give the net two heads: one that outputs my prediction of the value, call it ŷ(x), and one that outputs my prediction of the variance, call it σ̂²(x). Easy to add the wire. The hard question, the one that actually decides whether this works, is: what do I train the second head against? For ŷ I have a target — it's the observed d. For σ̂² I have no target. Nobody handed me "the true local variance at x"; I only ever see one realized d per input. I can't write a squared-error loss "(σ̂²(x) − true variance)²" because I don't have the true variance to put in it. I'm stuck. There's no supervised target for the variance head.

Let me back up to the principle underneath squared error, because that's where the answer has to come from. Squared-error training isn't fundamental — it's a consequence. The fundamental thing is: assume a probability model for how the target is generated given the input, then do gradient descent on the negative log-likelihood of the data under that model. If I assume the target is the net's output plus zero-mean Gaussian noise with a *fixed* variance, the negative log of that Gaussian is, up to constants, the squared error — that's where SSE comes from, and that's exactly why it only ever recovers the mean and a single global variance. The constant-variance assumption is baked in; the loss inherits it. So maybe I don't need a target for the variance head at all. If I put the variance *into the probability model* — let the model say the noise variance depends on x — then the likelihood depends on σ̂²(x), and minimizing the negative log-likelihood would train σ̂²(x) along with ŷ(x) from the same gradient. Whether that actually invents a usable training signal for the variance head is the thing I have to check, not assume; let me write it down and look.

Let me write it down carefully. I model the observed target as d(x) = y(x) + n(x): an underlying function y(x) I'm trying to recover, plus noise n(x) drawn from a distribution whose width I now allow to depend on x. Take the noise Gaussian, but with an input-dependent variance σ²(x). Then the conditional density of the target is

  P(d_i | x_i) = [2π σ²(x_i)]^{−1/2} · exp{ −[d_i − y(x_i)]² / (2 σ²(x_i)) }.

This is the same Gaussian as before with one change: σ² carries an index i, it varies from input to input. Now my net outputs ŷ(x_i) as the estimate of the mean y(x_i) and σ̂²(x_i) as the estimate of the variance σ²(x_i). The negative log-likelihood of one observation under this model is the thing I'll descend. Take −ln of that density:

  −ln P(d_i | x_i) = ½ ln[2π σ̂²(x_i)] + [d_i − ŷ(x_i)]² / (2 σ̂²(x_i))
                   = ½ ln σ̂²(x_i) + [d_i − ŷ(x_i)]² / (2 σ̂²(x_i)) + ½ ln 2π.

Sum over all patterns i — the data are assumed independent, so the joint log-likelihood is the sum of the per-pattern ones — and pull out a ½:

  C = ½ Σ_i { [d_i − ŷ(x_i)]² / σ̂²(x_i) + ln σ̂²(x_i) + ln 2π }.

There's my cost. Two new terms relative to plain squared error: the squared error is now *weighted* by one over the predicted variance, and there's a ln σ̂² penalty on the predicted variance. The ½ ln 2π is a constant and drops out of every gradient, so I can ignore it. Before I read any mechanism into this, let me check the one thing I can check immediately: does it reduce to what I already trust? Freeze σ̂²(x) to a single constant c. Then C = ½ Σ_i [ (d_i−ŷ_i)²/c + ln c ] = (1/(2c)) Σ_i (d_i−ŷ_i)² + (N/2) ln c. Let me put numbers on it so I'm sure I haven't dropped a sign: with residuals (0.2, −0.5, 1.0, −0.1, 0.3) and c = 0.37, the formula ½ Σ [r²/c + ln c] evaluates to −0.60725, and (1/(2c))·SSE + (N/2) ln c evaluates to the same −0.60725. So with the variance frozen, C is the sum-squared error times a *positive* constant 1/(2c) plus a term independent of the weights — minimizing C in the weights is exactly minimizing SSE. The heteroscedastic cost contains the homoscedastic one as the special case where the variance head is flat. I haven't broken the thing that works; I've generalized it.

Now I need to see whether this cost actually trains the variance head with no target. Forget the network for a second and ask: holding ŷ fixed, what single value of σ̂² minimizes [d−ŷ]²/σ̂² + ln σ̂² for one pattern? Differentiate with respect to σ̂²: −[d−ŷ]²/σ̂⁴ + 1/σ̂² = 0, which rearranges to σ̂² = [d−ŷ]². So the cost's preferred variance at a point is the squared residual there. Let me not just trust the algebra — let me put a residual in and scan. Take residual r = 0.7, so r² = 0.49, and sweep σ̂² over a fine grid minimizing r²/σ̂² + ln σ̂². The numeric argmin comes out at σ̂² = 0.49 — exactly r². And the ln σ̂² term is what stops the trivial cheat: without it the cost would send σ̂²→∞ everywhere to zero out the first term (a huge claimed error bar makes any residual "explained"); the ln penalty charges you for claiming large variance, and the two pull against each other with the balance landing at σ̂² = squared error. So the variance head, with no target of its own, is being asked to predict the squared error — that's the target the likelihood invented, and I've now watched it land where the algebra said it would rather than taken it on faith.

While I'm at it, the homoscedastic case is worth pinning down the same way, because I'll need it for initialization later. If a single shared σ̂² has to cover N patterns, the cost in that one variable is (Σ_i r_i²)/σ̂² + N ln σ̂², and the stationary point is σ̂² = (Σ r_i²)/N — the mean of the squared residuals, i.e. the global MSE. Scanning it numerically with the five residuals above, the argmin is 0.278, and mean(r²) is 0.278. Good: the best single error bar this cost can give is exactly the global MSE I was reading off before. That's the natural place to start the variance head from.

Let me make the training behavior precise by actually taking the gradient through the network, because the form of the gradient is going to tell me something important. Say the ŷ head is a linear unit reading hidden features h_j, and weight w_{ŷj} connects feature j to ŷ. The cost's dependence on ŷ is through the first term, [d−ŷ]²/σ̂². Differentiate:

  ∂C/∂w_{ŷj} = Σ_i (1/σ̂²(x_i)) · [ŷ(x_i) − d_i] · h_j(x_i),

so the gradient-descent update is

  Δw_{ŷj} = η · (1/σ̂²(x_i)) · [d_i − ŷ(x_i)] · h_j(x_i).

Stare at that. It's the ordinary delta-rule update for a linear output — error times input feature — with one extra factor: 1/σ̂²(x_i). The learning signal for the mean is scaled by the inverse predicted variance of that pattern. Patterns the net believes are low-noise get a *larger* effective learning rate; patterns it believes are high-noise get a *smaller* one. This is weighted regression, and it's appearing on its own, just from differentiating the likelihood — I didn't put it in by hand. And it's a genuinely good thing: in clean regions the net should chase the data hard, in noisy regions it should not let a single noisy d yank it around, and outliers from high-noise regions stop being able to drag network capacity away from fitting the low-noise regions that could otherwise be nailed. The likelihood automatically discounts learning where it expects the target to be unreliable.

Now the variance head. Let the σ̂² unit read its own hidden features h_k through weights w_{σ²k}. The cost depends on σ̂² through both the weighted-error term and the ln term. Let me write the variance as some function of a network pre-activation and differentiate. I'll get to *which* function in a moment, but provisionally treat σ̂² as the output. Both terms contribute:

  ∂/∂σ̂² [ (d−ŷ)²/σ̂² + ln σ̂² ] = −(d−ŷ)²/σ̂⁴ + 1/σ̂²,

and chaining through to the weight gives, after multiplying by the σ̂²→pre-activation→weight derivatives, an update of the form

  Δw_{σ²k} = η · (1/(2σ̂²(x_i))) · { [d_i − ŷ(x_i)]² − σ̂²(x_i) } · h_k(x_i).

Read the bracket: [d−ŷ]² − σ̂². The variance head is being driven by the difference between the actual squared error and its current prediction of it — regression of σ̂² onto the squared errors, the same fixed point I found by minimizing one pattern. Let me check the sign of the bracket does what I'd want in the three obvious cases, with r² = 0.49: if σ̂² = 0.2 (head under-claims), the bracket is +0.29, pushing σ̂² up; if σ̂² = 0.49 (matched), the bracket is exactly 0.000, no push; if σ̂² = 0.9 (head over-claims), the bracket is −0.41, pushing σ̂² down. So the head climbs toward the squared error from either side and stops when it matches. And there's a 1/σ̂² in front, the same self-weighting as on the mean.

So the architecture so far: a net with a hidden representation, a linear ŷ unit, and a σ̂² unit, trained jointly on C. But that 1/σ̂² factor that I just praised is also a trap, and I need to think hard about *when* it bites, because if I just turn this loose it might fail. Here's the worry. At the very start of training, ŷ is garbage — random weights, it fits nothing. So the residuals [d−ŷ]² are large *everywhere*, but not uniformly: by luck some patterns will start with smaller residuals than others. The variance head, reading those residuals, will tag the accidentally-small-residual patterns as "low noise" (small σ̂²) and the accidentally-large-residual patterns as "high noise" (large σ̂²). Now feed that back into the mean's update: the low-σ̂² patterns get a big 1/σ̂² learning rate and get fit hard; the high-σ̂² patterns get a tiny learning rate and are essentially ignored. But those high-residual patterns are high-residual *because ŷ hasn't learned them yet*, not because they're intrinsically noisy. The model would be confusing "I haven't fit this yet" with "this is irreducible noise," and the weighted regression then makes the confusion self-fulfilling: it stops learning exactly the patterns it most needs to learn, and could converge to a badly suboptimal solution that has explained-away its own failures as noise. The weighted-regression term is beneficial once ŷ is roughly right and harmful while ŷ is still wrong. The danger is entirely about *order*: variance estimation must not get to reweight the mean's learning until the mean is good enough that a large residual really does signal noise rather than underfitting.

That points the fix at staging, not at changing the likelihood. The conservative schedule is to let the dangerous coupling switch on only after the mean is established. Phase one: pretend the variance is constant — i.e., drop the 1/σ̂² weighting entirely and just train ŷ on plain squared error, the traditional way, until it is a good estimate of the conditional mean. Stop it at the minimum of a held-out squared error so I don't overfit ŷ. At the end of this phase the only variance information I have is the global residual, which is the constant-variance world I started in, but now ŷ is trustworthy. Phase two: bring in the variance head while keeping the mean fixed, and train the σ̂² unit to predict the squared errors of the fixed ŷ — that's the Δw_{σ²k} update with ŷ held still, which is just function approximation with the squared errors as targets. Since ŷ is fixed, the 1/σ̂² weighting on the *mean* does no damage here; I'm fitting the variance to honest residuals from a good mean. The sensible initialization is the homoscedastic answer I worked out above: set the variance bias so that σ̂² starts at the global mean squared error from phase one, then let it specialize. Phase three: unfreeze everything and minimize the full cost C jointly, so the mean and variance can co-adapt, with the weighted regression now in the regime where it helps rather than hurts because both heads start from a good place.

There's a subtlety I want to get right on the data splits across phases, because overfitting can now bite in two places. In the careful small-data version, phase one trains ŷ on one half and watches the squared error on the other half to know when to stop. In phase two, if I trained the variance head on the *same* half ŷ was trained on, any overfitting of ŷ to that half would show up as artificially small residuals, and the variance head would learn to predict a too-small variance — it would inherit ŷ's overfitting as false confidence. So phase two swaps the halves: train the variance on the half ŷ did *not* see, where ŷ's residuals are honest, and cross-validate on the other. Then phase three re-splits and fits the joint cost. The bookkeeping is just guarding against the variance head mistaking memorized training residuals for low noise.

Now the one piece I deferred: what function should the variance unit's output be? σ̂² has to be positive — it's a variance, a negative one is meaningless and would make ln σ̂² undefined and 1/σ̂² explode through zero. If I let a linear unit output σ̂² directly, nothing stops gradient descent from sending it negative, or to exactly zero, and at zero the 1/σ̂² weighting detonates. I need the head to be positive by construction. The clean trick is to let the network produce an unconstrained pre-activation z and exponentiate it:

  σ̂²(x) = exp[ Σ_k w_{σ²k} h_k(x) + β ],

where β is the unit's bias. Now z = Σ_k w_{σ²k} h_k + β ranges over all the reals, exp(z) is always strictly positive, it can get arbitrarily close to zero without reaching it, and the configuration where the variance is pinned at zero is simply unreachable. The exponential is the natural choice for a scale parameter, and there's a gradient reason to prefer it over a raw variance output: because d(exp z)/dz = exp z = σ̂², one power of the inverse variance cancels in the chain rule. The gradient with respect to z becomes ½(1 − [d−ŷ]²/σ̂²), so the weighting risk is still present when σ̂² is too small, but the derivative is the log-scale derivative of the likelihood rather than the derivative of an unconstrained positive number. And note what the pre-activation z *is*: z = ln σ̂². The network's variance unit is, in its natural coordinates, predicting the **log-variance**. That also makes the initialization trivial: to start σ̂² at the global mean squared error, just set the bias β = ln(mean squared error from phase one). The exponential and the log-variance view are the same statement read in two directions.

Let me reconsider the architecture for the two heads, because how much they share matters. The cheapest option is to give the variance unit no hidden layer of its own and let it read only the *same* hidden features the mean head uses. But those features were shaped to represent ŷ(x); the variance σ̂²(x) is generally a different function of x, and forcing it through a basis tailored to the mean is too constraining — the variance can't bend the way it needs to. The opposite extreme is two entirely separate networks with no shared features at all; that's right only if the map from inputs to the mean and the map from inputs to the variance share nothing, which throws away the overlap that often exists. The compromise that respects both is to give the variance side its own capacity while still letting the two functions see the same input. In the code I need for the climate emulator, the clean version is two regularized MLPs with the same input: one MLP maps x to the mean, the other maps x to log-precision. That is a stronger separation than a single shared trunk, but it keeps the implementation simple and gives the variance function enough room not to be trapped in features built only for the mean.

One more practical guard before I write it. The full cost C can run very negative — if the variance head becomes very confident (σ̂² small) on patterns it predicts well, the ln σ̂² contribution... wait, let me get the sign straight, this is exactly the kind of place I'd fool myself. The cost has +½ ln σ̂² per pattern; as σ̂²→0 that term goes to −∞. And the weighted-error term [d−ŷ]²/σ̂² goes to +∞ if the residual is nonzero, but if the residual is also driven near zero on a well-fit clean pattern, the +∞ does not fire and the −∞ from the log can dominate, sending C very negative. A sharp, well-calibrated density genuinely has high likelihood, but numerically a runaway scalar loss can make the update useless. So in code I can clamp the scalar loss before back-propagation as a coarse guard: inside the clamp range I am optimizing the same NLL, and outside it I refuse the runaway update instead of letting it dominate training.

Now let me also think about going from a scalar target to the climate target vector. The direct extension is a diagonal Gaussian — assume the output components are independent given x, each with its own mean and its own variance, and the joint NLL is just the sum of the per-component NLLs. So the mean MLP outputs one value per target component, the log-precision MLP outputs one log-precision per target component, and the per-batch loss is the average over batch and target dimensions of the per-element Gaussian NLL. A fuller version would carry a full covariance matrix with cross-correlations between targets; the diagonal form is the natural first choice when the point metric scores per-component errors and I want one uncertainty channel per output variable.

And finally, the inference side. At test time the harness scores the predicted *value* with normalized MSE, R², RMSE — point metrics on a single prediction per input. The mean head ŷ(x) is that prediction; the precision head is the uncertainty channel, used for calibration / error bars, not for the point score. If I want to report the standard deviation, I take σ̂(x) = exp(−½ · logprec). The variance head's whole contribution to the point prediction is indirect — it changes training through the weighted regression — and that is the quiet payoff: estimating the local error bar is not just an add-on, it can keep high-noise residuals from stealing capacity from low-noise regions.

Let me write the cost first as a per-element thing in the precision convention, because that's cleaner to differentiate and is how I'll code it. Let the head output, per element, the log-precision ℓ = ln(1/σ̂²) = −ln σ̂² (equivalently I can call the head's output the negative log-variance; same network, just a sign convention on the readout). Then σ̂² = exp(−ℓ), the precision is τ = exp(ℓ), and twice the per-element NLL after dropping the ln 2π constant is

  (d − ŷ)² · τ − ℓ = (d − ŷ)² · exp(ℓ) − ℓ.

Let me verify this is the same object as the cost C and not a sign-flipped impostor. Twice the per-element NLL after dropping the constant is (d−ŷ)²/σ̂² + ln σ̂². Since 1/σ̂² = exp(ℓ) and ln σ̂² = −ℓ, it becomes (d−ŷ)²·exp(ℓ) − ℓ — the form above. If the head instead emits log-variance v = ln σ̂², the same object is (d−ŷ)²·exp(−v) + v, with ℓ = −v. Let me put numbers through both forms to be sure they coincide and not just look like they should: with r² = 0.49 and v ∈ {−1.0, 0.0, +0.7} (so ℓ = +1.0, 0.0, −0.7), the log-precision form gives 0.33196, 0.49000, 0.94333, and the log-variance form gives 0.33196, 0.49000, 0.94333 — identical to five places. And the middle row is a useful spot check on the whole scheme: at ℓ = 0, i.e. σ̂² = 1, the per-element loss is just r² = 0.49, exactly plain squared error, which is what unit variance should give. I'll code the log-precision form because that is the convention the climate implementation uses. (And while I have it open, the per-element optimum over ℓ of r²·exp(ℓ) − ℓ is at ℓ = −ln r²; scanning numerically with r² = 0.49 the argmin lands at ℓ = 0.7132 against −ln 0.49 = 0.71335 — the same σ̂² = r² fixed point, now seen in the coordinate the code actually optimizes.)

The full three-phase schedule is the most careful way to avoid the early-weighting trap when I can control every split and freeze. The climate harness wants a compact training path, so I keep the part that matters most for the code path: first fit the mean with plain MSE, then switch on the joint Gaussian likelihood once the mean has a head start. The switch is a simple epoch threshold, `epoch < total_epochs / 3` — the first third of training is the mean-only warmup and the joint NLL takes over after. Concretely, for 30 epochs that warms up on epochs 0–9 and switches at epoch 10; for 9 epochs it warms up on 0–2 and switches at 3 — a third of the run on plain MSE before the weighting is ever allowed to act, which is the staging the gradient analysis demanded. Two MLPs, each built from Linear, LayerNorm, Dropout, and ReLU blocks, map the same input to mean and log-precision. The point prediction used by evaluation is the mean.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(p=dropout),
            ))
        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for block in self.blocks:
            x = F.relu(block(x))
        return self.final_linear(x)


class Custom(nn.Module):
    """Two MLPs: one predicts the per-dimension mean, the other the
    per-dimension log-precision (= -log variance)."""

    def __init__(self, input_dim, output_dim, hidden_dim=512, layers=1, dropout=0.0):
        super().__init__()
        self.mean = MLP(input_dim, output_dim, hidden_dim, layers, dropout)
        self.logprec = MLP(input_dim, output_dim, hidden_dim, layers, dropout)

    def forward(self, x):
        mu = self.mean(x)
        logprec = self.logprec(x)
        return mu, logprec

    def predict_mean(self, x):
        mu, _ = self(x)
        return mu

    def predict_distribution(self, x):
        mu, logprec = self(x)
        std = torch.exp(-0.5 * logprec)
        return mu, std


def gaussian_nll(mu, logprec, target):
    # twice the per-element NLL, with ln 2pi dropped:
    #   (d - mu)^2 / sigma^2 + ln sigma^2  =  (d - mu)^2 * exp(logprec) - logprec
    prec = torch.exp(logprec)                                       # tau = 1/sigma^2
    return (prec * (target - mu) ** 2 - logprec).mean()


def training_loss(model, x, y, epoch, total_epochs):
    mu, logprec = model(x)
    if epoch < total_epochs / 3:
        # Phase-I warmup: fit the mean only, with plain squared error, so the
        # 1/sigma^2 weighting never reshapes learning while mu is still bad.
        return ((y - mu) ** 2).mean()
    # Then the scaled Gaussian NLL co-adapts mean and precision.
    loss = gaussian_nll(mu, logprec, y)
    return torch.clamp(loss, min=-1e5, max=1e5)                     # coarse runaway guard
```
