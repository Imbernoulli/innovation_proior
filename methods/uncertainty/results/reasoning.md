Let me start from what actually hurts. I have one shared encoder feeding several heads — say a depth regressor and a semantic classifier, or a fine-class head and a coarse-class head — and I train them together because the shared representation is cheaper at test time and because the auxiliary task ought to regularise the features and help the one I care about. To train it I have to collapse the per-task losses into a single scalar, and the only thing everyone does is add them up with weights: L_total = Σ_i w_i L_i. And the weights are killing me. The losses aren't even in the same units — depth error is in meters-squared, cross-entropy is dimensionless nats, the instance-vector loss is in pixels — so their raw magnitudes can differ by orders of magnitude, and the "right" relative weight depends on those arbitrary units and on how noisy each task's labels are. When I sweep the weight of a two-task model across its range I see the same picture every time: each task is best somewhere in the interior, not at its single-task extreme, the two interiors don't line up, and there's a narrow compromise band where both tasks beat their single-task baselines, flanked on both sides by regions where pushing the weight further wrecks one task. Finding that band is a full training run per grid point — days each — and the search blows up combinatorially as I add tasks. So for three or more tasks I'm stuck either guessing uniform weights, which lands me outside the good band, or burning a week on grid search. I want the relative weighting to be set automatically and to adapt to the data.

The obvious first thing to try: just make the w_i trainable parameters and let SGD set them. Add w_1, w_2 as learnable scalars to L_total = w_1 L_1 + w_2 L_2 and optimise everything jointly. Stare at the gradient for a second. ∂L_total/∂w_i = L_i, which is nonnegative, so gradient descent pushes every w_i down, and there is nothing in this objective that pushes back. The global optimum over the w_i is w_1 = w_2 = 0, where L_total = 0 and the network has learned nothing. The weights collapse to zero. So I can't just learn the weights of a bare weighted sum — the objective is happy to turn off every task. Whatever scheme learns the weights has to carry a term that resists shrinking them, a cost for declaring a task unimportant. That's the real constraint: I need a *principled* coupling between "how much I down-weight a task" and "a penalty for doing so," not a free knob.

Where would such a coupling come from naturally? I keep coming back to the fact that most of these losses are negative log-likelihoods in disguise. The depth L2 loss ||y − f^W(x)||² is, up to constants, the NLL of a Gaussian observation model p(y | f^W(x)) = N(f^W(x), σ²): write out −log N and you get (1/2σ²)||y − f^W(x)||² + (1/2)log σ² + const. In ordinary single-task training I treat σ as a fixed constant, fold it into the learning rate, and never write it down — for one task a constant noise level is just an overall rescale of the loss. But here's the thing: that constant σ is *exactly* a per-task scale. A task with noisier labels, or with a loss measured on a bigger numerical scale, has a bigger σ. And σ doesn't have to be fixed — I can make it a free parameter and learn it. This is the move from heteroscedastic regression: Nix and Weigend, back in 1994, trained a network with a mean head and a variance head and learned the noise by maximising the Gaussian likelihood, minimising Σ_i ||y_i − μ(x_i)||²/σ²(x_i) + log σ²(x_i). They never needed "uncertainty labels" — the likelihood couples σ to the residuals, so it's recovered implicitly. Their σ depends on the input x, which is the heteroscedastic case, but I don't need input-dependence here. I want one constant σ_i per task — the homoscedastic, task-dependent version — because what varies across my problem isn't the noise from pixel to pixel, it's the noise and the scale from *task* to *task*.

So let me just write down the joint likelihood for two regression outputs and turn the crank, treating each σ_i as a learnable parameter alongside the weights W. If the two outputs are conditionally independent given the shared features, the likelihood factorises:

  p(y_1, y_2 | f^W(x)) = N(y_1; f^W(x), σ_1²) · N(y_2; f^W(x), σ_2²).

Minimise the negative log of that. Each Gaussian contributes its −log:

  L(W, σ_1, σ_2) = −log p(y_1, y_2 | f^W(x))
                 ∝ (1/2σ_1²)||y_1 − f^W(x)||² + (1/2σ_2²)||y_2 − f^W(x)||² + log σ_1 + log σ_2,

where I used −log N(y; μ, σ²) = (1/2σ²)||y − μ||² + (1/2)log σ² + const and (1/2)log σ_1² + (1/2)log σ_2² = log σ_1 + log σ_2, dropping the additive constants that don't affect the optimisation. Writing L_1(W) = ||y_1 − f^W(x)||² and L_2(W) = ||y_2 − f^W(x)||² for the bare per-task losses,

  L(W, σ_1, σ_2) = (1/2σ_1²)L_1(W) + (1/2σ_2²)L_2(W) + log σ_1 + log σ_2.

Now look at what fell out. The coefficient on each task's loss is 1/(2σ_i²) — an *inverse-variance weighting*. As σ_1 grows, the weight on L_1 shrinks: the noisier (or larger-scale) a task is, the less it's allowed to dominate the shared gradient. As σ_1 shrinks, its weight grows. That's precisely the adaptive relative weighting I wanted, and it's not a heuristic — it's the maximum-likelihood weighting, the statistically correct way to combine measurements of different precision. And crucially there's a +log σ_i sitting there, and it's exactly the anti-collapse term I argued I needed. Watch: if the optimiser tries the degenerate route of sending σ_i → ∞ to zero out the 1/(2σ_i²) coefficient and make a task free, the log σ_i term shoots off to +∞ and punishes it. The likelihood will not let me declare a task infinitely noisy and walk away. So the same probabilistic model that gives me inverse-variance weighting *automatically* supplies the regulariser that the bare learnable-weighted-sum was missing. The collapse I hit a moment ago is structurally impossible here. Let me confirm the balance is sane by actually setting the gradient w.r.t. σ_i to zero: ∂L/∂σ_i = −L_i(W)/σ_i³ + 1/σ_i = 0 gives σ_i² = L_i(W), so at the optimum the learned variance tracks the current value of that task's loss — a big-loss task is assigned a big σ (low weight), a well-fit task a small σ (high weight). The fixed point is exactly "weight each task by the inverse of how badly you're currently doing on it," which is sensible, and it can't run to zero or infinity because the two terms pull against each other.

I'd better make sure I can optimise σ stably, because there are a couple of landmines. σ appears as 1/σ² in the loss, so if σ ever wanders to zero during training I divide by zero and blow up; and σ is a variance scale, so it's constrained to be positive, which is awkward for an unconstrained optimiser like SGD that will happily step a parameter negative. Both problems vanish with one reparameterisation: don't learn σ, learn its log-variance, s := log σ². Then σ² = exp(s), and 1/σ² = exp(−s), and log σ = (1/2)log σ² = s/2. The parameter s ranges over all of ℝ, so SGD can step it freely; exp(−s) is always strictly positive, so there's never a divide-by-zero; and exp(·) is smooth, so the gradients are clean. In terms of s the per-task contribution for a regression task is

  (1/2)exp(−s_i)L_i(W) + (1/2)s_i,

and the precision weight exp(−s_i) is manifestly positive for any real s_i. That's the stable variable I'll actually train.

Now the harder half. My tasks aren't all regression — one head is a classifier, and its loss is cross-entropy, −log Softmax(f^W(x))_c, not a squared error. I can't just paste a Gaussian σ in front of a cross-entropy; I need the σ to enter the *classification* likelihood in a way that plays the same role. So what's the analogue of "scaling the observation noise" for a softmax? The softmax is a Boltzmann (Gibbs) distribution over classes with the logits as energies, and a Boltzmann distribution has a natural scale knob: temperature. Scale the logits by 1/σ² before the softmax,

  p(y | f^W(x), σ) = Softmax( (1/σ²) f^W(x) ),

so σ² plays the role of temperature. Large σ² (hot) flattens the distribution toward uniform — high uncertainty, the model is unsure which class; small σ² (cold) sharpens it toward a one-hot — high confidence. That's exactly the uncertainty interpretation I want, measured as the entropy of the resulting distribution, and it gives σ the same meaning across regression and classification: a noise/uncertainty scale. Good — so σ enters classification through the temperature.

Let me derive the loss this induces and see whether a clean +log σ regulariser drops out the way it did for regression, because if it doesn't this whole unification falls apart. The log-likelihood of the true class c under the temperature-scaled softmax is

  log p(y = c | f^W(x), σ) = (1/σ²) f_c^W(x) − log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ).

The negative of that is my classification term. I want to relate it to the *unscaled* cross-entropy L_2(W) = −log Softmax(f^W(x))_c = −f_c^W(x) + log Σ_{c'} exp(f_{c'}^W(x)), because that's the loss the network actually computes and the quantity I want σ to weight. From the cross-entropy, f_c^W(x) = log Σ_{c'} exp(f_{c'}^W(x)) − L_2(W). Substitute into −log p:

  −log p = −(1/σ²) f_c^W(x) + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) )
         = (1/σ²)[ L_2(W) − log Σ_{c'} exp(f_{c'}^W(x)) ] + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) )
         = (1/σ²) L_2(W) + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) − (1/σ²) log Σ_{c'} exp(f_{c'}^W(x))
         = (1/σ²) L_2(W) + log [ Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) / ( Σ_{c'} exp(f_{c'}^W(x)) )^{1/σ²} ].

So I get the inverse-variance weight (1/σ²) on the cross-entropy, exactly parallel to the regression case — good, that's the weighting I wanted — but the regulariser came out as this ugly log-ratio of two sums, log[ Σ exp((1/σ²)f_{c'}) / (Σ exp f_{c'})^{1/σ²} ], not a tidy log σ. That's a function of the logits f^W(x), so it couples the regulariser to the data and to the network output, which is both messy to optimise and conceptually wrong — the penalty for being uncertain about a task shouldn't depend on the particular image. I want it to reduce to log σ like the regression term. So when does that bracket simplify? Look at the numerator and denominator: Σ_{c'} exp((1/σ²)f_{c'}) versus (Σ_{c'} exp f_{c'})^{1/σ²}. If I make the explicit approximation

  (1/σ) Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) ≈ ( Σ_{c'} exp( f_{c'}^W(x) ) )^{1/σ²},

then the ratio inside the log is ≈ σ, and the whole regulariser collapses to log σ. When is that approximation any good? Rearrange it: the right side is the left's sum without the 1/σ prefactor and with the (1/σ²)-power moved outside — these two expressions of the log-sum-exp coincide exactly when σ → 1, because at σ = 1 both sides are literally Σ_{c'} exp(f_{c'}). So the approximation is anchored at the equal-temperature point and degrades smoothly as σ departs from 1. Trading the exact log-ratio for log σ buys me a *parallel* objective across regression and classification, with the same anti-collapse log σ penalty on every task — and it removes the data-dependent input x from the regulariser, which is what I wanted on principle. With the approximation, the classification term is

  (1/σ_2²) L_2(W) + log σ_2.

Now I can write the mixed objective for one continuous output y_1 (Gaussian) and one discrete output y_2 (softmax) and see the whole thing line up:

  L(W, σ_1, σ_2) = (1/2σ_1²) L_1(W) + (1/σ_2²) L_2(W) + log σ_1 + log σ_2,

with L_1(W) = ||y_1 − f^W(x)||² the Euclidean loss and L_2(W) = −log Softmax(y_2, f^W(x)) the unscaled cross-entropy. Both tasks are weighted by their inverse variance, both are regularised by log σ_i, and the construction extends to any number of continuous and discrete outputs by just adding terms. One asymmetry to keep honest about: the regression coefficient is 1/(2σ_1²) and the classification coefficient is 1/σ_2². The factor of two is real — it comes from the Gaussian, whose NLL carries the 1/2 in (1/2σ²)||·||², whereas the temperature scaling puts the full 1/σ² on the logits with no 1/2. So the derivation gives the task-type constants; the small implementation I want needs one rule that can consume any already-reduced task loss. In the log-variance variable s = log σ², the regression term is (1/2)exp(−s)L + (1/2)s, and multiplying that scalar term by 2 gives exp(−s)L + s without moving its optimum in s. For the classifier, after the σ → 1 approximation, the same precision term exp(−s)L appears with a logarithmic scale penalty. For a compact module, I choose the canonical uniform log-variance convention: one scalar per task and the term exp(−s_i)L_i + s_i, while remembering that the probabilistic derivation explains the regression/classification factor-of-two asymmetry before this coding convention is chosen.

There's a reason to expect this to avoid the weakness of a grid search, not merely automate it. A grid search is limited by its resolution — it samples the weighting space coarsely, and the good band is narrow, so it can easily step over the best point, while the learned σ moves continuously by gradient descent and isn't quantised. And more importantly, the grid weight is *static* for the whole run, but the learned σ is *dynamic*. Early in training the model is terrible at every task, all the losses are large, so every σ_i is large (recall the fixed point σ_i² = L_i) and the weighting is roughly even; as the model gets good at an easy task, that task's loss drops, its σ drops, and its weight *rises* — the schedule of relative weights evolves over training in a way no single fixed grid point can. A static weighting is forced to compromise across the whole trajectory; the learned one adapts to wherever training currently is. I'd want to validate this empirically, but the mechanism is clear from the σ_i² = L_i fixed point alone.

I should sanity-check robustness to where I start s, because if the answer depended sensitively on the initial log-variance I'd just be trading one tuning problem for another. The objective in s is, per task, (1/2)exp(−s)L + s/2 (regression) — differentiate: ∂/∂s = −(1/2)exp(−s)L + 1/2, zero at exp(−s) = 1/L, i.e. s = log L, the same σ² = L fixed point. The second derivative is (1/2)exp(−s)L > 0, so it's strictly convex in s with a single minimum. The implemented term exp(−s)L + s has the same minimiser and second derivative exp(−s)L > 0. Practically this means I can initialise s = 0 — that's σ² = 1, exp(−s) = 1, every task weighted equally, the most neutral possible start, no preference baked in — and let the optimisation move each scalar toward its fixed point without another hand-tuned hyperparameter.

Now let me put it in the code I'd actually ship, filling the one empty slot in the multi-task harness — the combination rule and its trainable state. The state is one log-variance scalar per task, initialised at zero and registered as a Parameter so the optimiser trains it jointly with the network; the forward computes Σ_i exp(−s_i) L_i + s_i:

```python
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty weighting of K task losses.

    Learns one log-variance s_i = log(sigma_i^2) per task. Each task loss is
    weighted by its precision exp(-s_i) and regularized by the log-variance
    term + s_i. exp(-s_i) > 0 always (no divide-by-zero); s_i is
    unconstrained so plain SGD can step it; the + s_i term forbids the
    degenerate sigma -> inf (weight -> 0) collapse a bare learnable weight has."""

    def __init__(self, num_tasks=2):
        super().__init__()
        # s_i = log(sigma_i^2), init 0  ->  sigma_i^2 = 1  ->  equal weighting.
        # The optimum is reached from a wide range of inits (convex in s), so 0
        # is a safe neutral start. Registered as a Parameter => trained by SGD.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        losses = [fine_loss, coarse_loss]
        total = 0.0
        for i in range(len(losses)):
            precision = torch.exp(-self.log_vars[i])   # 1/sigma_i^2, inverse-variance weight
            total = total + precision * losses[i] + self.log_vars[i]  # weighted loss + log-var regularizer
        return total
```

At the fixed point of this implemented term, −exp(−s_i)L_i + 1 = 0, so each task's learned variance equals its current loss. The weighting is "down-weight whatever you're currently bad at," and because the harness passes already-reduced task losses, the uniform exp(−s)·L + s shape is the simple log-variance convention: the likelihood derivation explains the exp(−s) precision, and the log-variance penalty is the anti-collapse term.

Pulling the chain together: I started stuck with a hand-weighted sum of incommensurable task losses where the good weighting is a narrow, expensive-to-find band that depends on units and noise. Making the weights bare learnable parameters collapsed them to zero, because nothing penalised turning a task off. Reading each loss as a negative log-likelihood put a per-task observation-noise σ in front of it, and maximising the joint Gaussian likelihood produced inverse-variance weighting 1/(2σ_i²) *plus* a log σ_i term — and that log σ_i is precisely the anti-collapse penalty the bare weighted sum lacked, so the same probabilistic model that weights the tasks also keeps the weights finite. Reparameterising to the log-variance s = log σ² made it numerically stable and unconstrained. Extending to classification through a temperature-scaled softmax gave the same inverse-variance weight, and a σ→1 approximation collapsed the messy log-sum-exp regulariser back to a clean log σ, with the real regression/classification coefficient asymmetry still visible in the derivation. The implementation convention is the uniform canonical term Σ_i exp(−s_i)L_i + s_i, with log_vars initialised to zero. The fixed point σ_i² = L_i shows the weighting tracks each task's current difficulty and is dynamic over training; convexity in s makes it robust to initialisation, so s = 0 (equal weights) is a safe start. And the whole thing drops into the existing multi-task loop as one log-variance Parameter per task and a two-line forward, trained jointly with the network by the same optimiser.
