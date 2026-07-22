Let me start from what actually hurts. I have one shared encoder feeding several heads — say a depth regressor and a semantic classifier, or a fine-class head and a coarse-class head — and I train them together because the shared representation is cheaper at test time and because the auxiliary task ought to regularise the features and help the one I care about. To train it I have to collapse the per-task losses into a single scalar, and the only thing everyone does is add them up with weights: L_total = Σ_i w_i L_i. And the weights are killing me. The losses aren't even in the same units — depth error is in meters-squared, cross-entropy is dimensionless nats, the instance-vector loss is in pixels — so their raw magnitudes can differ by orders of magnitude, and the "right" relative weight depends on those arbitrary units and on how noisy each task's labels are. When I sweep the weight of a two-task model across its range I see the same picture every time: each task is best somewhere in the interior, not at its single-task extreme, the two interiors don't line up, and there's a narrow compromise band where both tasks beat their single-task baselines, flanked on both sides by regions where pushing the weight further wrecks one task. Finding that band is a full training run per grid point — days each — and the search blows up combinatorially as I add tasks. So for three or more tasks I'm stuck either guessing uniform weights, which lands me outside the good band, or burning a week on grid search. I want the relative weighting to be set automatically and to adapt to the data.

The obvious first thing to try: just make the w_i trainable parameters and let SGD set them. Add w_1, w_2 as learnable scalars to L_total = w_1 L_1 + w_2 L_2 and optimise everything jointly. Let me stare at the gradient before I commit to this. ∂L_total/∂w_i = L_i, and L_i is a loss — it's nonnegative. So the gradient on every w_i is nonnegative, gradient descent steps each w_i *down*, and I can't see anything in this objective that pushes back. Follow it to its conclusion: the global optimum over the w_i is w_1 = w_2 = 0, where L_total = 0 identically and the network gets no gradient at all and learns nothing. So this doesn't work — and the failure isn't a tuning artefact, it's the objective doing exactly what I asked. There's no cost anywhere for declaring a task unimportant, so it declares every task unimportant. Whatever scheme learns the weights has to carry a term that resists shrinking them. That's the real constraint I take away from this dead end: I need a *principled* coupling between "how much I down-weight a task" and "a penalty for doing so," not a free knob.

Where would such a coupling come from naturally? I keep coming back to the fact that most of these losses are negative log-likelihoods in disguise. The depth L2 loss ||y − f^W(x)||² is, up to constants, the NLL of a Gaussian observation model p(y | f^W(x)) = N(f^W(x), σ²): write out −log N and you get (1/2σ²)||y − f^W(x)||² + (1/2)log σ² + const. In ordinary single-task training I treat σ as a fixed constant, fold it into the learning rate, and never write it down — for one task a constant noise level is just an overall rescale of the loss. But here's the thing: that constant σ is *exactly* a per-task scale. A task with noisier labels, or with a loss measured on a bigger numerical scale, has a bigger σ. And σ doesn't have to be fixed — I can make it a free parameter and learn it. This is the move from heteroscedastic regression: Nix and Weigend, back in 1994, trained a network with a mean head and a variance head and learned the noise by maximising the Gaussian likelihood, minimising Σ_i ||y_i − μ(x_i)||²/σ²(x_i) + log σ²(x_i). They never needed "uncertainty labels" — the likelihood couples σ to the residuals, so it's recovered implicitly. Their σ depends on the input x, which is the heteroscedastic case, but I don't need input-dependence here. I want one constant σ_i per task — the homoscedastic, task-dependent version — because what varies across my problem isn't the noise from pixel to pixel, it's the noise and the scale from *task* to *task*.

So let me write down the joint likelihood for two regression outputs and turn the crank, treating each σ_i as a learnable parameter alongside the weights W, and see what objective falls out. If the two outputs are conditionally independent given the shared features, the likelihood factorises:

  p(y_1, y_2 | f^W(x)) = N(y_1; f^W(x), σ_1²) · N(y_2; f^W(x), σ_2²).

Minimise the negative log of that. Each Gaussian contributes its −log:

  L(W, σ_1, σ_2) = −log p(y_1, y_2 | f^W(x))
                 ∝ (1/2σ_1²)||y_1 − f^W(x)||² + (1/2σ_2²)||y_2 − f^W(x)||² + log σ_1 + log σ_2,

where I used −log N(y; μ, σ²) = (1/2σ²)||y − μ||² + (1/2)log σ² + const and (1/2)log σ_1² + (1/2)log σ_2² = log σ_1 + log σ_2, dropping the additive constants that don't affect the optimisation. Writing L_1(W) = ||y_1 − f^W(x)||² and L_2(W) = ||y_2 − f^W(x)||² for the bare per-task losses,

  L(W, σ_1, σ_2) = (1/2σ_1²)L_1(W) + (1/2σ_2²)L_2(W) + log σ_1 + log σ_2.

Now look at what fell out, because I didn't put it there by hand. The coefficient on each task's loss is 1/(2σ_i²) — an *inverse-variance weighting*. As σ_1 grows the weight on L_1 shrinks, and as σ_1 shrinks its weight grows: the noisier (or larger-scale) a task is, the less it's allowed to dominate the shared gradient. That's the adaptive relative weighting I wanted, and it isn't a heuristic I imposed — it's the maximum-likelihood weighting, the statistically standard way to combine measurements of different precision. And sitting next to it is a +log σ_i term. Is that the anti-collapse term the bare weighted sum was missing? I have to actually check whether it resists the degenerate direction rather than just hope it does. The dangerous move for the optimiser is σ_i → ∞, which sends the coefficient 1/(2σ_i²) → 0 and makes a task free. Trace what happens to the rest of the objective along that path: 1/(2σ_i²)·L_i → 0, but log σ_i → +∞. So the limit is +∞, not −∞ — the objective gets *worse*, unboundedly, exactly where the bare weighted sum got better. The likelihood will not let me declare a task infinitely noisy and walk away. Good: the same probabilistic model that gives me inverse-variance weighting also supplies the regulariser the bare sum lacked.

But "the limit is +∞" only rules out the extreme; I want to know where the objective actually settles, to be sure it doesn't settle somewhere silly like σ pinned at zero. So set the gradient w.r.t. σ_i to zero: ∂L/∂σ_i = −L_i(W)/σ_i³ + 1/σ_i = 0. Multiply through by σ_i³: −L_i(W) + σ_i² = 0, so σ_i² = L_i(W). That's a clean, interpretable fixed point — the learned variance equals the current value of that task's loss. A task the network is doing badly on (big L_i) gets a big σ (low weight); a task it fits well (small L_i) gets a small σ (high weight). And it's finite and strictly positive for any positive loss, so the optimiser lands in the interior, not at 0 or ∞. The two terms genuinely pull against each other: the precision term wants σ small (to weight the task heavily and drive L_i down), the log term wants σ large, and they balance at σ_i² = L_i. Let me put a couple of numbers through it to make sure the fixed point is where the objective actually bottoms out and not just a stationary point I've mislabelled. Minimising exp(−s)L + s in the log variable (which I'll switch to in a moment; same fixed point), for L = 4 the formula says the minimiser is at σ² = 4, i.e. s = log 4 = 1.386; scanning the scalar objective on a fine grid around there puts the numerical argmin at 1.386 to four decimals, with the second derivative exp(−s)L = 1 > 0 there. For L = 100 the formula gives σ² = 100, s = 4.605, and the grid argmin agrees at 4.605. The fixed point is a genuine minimum, it tracks the loss as advertised, and it's stable. So the collapse I hit with the bare weighted sum can't recur here, and I've checked that rather than asserted it.

I'd better make sure I can optimise σ stably, because there are a couple of landmines visible in that objective. σ appears as 1/σ² in the loss, so if σ ever wanders to zero during training I divide by zero and blow up; and σ is a variance scale, constrained to be positive, which is awkward for an unconstrained optimiser like SGD that will happily step a parameter negative. Both problems vanish with one reparameterisation: don't learn σ, learn its log-variance, s := log σ². Then σ² = exp(s), and 1/σ² = exp(−s), and log σ = (1/2)log σ² = s/2. The parameter s ranges over all of ℝ, so SGD can step it freely; exp(−s) is always strictly positive, so there's never a divide-by-zero; and exp(·) is smooth, so the gradients are clean. In terms of s the per-task contribution for a regression task is

  (1/2)exp(−s_i)L_i(W) + (1/2)s_i,

and the precision weight exp(−s_i) is manifestly positive for any real s_i. That's the stable variable I'll actually train. (This is also the variable I scanned above; the fixed point I derived as σ_i² = L_i reads here as s_i = log L_i, which is the convex minimum of (1/2)exp(−s)L + (1/2)s.)

Now the harder half. My tasks aren't all regression — one head is a classifier, and its loss is cross-entropy, −log Softmax(f^W(x))_c, not a squared error. I can't just paste a Gaussian σ in front of a cross-entropy; I need the σ to enter the *classification* likelihood in a way that plays the same role. So what's the analogue of "scaling the observation noise" for a softmax? The softmax is a Boltzmann (Gibbs) distribution over classes with the logits as energies, and a Boltzmann distribution has a natural scale knob: temperature. Scale the logits by 1/σ² before the softmax,

  p(y | f^W(x), σ) = Softmax( (1/σ²) f^W(x) ),

so σ² plays the role of temperature. Large σ² (hot) flattens the distribution toward uniform — high uncertainty, the model is unsure which class; small σ² (cold) sharpens it toward a one-hot — high confidence. That gives σ the same meaning across regression and classification: a noise/uncertainty scale, measurable here as the entropy of the resulting distribution. So σ enters classification through the temperature. The open question is whether the *regulariser* it induces matches the regression one, and I have no right to assume it does until I derive it.

Let me derive the loss this temperature scaling induces, because if a clean +log σ regulariser doesn't drop out the way it did for regression, the unification I'm hoping for doesn't actually exist. The log-likelihood of the true class c under the temperature-scaled softmax is

  log p(y = c | f^W(x), σ) = (1/σ²) f_c^W(x) − log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ).

The negative of that is my classification term. I want to relate it to the *unscaled* cross-entropy L_2(W) = −log Softmax(f^W(x))_c = −f_c^W(x) + log Σ_{c'} exp(f_{c'}^W(x)), because that's the loss the network actually computes and the quantity I want σ to weight. From the cross-entropy, f_c^W(x) = log Σ_{c'} exp(f_{c'}^W(x)) − L_2(W). Substitute into −log p:

  −log p = −(1/σ²) f_c^W(x) + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) )
         = (1/σ²)[ L_2(W) − log Σ_{c'} exp(f_{c'}^W(x)) ] + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) )
         = (1/σ²) L_2(W) + log Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) − (1/σ²) log Σ_{c'} exp(f_{c'}^W(x))
         = (1/σ²) L_2(W) + log [ Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) / ( Σ_{c'} exp(f_{c'}^W(x)) )^{1/σ²} ].

So I get the inverse-variance weight (1/σ²) on the cross-entropy, exactly parallel to the regression case — that part of the hope holds, and it's the weighting I wanted. But the regulariser came out as this log-ratio of two sums, log[ Σ exp((1/σ²)f_{c'}) / (Σ exp f_{c'})^{1/σ²} ], not a tidy log σ. It's a function of the logits f^W(x), so the penalty for being uncertain about a task would depend on the particular image — that's messy to optimise and conceptually off, since the cost of declaring a task noisy shouldn't ride on which image came up. So the unification is *not* automatic; it only happens if this bracket reduces to log σ. When would it? Stare at numerator and denominator: Σ_{c'} exp((1/σ²)f_{c'}) versus (Σ_{c'} exp f_{c'})^{1/σ²}. If I make the explicit approximation

  (1/σ) Σ_{c'} exp( (1/σ²) f_{c'}^W(x) ) ≈ ( Σ_{c'} exp( f_{c'}^W(x) ) )^{1/σ²},

then the ratio inside the log is ≈ σ and the regulariser collapses to log σ. Now I should be honest with myself about how good this approximation is rather than wave it through, because I'm about to build the whole unification on it. At σ = 1 (σ² = 1) both sides are literally Σ_{c'} exp(f_{c'}), so it's exact there — but "exact at one point" tells me nothing about the size of the error away from that point. Let me actually compute the error. Take a concrete 4-class logit vector, f = (2.0, 0.5, −1.0, 0.0), and compare the *true* regulariser log[ Σ exp((1/σ²)f) / (Σ exp f)^{1/σ²} ] against the approximation log σ = (1/2)log σ² across a range of σ²:

  σ² = 0.25 :  true −1.367,  log σ −0.693,  error −0.673
  σ² = 0.50 :  true −0.617,  log σ −0.347,  error −0.270
  σ² = 1.00 :  true  0.000,  log σ  0.000,  error  0.000
  σ² = 2.00 :  true +0.553,  log σ +0.347,  error +0.207
  σ² = 4.00 :  true +0.932,  log σ +0.693,  error +0.239
  σ² = 10.0 :  true +1.196,  log σ +1.151,  error +0.044

So the approximation is exact at σ² = 1 and the error stays under about 0.7 nat over this whole two-orders-of-magnitude span, smallest near σ = 1 and growing as σ pulls away — and it depends on the logits, so it isn't a universal bound, just what this vector gives. This is not negligible, and I want to be clear-eyed that I'm trading an exact (but data-coupled, image-dependent) regulariser for an approximate one. What do I get for the trade? A *parallel* objective across regression and classification with the same anti-collapse log σ penalty on every task, and — the part I actually care about on principle — a regulariser with the input x removed, so the cost of being uncertain about a task no longer depends on which image showed up. Given that the network spends most of training with the σ_i not wildly far from 1 (they're initialised there, and the fixed point σ_i² = L_i keeps them at the scale of typical losses), an error of a few tenths of a nat in the *regulariser* — not in the task loss itself, which still gets the exact (1/σ²) weight — is a price I'm willing to pay for the unification and the data-independence. With the approximation, the classification term is

  (1/σ_2²) L_2(W) + log σ_2.

Now I can write the mixed objective for one continuous output y_1 (Gaussian) and one discrete output y_2 (softmax) and see the whole thing line up:

  L(W, σ_1, σ_2) = (1/2σ_1²) L_1(W) + (1/σ_2²) L_2(W) + log σ_1 + log σ_2,

with L_1(W) = ||y_1 − f^W(x)||² the Euclidean loss and L_2(W) = −log Softmax(y_2, f^W(x)) the unscaled cross-entropy. Both tasks are weighted by their inverse variance, both are regularised by log σ_i, and the construction extends to any number of continuous and discrete outputs by adding terms. One asymmetry I should keep honest about: the regression coefficient is 1/(2σ_1²) and the classification coefficient is 1/σ_2². The factor of two is real — it comes from the Gaussian, whose NLL carries the 1/2 in (1/2σ²)||·||², whereas the temperature scaling puts the full 1/σ² on the logits with no 1/2. So the derivation hands me task-type-dependent constants, but the small module I want to ship takes already-reduced task losses and shouldn't have to know which kind each one is. In the log-variance variable s = log σ², the regression term is (1/2)exp(−s)L + (1/2)s, and multiplying that scalar by 2 gives exp(−s)L + s — which doesn't move its minimum in s, since ∂/∂s[(1/2)exp(−s)L + (1/2)s] = 0 and ∂/∂s[exp(−s)L + s] = 0 have the same solution s = log L. For the classifier, after the σ → 1 approximation, the same precision term exp(−s)L appears with a logarithmic scale penalty s. So for a compact module I adopt one uniform log-variance convention — one scalar per task and the term exp(−s_i)L_i + s_i — while noting that the regression/classification factor-of-two lives in the derivation, ahead of this coding convention.

There's a reason to expect this to beat a grid search rather than merely automate it, and I can argue it from the fixed point I already verified. A grid search samples the weighting space at fixed resolution, and the good band is narrow, so it can step over the best point; the learned σ moves continuously by gradient descent and isn't quantised. More importantly, a grid weight is *static* for the whole run, whereas the learned σ is *dynamic*: by σ_i² = L_i, early in training every loss is large, so every σ_i is large and the weighting is roughly even; as the model gets good at an easy task, that task's loss drops, its σ drops, and its weight *rises*. A single fixed grid point can't track that trajectory — it's forced to compromise across all of training — while the learned weighting follows wherever training currently is. I'd want to confirm the end-to-end win empirically, but the dynamic mechanism is already visible in the σ_i² = L_i fixed point I checked numerically above.

I should sanity-check robustness to where I initialise s, because if the answer depended sensitively on the starting log-variance I'd just be trading one tuning problem for another. The objective in s is, per task, (1/2)exp(−s)L + s/2 (regression). Differentiate: ∂/∂s = −(1/2)exp(−s)L + 1/2, zero at exp(−s) = 1/L, i.e. s = log L — the same σ² = L fixed point. The second derivative is (1/2)exp(−s)L > 0 everywhere, so the objective is strictly convex in s with a single global minimum and no other stationary points to get stuck in. The implemented term exp(−s)L + s has the same minimiser and second derivative exp(−s)L > 0. Strict convexity is what makes the initialisation safe: from any starting s, gradient descent runs monotonically downhill to the unique minimum. So I can initialise s = 0 — σ² = 1, exp(−s) = 1, every task weighted equally, the most neutral possible start, no preference baked in — and let the optimisation carry each scalar to its fixed point without another hand-tuned hyperparameter.

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
