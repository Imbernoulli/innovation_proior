Let me start from a discomfort I have with how I pick a classification loss. I reach for cross-entropy by default, `-log(P_t)` where `P_t` is the softmax probability my model puts on the true class, and when the data is imbalanced I reach for focal loss, `-(1-P_t)^γ log(P_t)`. But these are just two fixed curves. A loss is, in principle, *any* differentiable map from predictions and labels to a scalar — an enormous space — and I have been picking two points in it out of habit, with no argument that either is the right curve for the architecture and dataset in front of me. I would like to open up that space into a small, interpretable family and see cross-entropy and focal loss as particular members. The trouble is the space is far too big to search directly; I need a parameterization that is low-dimensional enough to tune but rich enough to contain the losses I already trust.

So let me stare at cross-entropy itself and see if it suggests its own family. `-log(P_t)`. The logarithm has a series I know: `-log(x) = (1-x) + ½(1-x)² + ⅓(1-x)³ + …`, valid for `x` in `(0,1]`, which is exactly the range a probability lives in. Substitute `x = P_t`:

  `L_CE = Σ_{j≥1} (1/j)(1-P_t)^j = (1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + …`

That is more than an algebraic curiosity. It says cross-entropy is a *weighted sum of polynomial bases* `(1-P_t)^j`, and the weights are `α_j = 1/j`. Now the family writes itself: `L = Σ_j α_j (1-P_t)^j`; with nonnegative coefficients, it is monotone decreasing in `P_t`, and cross-entropy is just the one member whose coefficients happen to be `1/j`. The bases have a clean meaning too — `(1-P_t)` is the distance from a perfect prediction, so `(1-P_t)^j` is that distance raised to the `j`th power, and the loss is an ensemble of these distance-powers. Each `α_j` is a dial I can turn to emphasize or de-emphasize one power of the prediction error.

Does focal loss live in here? Apply the same substitution to `-(1-P_t)^γ log(P_t)`. The `(1-P_t)^γ` factor pulls straight through the series:

  `L_FL = (1-P_t)^γ Σ_j (1/j)(1-P_t)^j = Σ_j (1/j)(1-P_t)^{j+γ}`.

So focal loss is cross-entropy with every term's *power* bumped up by `γ`. In coefficient space, that is a *horizontal* shift of the whole `1/j` profile: the coefficient that used to sit on power `j` now sits on power `j+γ`. Focal loss and cross-entropy are the same profile, slid sideways. That is a genuinely new way to see the relationship, and it immediately tells me focal loss has exactly *one* degree of freedom — how far to slide — and that it can only slide in one direction. If the right move for a given task is not a horizontal slide, focal loss cannot make it.

Before I start turning dials, I want to understand the gradient, because that is what actually trains the network, not the printed loss value. Differentiate cross-entropy term by term: `d/dP_t (1/j)(1-P_t)^j = (1/j)·j·(1-P_t)^{j-1}·(-1) = -(1-P_t)^{j-1}`. The `1/j` coefficient *exactly cancels* the power `j` that comes down. So

  `-dL_CE/dP_t = Σ_{j≥1} (1-P_t)^{j-1} = 1 + (1-P_t) + (1-P_t)² + …`,

a clean geometric series. Two things jump out. The leading term is the constant `1`, completely independent of `P_t`: every example, no matter how confidently right or wrong, contributes the same `1` to this part of the gradient. And the higher-order terms `(1-P_t)^{j-1}` are strongly suppressed once `P_t` is close to 1, but are very much alive when `P_t` is near zero. That constant leading term is suspicious. Its gradient contribution, summed over a batch, is just the *count* of examples, so it pushes hardest on whatever class is most numerous, which is exactly the wrong instinct on imbalanced data. Focal loss points in the opposite direction: differentiating the shifted series gives the leading push `(1+γ)(1-P_t)^γ`, so the `P_t`-independent constant is gone and confident examples fade out.

Now the obvious idea: if the coefficients are dials, maybe the `1/j` assignment is suboptimal and I should just retune all of them. Let me think about which ones I can afford to touch. The cheapest move people have already tried is to *truncate* — keep the first `N` terms, `L_Drop = Σ_{j=1}^N (1/j)(1-P_t)^j`, zeroing every coefficient above order `N`. In coefficient space that is a *vertical* push: drive a set of `α_j` down to zero. The appeal is that low-order truncation makes the loss behave like mean-absolute-error, which is robust to label noise. But let me check what it costs on a clean, many-class problem before I believe it. Picture the start of training on ImageNet-1K: the model is near chance, so `P_t ≈ 0.001` for the true class. Look at the gradient term of order `j = 500`: its coefficient is `(1-P_t)^{499} = 0.999^{499} ≈ 0.6`. That is *not* small. Early in training, when `P_t` is tiny, the high-order terms still carry real gradient because `(1-P_t)` is close to 1 and its high powers have not yet decayed. So truncating the tail throws away signal the model needs to escape the near-chance regime. The empirical fact that lines up with this — that you need to keep hundreds of terms before truncation stops hurting, and that no learning-rate retune recovers the gap — rules out truncation for high-class-count classification. I cannot reduce the number of coefficients by chopping the tail.

Let me make the tail argument clean instead of relying only on that one number. If I cut after `N` terms, the missing tail is `R_N(p) = L_CE - L_Drop = Σ_{j=N+1}^∞ (1/j)(1-p)^j`. On any interval `p ∈ [δ, 1]`, I can upper-bound the loss residual by dropping the `1/j` factor: `|R_N(p)| ≤ Σ_{j=N+1}^∞ (1-p)^j = (1-p)^{N+1}/p ≤ (1-δ)^N/δ`. The derivative residual is just as important for training: `|R'_N(p)| = Σ_{j=N}^∞ (1-p)^j = (1-p)^N/p ≤ (1-δ)^N/δ`. So if I want both residuals below a tolerance `ζ`, I need `(1-δ)^N/δ < ζ`, or `N > log_{1-δ}(ζδ)`. As `δ` moves toward zero, the cutoff needed for uniform closeness explodes. That is exactly the many-class early-training regime, so the tail is not decorative.

So the tail is load-bearing and must be left alone. That kills naive truncation, and it also kills the dream of retuning *all* the coefficients: if hundreds of terms matter, tuning hundreds of `α_j` is a search space I cannot grid over, and black-box search over that many dials is what the meta-learning approaches do, only feasible when a handful of parameters suffice — which is the few-class regime, not the thousand-class one. I am stuck between "the tail matters, so keep it" and "I cannot tune the whole thing."

The way out is to keep the entire cross-entropy profile intact — every `1/j`, tail included — and *perturb* only the first few coefficients on top of it. Write the leading `N` coefficients as `1/j + ε_j` and leave the rest at `1/j`:

  `L_Poly-N = Σ_{j=1}^N (1/j + ε_j)(1-P_t)^j + Σ_{j>N} (1/j)(1-P_t)^j = -log(P_t) + Σ_{j=1}^N ε_j (1-P_t)^j`.

The infinite tail collapses back into `-log(P_t)`, so this is just cross-entropy *plus* a short polynomial correction `Σ_{j=1}^N ε_j (1-P_t)^j`. Now I have only `N` dials, the tail is preserved for free, and `ε_j = 0` recovers cross-entropy exactly. This is a search I can actually run.

How small can `N` be? Here is where the gradient observation pays off. The diagnostic I care about is how much of cross-entropy's total training push comes from the first polynomial versus all the rest. Once training has moved away from the near-chance regime, the first term `(1-P_t)` contributes more than half of the gradient for most of the remaining steps, so the coefficient most worth perturbing is the very first one. If I want the smallest useful search, I should start with `N = 1` and perturb the single leading coefficient.

  `L_Poly-1 = (1 + ε₁)(1-P_t) + ½(1-P_t)² + ⅓(1-P_t)³ + … = -log(P_t) + ε₁(1-P_t)`.

One extra hyperparameter, one extra term added to cross-entropy. Now which sign of `ε₁`, and how big? Let me reason from the training push, `-dL/dP_t`, so the sign is explicit. Adding `ε₁(1-P_t)` changes it to `(1 + ε₁) + (1-P_t) + (1-P_t)² + …`; the actual derivative with respect to `P_t` is the negative of this quantity. With `ε₁ > 0` I am *strengthening* the constant push that survives all the way until `P_t` reaches 1, i.e. I keep pressing the model toward higher target-class confidence even when it is already fairly confident. The intuition that fits the balanced-classification case is that plain cross-entropy can leave the confidence pressure too weak on moderately-correct examples, and a positive `ε₁` restores some of that pressure, nudging predictions to be more decisive. On a balanced dataset, where there is no majority class to over-serve, that stronger confidence push is the helpful direction. The picture also explains why this is the opposite of focal loss: focal loss suppresses the easy-example push by removing the `P_t`-independent leading constant; `ε₁ > 0` adds to that constant because the balanced-classification failure mode points toward under-confidence, not over-confidence. Same coefficient axis, opposite directions, each tied to the task's confidence failure mode rather than a universal sign. If the failure mode is an over-confident, over-served majority, then the coefficient can move negative and reduce the leading push.

For magnitude, this is a one-dimensional grid search and there is no closed form — the right `ε₁` depends on the dataset's class count, balance, and confidence behavior. For image classification I will take `ε₁ = 2.0` as the default: it raises the leading push from `1` to `3`, large enough to matter while still leaving the entire higher-order cross-entropy profile in place. I can still sweep the single knob when the task changes.

Let me also sanity-check the connection back to regression, because it tells me I have not done anything exotic. With `y = 1` the effective target probability, the base `(1-P_t)` is `(y - P_t)`, so `(1-P_t)^j = (y - P_t)^j`, and the whole family is a weighted ensemble of prediction-to-target distances raised to integer powers. Cross-entropy is one particular weighting of those distances; focal loss is another; Poly-1 is cross-entropy with the first distance term up-weighted. Nothing here leaves the world of "penalize the gap between prediction and label" — I have only made the *weighting of that gap* tunable, with a single dial that has a clear gradient meaning.

Now the implementation, and it is almost nothing. I already compute cross-entropy. `P_t` is the softmax probability at the true class — gather it out of `softmax(logits)`. Then add `ε₁ (1-P_t)`, averaged over the batch. That is the entire change: one line on top of the standard criterion.

```python
import torch
import torch.nn.functional as F


def poly1_cross_entropy(logits, targets, epsilon=2.0):
    """PolyLoss Poly-1: cross-entropy with the leading polynomial coefficient
    perturbed by epsilon.

        L = -log(P_t) + epsilon * (1 - P_t)
          = (1 + epsilon)(1 - P_t) + 1/2 (1 - P_t)^2 + ...

    Recovers cross-entropy at epsilon = 0. Keeping epsilon >= -1 leaves the
    leading coefficient nonnegative. P_t is the softmax probability the model
    assigns to the ground-truth class.
    """
    # cross-entropy term: -log(P_t)
    ce = F.cross_entropy(logits, targets, reduction="none")          # [B]
    # P_t: softmax probability at the target class
    p_t = torch.softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1)                          # [B]
    # leading-coefficient perturbation: epsilon * (1 - P_t)
    poly1 = ce + epsilon * (1.0 - p_t)                               # [B]
    return poly1.mean()
```

So I end up with a narrow path through the enormous loss-function space. Expanding `-log(P_t)` in the Mercator series exposes cross-entropy as a polynomial in `(1-P_t)` with fixed coefficients `1/j`, and it puts focal loss in the same family as a horizontal shift of that profile whose training push has no constant leading term. The training push from cross-entropy is the geometric series `-dL_CE/dP_t = Σ (1-P_t)^{j-1}`, whose leading term is a constant `1` that carries more than half the gradient through most of training; truncating the tail to zero fails on many-class data because the high-order terms are still important early when `P_t ≈ 0`, and tuning all coefficients is an infeasible search. I keep the whole cross-entropy profile and perturb only the leading coefficient, `L = -log(P_t) + ε₁(1-P_t)`, a one-knob, one-line correction. A positive `ε₁` strengthens the surviving confidence-pressure that plain cross-entropy lets decay too soon, which helps on balanced data and is exactly the opposite of focal loss's move, with `ε₁ = 2.0` the image-classification default.
