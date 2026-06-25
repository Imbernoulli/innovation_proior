I start with the ordinary softmax classifier because the failure is already visible in the loss. For logits `z_k`, the model predicts `p_k = exp(z_k) / sum_j exp(z_j)`. With a one-hot target for class `y`, the loss is `-log p_y`, and the gradient with respect to logit `z_k` is `p_k - 1[k = y]`. That gradient is bounded, but the target it is pursuing is not a finite logit configuration. To make `p_y` equal to one, the gap `z_y - z_j` must go to infinity for every incorrect class `j`.

So the pressure I need to remove is not cross-entropy itself. Cross-entropy gives clean gradients and fast learning. The problem is the zero target on every incorrect class. Once an example is already classified correctly, the loss can still be reduced by pushing the correct logit farther from all the others. That creates overconfident probabilities, and it also leaves the final layer with very large margins that are slow to revise.

The most direct repair I can think of is to stop using a target that has zeros. I take a small mass `epsilon` away from the one-hot label and put it on a fixed prior `u` over classes:

`q'_k = (1 - epsilon) 1[k = y] + epsilon u_k`.

If I have no reason to prefer one incorrect class over another, the prior is uniform, `u_k = 1/K`. Then the target is `q'_y = 1 - epsilon + epsilon/K` for the correct class and `q'_j = epsilon/K` for each incorrect class. Now every class has positive target mass.

I want to know whether this actually removes the incentive to run the correct logit away to infinity, or whether it just shifts the optimum a little. Let me work a concrete case. Take `K = 3` and `epsilon = 0.1`, correct class first. The target is `q'_y = 1 - 0.1 + 0.1/3 = 0.9333`, and `q'_j = 0.1/3 = 0.0333` for each of the two wrong classes; these sum to one, as they should. Now I hold the two incorrect logits at zero and sweep the correct logit `z_y`, computing the smoothed loss `-(0.9333 log p_y + 0.0333 log p_1 + 0.0333 log p_2)` against the hard loss `-log p_y`:

```
z_y =  3.33:  smoothed = 0.291,  hard = 0.069
z_y = 10   :  smoothed = 0.667,  hard = 0.000091
z_y = 30   :  smoothed = 2.000,  hard ~ 0
z_y = 100  :  smoothed = 6.667,  hard ~ 0
```

This is the behavior I was hoping for, and it is sharper than I expected. The hard loss keeps falling toward zero as the gap grows, so a hard-label model is always rewarded for a bigger margin. The smoothed loss does the opposite past a point: it bottoms out and then climbs without bound, because the `-(epsilon/K) log p_j` terms blow up as the incorrect probabilities are crushed to zero. So a runaway gap is not merely unrewarded under the smoothed target; it is actively punished.

Where is the bottom? At the softmax fit `p = q'` the loss is minimized, and since the two incorrect targets are equal, their logits are equal up to a shared constant. The correct-vs-incorrect gap there is

`z_y - z_j = log(q'_y / q'_j) = log((1 - epsilon + epsilon/K) / (epsilon/K)) = log(1 + K(1 - epsilon)/epsilon)`.

For my numbers `log(0.9333 / 0.0333) = 3.332`, and the closed form `log(1 + 3 * 0.9 / 0.1) = log(28) = 3.332` agrees. To be sure that is the minimum and not just an algebraic fixed point, I scan `z_y` on a fine grid around it: the argmin sits at `3.332` exactly, matching the formula. So the smoothed target asks for a specific finite gap. It grows when `epsilon` shrinks and goes to infinity as `epsilon` goes to zero, which recovers the hard-label case in the limit. That is the mechanism I was after: the desired confidence is now finite.

The loss decomposition lets me see the same effect as an explicit penalty. Cross-entropy is linear in the target, so

`H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`.

Before I lean on this, I should check it is really an identity and not just a plausible rearrangement, since I am going to read meaning into the second term. I pick `K = 4`, `epsilon = 0.1`, logits `[2.0, 0.5, -1.0, 0.3]`, correct class first. Computing `-(1-epsilon) log p_y - epsilon * mean_k(-log p_k)` gives `0.5304`, and forming the smoothed target directly and taking `-sum_k q'_k log p_k` also gives `0.5304`. They match, so the decomposition holds and the per-example loss is exactly `(1 - epsilon)` times the ordinary NLL plus `epsilon` times the mean over classes of `-log p_k` — which is `H(u, p)` for uniform `u`. That mean-over-log-probs form is also what I would implement.

For uniform `u`, `H(u, p) = KL(u || p) + H(u)`, and `H(u)` is constant. I verify this on the same logits: `H(u, p) = 1.9254` while `KL(u || p) + H(u) = 1.9254`. So the smoothed loss is the hard-label loss plus an `epsilon`-weighted penalty for drifting away from the uniform prior.

Here I have to be careful about a tempting near-equivalence, because there is an existing tool that also discourages overconfidence: a confidence penalty that adds the negative entropy of the model's own output. It would be easy to call these the same regularizer, but they are not. Penalizing low output entropy adds `KL(p || u)` up to a constant; my term is `KL(u || p)`. The arguments are swapped, and for KL that is not cosmetic. `KL(u || p)` puts `u`'s mass — uniform, so nonzero everywhere — against `log(u_k / p_k)`, which diverges wherever `p_k` is driven to zero. `KL(p || u)` weights by `p_k`, so it stops caring about a class precisely as that class's probability vanishes. The smoothed target uses the direction that refuses to let any probability reach zero, which is exactly the divergence I saw numerically above. So I keep the soft-target form rather than restating it as a confidence penalty; they pull differently in the limit that matters.

That explains bounded confidence, but it does not yet tell me what happens inside the representation. I look at the last layer. Write the logit as `z_k = x^T w_k`, where `x` is the penultimate activation with a constant appended for the bias and `w_k` is the class template. The squared distance to a template is

`||x - w_k||^2 = x^T x - 2 x^T w_k + w_k^T w_k`.

Inside a softmax over classes, `x^T x` is common to all `k`, and the template norms are usually similar enough to treat `w_k^T w_k` as approximately class-independent. The varying term is `-2 x^T w_k`, so larger logits correspond approximately to smaller squared distance from `x` to the class template.

Now the target distribution has a geometric meaning. With hard labels, the loss mainly asks `x` to be closer to `w_y` than to the other templates by an ever-growing margin; it says little about the relative distances to the incorrect templates. With the softened target, the desired probabilities give a finite gap to the correct class and equal probabilities for all incorrect classes. Equal incorrect probabilities encourage equal incorrect logits, and under the distance picture that means `x` is encouraged to be equally distant from the incorrect templates while remaining closer to the correct template by the finite gap above.

So I expect the penultimate activations for each class to tighten around their own template and arrange themselves in a more regular geometry relative to other class templates. For three classes, the diagnostic picture should be especially clear: project examples onto the plane through the three templates. Hard-label training should allow broad clusters and large scales; the softened target should create tighter clusters with a more regular triangular arrangement. I have not run this, so I hold it as a prediction I would want to confirm by actually projecting trained activations; for semantically similar classes I would watch whether the continuous similarity structure survives or gets flattened into the equal-distance constraint.

This gives me a calibration prediction. If the logit gaps stop growing without bound, the output probabilities should stop saturating as much. A hard-label model can be calibrated after training by temperature scaling, which divides logits by a positive scalar and leaves class predictions unchanged. A softened-target model may do part of that work during training because the target itself asks for finite confidence. Expected calibration error and reliability diagrams are the right tests: compare confidence bins to empirical accuracy. I would not assume the gain transfers automatically — it is a hypothesis the reliability diagram either supports or refutes.

Translation makes this calibration question matter operationally. Beam search does not merely ask for the argmax at each position; it ranks sequence hypotheses using next-token probabilities. If the probabilities are overconfident or poorly shaped, the search can prefer bad partial sequences. I therefore expect softened targets to help a decoder when they improve calibration. But I should not reduce the whole translation story to calibration: a model can have better BLEU while having worse negative log-likelihood, and temperature scaling a hard-label model may improve calibration without fully matching the softened-target BLEU. Calibration can explain part of the gain without explaining all of it.

The same geometry also warns me about a cost. Distillation relies on the teacher's relative probabilities for incorrect classes. A teacher saying "this digit is mostly a 3 but somewhat like an 8" gives a student information that the hard label cannot. If my training target makes all incorrect classes equally likely for each example of a class, then the teacher is encouraged to erase exactly those example-specific relative similarities. A more accurate teacher could then become a worse teacher for distillation.

I can try to phrase that loss of structure as mutual information. Let `X` be the training-example index and let `Y` be a scalar logit difference for two classes after random data augmentation. For each example `x`, collect Monte Carlo samples `y_{x,l} = f(d_l(z_x))`, estimate a mean `mu_x = (1/L) sum_l y_{x,l}`, and use a pooled variance `sigma^2 = (1/(NL)) sum_{x,l} (y_{x,l} - mu_x)^2`. With a shared-variance Gaussian approximation,

`I(X; Y)` can be estimated by averaging

`-(y_{x,l} - mu_x)^2/(2 sigma^2) - log((1/N) sum_{x'} exp(-(y_{x,l} - mu_{x'})^2/(2 sigma^2)))`

over examples and augmentations. The normalization constants cancel between the conditional Gaussian density and the mixture density. I want a sanity check on the scale this estimator should report. Suppose smoothing collapses the representation so that every example of class A lands at one logit-difference value and every example of class B at another, well separated, with two classes equally represented. Then knowing `y` tells me the class but nothing about which individual example produced it. The example index `X` ranges over `N` values but `Y` carries only a one-bit class label, so `I(X; Y)` should saturate near `log 2` nats. That is the number I would expect this estimator to approach in the collapsed case, and a value well below `log 2` would instead signal that example-specific structure survived. So the estimator gives me a concrete dial for the dark-knowledge cost rather than a vague worry.

Putting the pieces together: I use the softened target when I want finite confidence, regularization, and better-calibrated probabilities. I stay cautious when the downstream task needs the teacher's fine-grained wrong-class structure, especially distillation and transfer from final-layer features. The same equal-distance pressure that gives tidy clusters and calibrated logits is also what can destroy the dark knowledge a student would otherwise learn — so the decision is task-dependent, not unconditional.
