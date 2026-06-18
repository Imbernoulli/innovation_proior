I start with the ordinary softmax classifier because the failure is already visible in the loss. For logits `z_k`, the model predicts `p_k = exp(z_k) / sum_j exp(z_j)`. With a one-hot target for class `y`, the loss is `-log p_y`, and the gradient with respect to logit `z_k` is `p_k - 1[k = y]`. That gradient is bounded, but the target it is pursuing is not a finite logit configuration. To make `p_y` equal to one, the gap `z_y - z_j` must go to infinity for every incorrect class `j`.

So the pressure I need to remove is not cross-entropy itself. Cross-entropy gives clean gradients and fast learning. The problem is the zero target on every incorrect class. Once an example is already classified correctly, the loss can still be reduced by pushing the correct logit farther from all the others. That creates overconfident probabilities, and it also leaves the final layer with very large margins that are slow to revise.

The most direct repair is to stop using a target that has zeros. I take a small mass `epsilon` away from the one-hot label and put it on a fixed prior `u` over classes:

`q'_k = (1 - epsilon) 1[k = y] + epsilon u_k`.

If I have no reason to prefer one incorrect class over another, the prior is uniform, `u_k = 1/K`. Then the target is `q'_y = 1 - epsilon + epsilon/K` for the correct class and `q'_j = epsilon/K` for each incorrect class. Now every class has positive target mass. If the correct logit tries to run infinitely far above the rest, the incorrect probabilities go to zero, and the terms `-(epsilon/K) log p_j` make the loss diverge. The runaway gap is no longer rewarded.

The exact finite gap is useful to write down. At the ideal softmax fit, `p = q'`. All incorrect classes have equal target probability, so their logits are equal up to a shared constant. The correct-vs-incorrect gap is

`z_y - z_j = log(q'_y / q'_j) = log((1 - epsilon + epsilon/K) / (epsilon/K)) = log(1 + K(1 - epsilon)/epsilon)`.

That is the constant the loss is now asking for. It grows when `epsilon` shrinks, and it becomes infinite as `epsilon` goes to zero, recovering the hard-label case. This is the first real mechanism: the smoothed target makes the desired confidence finite.

The loss decomposition says the same thing from another angle. Cross-entropy is linear in the target, so

`H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`.

For uniform `u`, `H(u, p) = KL(u || p) + H(u)`, and `H(u)` is constant. So I am still training on the hard label, but I add a penalty for drifting too far from the uniform prior. This is close to a confidence penalty, but not identical: penalizing low output entropy is `KL(p || u)` up to a constant, while this term is `KL(u || p)`. The direction matters because `KL(u || p)` is intolerant of probabilities driven all the way to zero.

That explains bounded confidence, but it does not yet explain what happens inside the representation. I look at the last layer. Write the logit as `z_k = x^T w_k`, where `x` is the penultimate activation with a constant appended for the bias and `w_k` is the class template. The squared distance to a template is

`||x - w_k||^2 = x^T x - 2 x^T w_k + w_k^T w_k`.

Inside a softmax over classes, `x^T x` is common to all `k`, and the template norms are usually similar enough to treat `w_k^T w_k` as approximately class-independent. The varying term is `-2 x^T w_k`, so larger logits correspond approximately to smaller squared distance from `x` to the class template.

Now the target distribution has a geometric meaning. With hard labels, the loss mainly asks `x` to be closer to `w_y` than to the other templates by an ever-growing margin; it says little about the relative distances to the incorrect templates. With the softened target, the desired probabilities give a finite gap to the correct class and equal probabilities for all incorrect classes. Equal incorrect probabilities encourage equal incorrect logits, and under the distance picture that means `x` is encouraged to be equally distant from the incorrect templates while remaining closer to the correct template by the finite gap above.

So I expect the penultimate activations for each class to tighten around their own template and arrange themselves in a more regular geometry relative to other class templates. For three classes, the diagnostic picture should be especially clear: project examples onto the plane through the three templates. Hard-label training should allow broad clusters and large scales; the softened target should create tighter clusters with a more regular triangular arrangement. For semantically similar classes, I should watch whether the continuous similarity structure survives or gets flattened into the equal-distance constraint.

This gives me a calibration prediction. If the logit gaps stop growing without bound, the output probabilities should stop saturating as much. A hard-label model can be calibrated after training by temperature scaling, which divides logits by a positive scalar and leaves class predictions unchanged. A softened-target model may do part of that work during training because the target itself asks for finite confidence. Expected calibration error and reliability diagrams are the right tests: compare confidence bins to empirical accuracy.

Translation makes this calibration question matter operationally. Beam search does not merely ask for the argmax at each position; it ranks sequence hypotheses using next-token probabilities. If the probabilities are overconfident or poorly shaped, the search can prefer bad partial sequences. I therefore expect softened targets to help a decoder when they improve calibration. But I should not reduce the whole translation story to calibration: a model can have better BLEU while having worse negative log-likelihood, and temperature scaling a hard-label model may improve calibration without fully matching the softened-target BLEU. Calibration can explain part of the gain without explaining all of it.

The same geometry also warns me about a cost. Distillation relies on the teacher's relative probabilities for incorrect classes. A teacher saying "this digit is mostly a 3 but somewhat like an 8" gives a student information that the hard label cannot. If my training target makes all incorrect classes equally likely for each example of a class, then the teacher is encouraged to erase exactly those example-specific relative similarities. A more accurate teacher could then become a worse teacher for distillation.

I can phrase that loss of structure as mutual information. Let `X` be the training-example index and let `Y` be a scalar logit difference for two classes after random data augmentation. For each example `x`, collect Monte Carlo samples `y_{x,l} = f(d_l(z_x))`, estimate a mean `mu_x = (1/L) sum_l y_{x,l}`, and use a pooled variance `sigma^2 = (1/(NL)) sum_{x,l} (y_{x,l} - mu_x)^2`. With a shared-variance Gaussian approximation,

`I(X; Y)` can be estimated by averaging

`-(y_{x,l} - mu_x)^2/(2 sigma^2) - log((1/N) sum_{x'} exp(-(y_{x,l} - mu_{x'})^2/(2 sigma^2)))`

over examples and augmentations. The normalization constants cancel between the conditional Gaussian density and the mixture density. If the representation collapses to two class-level clusters, this information should approach `log 2` nats: the logit difference tells me which of the two classes the example belongs to, but not which individual example it is.

Now the rule is clear. I use the softened target when I want finite confidence, regularization, and better-calibrated probabilities. I am cautious when the downstream task needs the teacher's fine-grained wrong-class structure, especially distillation and transfer from final-layer features. The same equal-distance pressure that gives tidy clusters and calibrated logits can also destroy the dark knowledge that a student would otherwise learn.
