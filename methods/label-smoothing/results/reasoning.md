Let me start from the thing that bugs me about ordinary classification training. I have a softmax head, p(k) = exp(z_k)/Σ_i exp(z_i), and I train it with cross-entropy against a one-hot target q(k) = δ_{k,y}, so the loss on an example is just −log p(y). Fine. But look at when that loss is actually minimized. To get p(y) → 1 I need z_y to dominate every other logit — and not by a fixed margin, by an *unbounded* one: p(y) → 1 only as z_y − z_k → +∞ for all k ≠ y. The one-hot target is a maximum I can never reach at finite logits; it just keeps pulling the correct logit further and further above the rest, forever. The gradient ∂ℓ/∂z_k = p(k) − q(k) is bounded in [−1, 1], so each step is gentle, but the *target* it's chasing sits at infinity.

So what? Two things go wrong, and they're the same thing. The network gets more and more confident on the training data — it learns to slam full probability onto the training label of every example, which is precisely overfitting: fitting the labels says nothing about generalizing. And the unbounded growth of the largest-logit-minus-the-rest gap, against a bounded gradient, makes the model rigid — once a huge gap is established it's slow to revise. The unifying word is *over-confidence*. Szegedy and co. (2016) named this when they were regularizing the Inception net: hard targets drive z_y ≫ z_k, which both overfits and reduces adaptability.

If the problem is "the target lives at infinity, so confidence runs away," the cure suggests itself: don't put the target at infinity. Give every class a little floor of target probability so the correct logit has no incentive to escape to +∞. Take the one-hot and bleed a little mass ε onto a fixed distribution u over labels:

  q'(k) = (1 − ε) δ_{k,y} + ε u(k).

The natural u, absent any prior knowledge, is uniform, u(k) = 1/K, so q'(k) = (1 − ε)δ_{k,y} + ε/K. Now check that this kills the runaway. Every entry of q' is at least ε/K > 0. If z_y tried to run off to +∞, then p(y) → 1 and p(k) → 0 for k ≠ y, and the cross-entropy −Σ_k q'(k) log p(k) would blow up on those wrong-class terms, because q'(k) = ε/K is positive but log p(k) → −∞. So an infinite logit gap is now infinitely *expensive*, not free. The target no longer sits at infinity; it sits at a finite, ε-controlled configuration of the logits. That's label smoothing, and it's almost nothing to implement.

Let me rewrite the loss to see its structure. Cross-entropy is linear in the target, so

  H(q', p) = −Σ_k q'(k) log p(k) = (1 − ε)·(−Σ_k δ_{k,y} log p(k)) + ε·(−Σ_k u(k) log p(k)) = (1 − ε) H(q, p) + ε H(u, p).

So smoothing is exactly the ordinary hard-label cross-entropy, downweighted by (1 − ε), plus an ε-weighted term H(u, p) that pulls the prediction toward the prior u. And H(u, p) = D_KL(u‖p) + H(u); H(u) is a constant, so that second term is, up to a constant, a penalty on how far p has drifted from uniform — relative weight ε/(1 − ε). It's a regularizer that says "stay a bit humble, don't get too far from the prior." (Reversing the KL direction lands you on Pereyra's confidence penalty, −β H(p); same family — penalize over-confidence. And it's the marginalized version of label-dropout, DisturbLabel, where you'd randomly swap labels: smoothing is just the expectation of that.) On ImageNet, K = 1000 and ε = 0.1 gives a consistent improvement. Good — it works, and I can see *that* it bounds confidence.

But "it bounds confidence" is a thin explanation for something this widely used. I want to know what it actually does to the network — to the representation — because that's where the real "when does it help, when does it hurt" lives. So let me look at the last layer geometrically.

The logit for class k is z_k = x^T w_k, where x is the penultimate-layer activation vector (append a 1 to absorb the bias) and w_k is the weight vector for class k — call it the *template* for class k. Now stare at a squared distance:

  ‖x − w_k‖² = x^T x − 2 x^T w_k + w_k^T w_k.

The x^T x term is the same for every class, so it cancels inside the softmax. And w_k^T w_k is roughly constant across classes (templates have similar norms). What's left that varies with k is −2 x^T w_k = −2 z_k. So the logit is, up to a class-independent constant and a factor, *minus the squared Euclidean distance from the activation x to the template w_k*. The softmax is just a soft nearest-template classifier in penultimate space. Minimizing cross-entropy pulls x toward its correct template w_y.

Now ask what hard targets versus smoothed targets do to *where x lands*. With hard targets, all that matters is the gap z_y − z_k, and it wants to be huge; the wrong-class logits z_k are otherwise unconstrained — they can differ wildly from one another. So x just needs to be much closer to w_y than to anything else; it's free to be at any large distance, any direction, relative to the wrong templates. Result: broad, sprawling clusters, with big-magnitude activations (the over-confidence again, now visible as scale).

With label smoothing the target says something much more specific. The wrong-class target is the *same* value ε/K for *every* wrong class. So the loss wants all the wrong-class logits equal — z_k equal for all k ≠ y — which by the distance reading means x must be *equidistant* from every incorrect template. And it wants a *particular finite* gap to the correct one, set by ε, not an infinite one. Put those together: smoothing drives x to sit close to its own template and equally far from all the others, at a bounded magnitude. The clusters become tight and equally separated. If I pick three classes and project the penultimate activations onto the plane through their three templates, smoothing makes them snap into a regular triangle — tight blobs at the corners — whereas hard targets give a vaguer, broader spread. (Same effect regardless of architecture or dataset; it's a property of the loss geometry, not the net.) I'd want to actually run that projection — pick three classes, orthonormalize the plane through their templates, scatter the activations — to confirm the triangle.

Here's a consequence I didn't expect, and it's the calibration story. If smoothing bounds the magnitudes and the gaps, then it directly stops the softmax from saturating to over-confident probabilities. Modern nets are notoriously *miscalibrated* — Guo and co. showed their confidence runs well above their accuracy — and the standard fix is temperature scaling, dividing the logits by some T > 1 after training to flatten them. But preventing the logits from blowing up in the first place is doing the same job *during* training. So I'd predict that a model trained with a modest ε comes out about as well-calibrated, by Expected Calibration Error on a reliability diagram, as a hard-target model that's been temperature-scaled after the fact — without the post-hoc tuning step. (Concretely I'd expect something like a CIFAR-100 ResNet-56's ECE dropping from badly over-confident down to the temperature-scaled level at ε around 0.05; on ImageNet around ε = 0.1.) Smoothing is implicit calibration.

Does calibration actually buy anything beyond a nicer reliability plot? For plain classification, no — top-1 only looks at the argmax, which calibration doesn't move. But the moment the *soft* probabilities feed a downstream algorithm, calibration is everything. Translation is the case: beam search is an approximate maximum-likelihood (Viterbi-style) search over sequences, and it consumes the next-token probabilities directly. A better-calibrated next-token distribution — confidence that matches accuracy — should steer the search better. That squares with a puzzle in the Transformer work: label smoothing at ε = 0.1 *improved* BLEU even though it made perplexity *worse*. Better BLEU with worse likelihood is exactly what you'd get if smoothing's contribution is calibration of the decoding distribution rather than raw likelihood — the model is a worse density estimator (higher NLL at every temperature) but a better-calibrated one, and beam search rewards the latter. So part of the BLEU gain is the calibration; I shouldn't claim it's the *whole* story, since the NLL is uniformly worse and yet BLEU still wins.

Now I have to be honest and ask the dual question: if smoothing collapses each class into a tight, equidistant cluster, what did I *lose*? The thing that's gone is the structure *within* a class. With hard targets, two examples of "dog" can sit at quite different places relative to the other templates — one dog activation a bit toward "cat," another a bit toward "wolf" — because the wrong-class logits were unconstrained. That spread *is* information: it says "this particular dog resembles a cat somewhat." Smoothing forces every wrong-class logit equal, so every dog example ends up with the same relative similarities to every other class. The example-specific inter-class resemblance is flattened away.

When does that erasure bite? Knowledge distillation. The whole point of distilling a teacher into a student (Hinton 2015) is the "dark knowledge" in the teacher's *relative* wrong-class probabilities — "a 3 that looks a little like an 8" — exaggerated with a temperature. If the teacher was trained with label smoothing, those relative similarities have been homogenized: every example of a class hands the student the same soft pattern, so there's nothing example-specific left to transfer. So I'd predict the unsettling result: a teacher trained with smoothing can be *more accurate* than a hard-target teacher and yet distill into a *worse* student. A better teacher is not necessarily a better distiller. (And the same logic should make smoothing hurt transfer learning, which likewise feeds on non-class-relevant structure in the final layer — Kornblith and co. saw exactly that.)

I'd like to pin "erasure of information" down to a number rather than a picture. Let X be the index of a training example and Y the difference between two of its logits; the randomness comes from data augmentation jittering the input. The mutual information I(X; Y) measures how much the logit-difference still tells you about *which specific example* it was — i.e., how much example-specific structure survives in the logits beyond the class label. Approximate Y as Gaussian per example and estimate I by Monte Carlo over augmentations:

  Î(X; Y) = (1/N) Σ_x [ −(f(d(z_x)) − μ_x)² / (2σ²) − log( (1/N) Σ_{x'} e^{−(f(d(z_{x'})) − μ_x)² / (2σ²)} ) ],

with μ_x the per-example mean logit-difference over augmentations and σ² the pooled variance. As training proceeds I'd expect this to rise and then *decay*, and to decay more under label smoothing as the clusters tighten — toward log(2) in the two-class case, the floor where all that's left is one bit, "which of the two classes," and nothing about the individual example. That floor is the formal statement of the erasure: at the limit the teacher's logits carry exactly the label and not a scrap more, so distillation can't outperform just training the student with smoothing directly.

So the verdict on *when*: label smoothing bounds the runaway logit, which (1) regularizes and lifts accuracy, and (2) implicitly calibrates the output — and calibration is a real win wherever the soft probabilities are consumed downstream, beam search above all. But the same cluster-tightening that calibrates also *erases* the example-specific inter-class structure, so it (3) hurts any downstream use that needs that structure — distillation and transfer. Same mechanism, opposite signs depending on what you do with the logits.

The implementation is just the criterion, built straight from the H(q', p) = (1 − ε) H(q, p) + ε H(u, p) split with u uniform. The first term is the ordinary negative-log-likelihood at the target index; the second, with u(k) = 1/K, is ε times the mean over classes of −log p(k) — because H(u, p) = Σ_k (1/K)(−log p(k)) = mean_k(−log p_k). No need to materialize the soft target vector.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy against q'(k) = (1-eps) delta_{k,y} + eps/K,
    written via H(q',p) = (1-eps) H(q,p) + eps H(u,p), u uniform."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing                 # eps
        self.confidence = 1.0 - smoothing          # 1 - eps

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        # hard-label term H(q,p): -log p(y)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # uniform term H(u,p) = mean_k(-log p(k))  (u(k)=1/K)
        smooth_loss = -logprobs.mean(dim=-1)
        # (1-eps) H(q,p) + eps H(u,p); the eps/K * sum is exactly eps * mean
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

The chain: one-hot cross-entropy places its optimum at an infinite correct-vs-rest logit gap, so training drives the network to ever-greater confidence — overfitting and rigidity. Bleeding a fraction ε of the target onto a uniform prior, q'(k) = (1−ε)δ_{k,y} + ε/K, puts a positive floor under every class, makes an infinite gap infinitely costly, and decomposes into hard cross-entropy plus a uniform-deviation penalty. Reading the logit as negative squared distance to a class template shows the deeper effect: smoothing forces each activation close to its own template and *equidistant* from all others, collapsing classes into tight, equally-separated clusters at bounded magnitude. Bounded magnitude is implicit calibration — confidence matched to accuracy — which lifts accuracy and, where soft probabilities drive a downstream search like beam search, lifts that too. But the same tightening erases the example-specific inter-class resemblance the logits used to carry, measurable as a drop in input-logit mutual information toward the one-bit floor, which is why a smoothing-trained teacher, however accurate, distills poorly. The criterion is the two-term loss, a few lines over a standard training loop.
