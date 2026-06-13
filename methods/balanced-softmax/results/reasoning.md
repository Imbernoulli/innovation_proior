Let me start from what actually goes wrong. I train a classifier by minimizing cross-entropy on a long-tailed split — head classes with thousands of images, tail classes with a handful — and then I grade it on a balanced test set, where every class gets equal say and the score is top-1 accuracy, or equivalently the average of the per-class error rates, the balanced error. And it underperforms on exactly the tail. The features look fine; people have shown that ordinary instance-balanced training already learns good representations and it's the linear head whose decision boundary is skewed toward the head classes. So the damage is in the *decision rule*, and I want to understand precisely why minimizing the training loss produces the wrong decision rule for the balanced test.

Write down what the Softmax actually estimates. Logits η_j = θ_j^T f(x), and φ_j = e^{η_j} / Σ_i e^{η_i} is trained, by the negative log-likelihood −log φ_y, to match the conditional label distribution *of the data it sees*. By Bayes' rule that conditional is φ_j = p(y=j|x) = p(x|y=j) p(y=j) / p(x). Stare at that factorization for a second. There are two pieces. p(x|y=j) is the class-conditional likelihood — what class j actually looks like — and that is a property of the images, shared between training and testing; the same banana looks like a banana in both splits. p(y=j) is the class prior — how often class j shows up — and *that* is the thing that differs: in the training split the prior is n_j / n, heavily weighted to the head, while on the balanced test set the prior is uniform, 1/k. So when I fit the Softmax to the training data, I am fitting p̂(y=j|x) ∝ p(x|y=j) · (n_j/n), a posterior baked with the *training* prior. But the test metric scores me against p(y=j|x) ∝ p(x|y=j) · (1/k), the posterior with a *uniform* prior. The Softmax isn't broken; it's faithfully estimating the wrong posterior. Its argmax favors head classes because it has multiplied the likelihood by the head-heavy training prior, and at test time that prior is gone.

Now my instinct, like everyone's, is to rebalance: reweight the loss so each class counts the same, w_c ∝ 1/n_c, or resample so the tail shows up as often as the head. Let me actually think about whether that fixes the posterior problem, because I have a nagging feeling it doesn't. Reweighting multiplies class j's cross-entropy term, and hence its whole gradient, by a class-constant scalar. That scalar depends only on n_c — it never looks at the model's output. So it uniformly rescales gradients; it doesn't reach inside the Softmax and correct *where* the boundary sits relative to the prior. And there's a sharper problem I should take seriously: on separable data — and over-parameterized nets do separate the training set — unregularized logistic regression converges to the max-margin solution, and the max-margin solution is *invariant* to importance weights. Soudry and others showed the implicit bias of gradient descent here, and Byrd and Lipton confirmed empirically that importance weighting has little lasting effect without regularization. So in the regime I'm in, scaling each class's loss by a constant can leave the converged classifier essentially unchanged. Reweighting is fighting the symptom with a tool that the optimization can ignore. And when the imbalance is severe the weights blow up — 1/n_c for a class seen once is enormous — and those huge weights produce abnormally large, unstable gradients exactly when I need stability most. The effective-number refinement, w_c ∝ (1 − β)/(1 − β^{n_c}), smooths the weights so near-duplicate samples don't over-credit the head, and it's a nicer member of the family, but it's the same family: a static, model-blind scalar on the loss, plus now a β to tune, and it still says nothing about the train/test posterior mismatch. So the whole reweighting line treats the wrong object. The wrong object is the prior buried inside the Softmax, and I should attack *that*.

I don't want to reweight the loss. I want the network's logits η to model the *balanced* posterior φ — the test-time thing — while I *train* on the imbalanced data. The trouble is the training data only ever lets me match the *training* posterior φ̂. So the question becomes mechanical: if I declare that η is the standard Softmax parameterization of φ (the balanced one), what is the corresponding parameterization of φ̂ (the imbalanced one I can actually fit) in terms of the same η? If I can write φ̂ as some explicit function of η, then I train that function against the labels, and the η underneath comes out modeling φ — and at test time I just use plain Softmax(η), no shift, because η already *is* the balanced posterior's logits.

Let me try to derive the link. I have two posteriors built from the *same* model output η and the *same* likelihood, differing only in their prior and evidence:

  φ_j  = p(y=j|x)  = [p(x|y=j) / p(x)] · (1/k),
  φ̂_j = p̂(y=j|x) = [p(x|y=j) / p̂(x)] · (n_j / n).

The exponential-family parameterization of the multinomial gives me both the canonical response function — the Softmax, φ_j = e^{η_j}/Σ_i e^{η_i} — and its inverse, the canonical link, η_j = log(φ_j / φ_k), each logit as a log-ratio against a fixed reference class k. That link is the lever. I want to turn an equation about η and φ into one about η and φ̂. So take the link and add −log(φ_j/φ̂_j) to both sides:

  η_j − log(φ_j/φ̂_j) = log(φ_j/φ_k) − log(φ_j/φ̂_j) = log(φ̂_j/φ_k).

The φ_j cancels on the right and I'm left with log(φ̂_j/φ_k). Exponentiate:

  φ_k · e^{η_j − log(φ_j/φ̂_j)} = φ̂_j.

This already has the shape I want — φ̂_j is an exponential of (η_j minus a per-class log-correction), scaled by the single unknown φ_k. To pin φ_k, sum over j and use that the φ̂ are a distribution, Σ_j φ̂_j = 1:

  φ_k · Σ_{i=1}^k e^{η_i − log(φ_i/φ̂_i)} = 1   ⟹   φ_k = 1 / Σ_i e^{η_i − log(φ_i/φ̂_i)}.

Substitute back:

  φ̂_j = e^{η_j − log(φ_j/φ̂_j)} / Σ_{i=1}^k e^{η_i − log(φ_i/φ̂_i)}.

So φ̂ is *a Softmax of shifted logits*, each logit η_i shifted by −log(φ_i/φ̂_i). The φ_k vanished cleanly — it was just the normalizer. Now I need that shift in terms of things I know, not in terms of the unknown posteriors. Plug in the two definitions:

  log(φ_j/φ̂_j) = log( [p(x|y=j)/p(x)·(1/k)] / [p(x|y=j)/p̂(x)·(n_j/n)] ).

The likelihood p(x|y=j) is identical top and bottom — that's the whole premise, same generative process — so it cancels. What's left:

  log(φ_j/φ̂_j) = log( (1/k) / (n_j/n) ) + log( p̂(x)/p(x) ) = log( n/(k n_j) ) + log( p̂(x)/p(x) ).

Two terms. The second, log(p̂(x)/p(x)), is the evidence ratio — and crucially it does *not* depend on the class j. So inside the Softmax over j it's a constant added to every shifted logit, and a constant added to all logits cancels in the Softmax. Gone. The first term splits as log(n/k) − log n_j; the log(n/k) piece is *also* class-independent, so it too cancels. The only surviving, class-dependent part of the shift is +log n_j (it enters as −(−log n_j) = +log n_j once I track the sign through). Let me just substitute the full thing and watch everything die:

  φ̂_j = e^{η_j − log(n/(k n_j)) − log(p̂(x)/p(x))} / Σ_i e^{η_i − log(n/(k n_i)) − log(p̂(x)/p(x))}.

Factor the common constants e^{−log(n/k)} and e^{−log(p̂(x)/p(x))} out of both numerator and denominator — they cancel — and −log(1/n_j) = +log n_j survives:

  φ̂_j = n_j e^{η_j} / Σ_{i=1}^k n_i e^{η_i}.

There it is, and it's almost embarrassingly clean. If the model's logits η are the standard Softmax parameterization of the *balanced* posterior, then the *training* posterior is the same Softmax but with each logit nudged by +log n_j — equivalently, the class scores e^{η_j} multiplied by their training counts n_j inside the normalizer. The imbalanced posterior is just the balanced one with the prior multiplied back in, which in log-space is an additive log-count offset. So the loss I should minimize on the training data is the negative log of *that*:

  l̂(θ) = −log φ̂_y = −log( n_y e^{η_y} / Σ_i n_i e^{η_i} ).

And this is exactly −log Softmax(η + log n)_y: take the raw logits, add the per-class vector log n, and apply ordinary cross-entropy. Nothing else. At test time I drop the +log n shift entirely and predict argmax η, because η was trained, through this loss, to model the balanced posterior φ. No reweighting, no resampling, no test-time change, no extra parameters — just a log-count shift on the logits during training.

Let me make sure I believe the *direction* of the shift, because a sign error here would silently invert everything. During training I add +log n_j. A head class has large n_j, so +log n_j is a large positive offset on its training logit. That means: to make the *training* posterior assign the right (high) probability to a frequently-correct head class, the network only needs a *smaller* underlying η_y, because the +log n_j is doing part of the work. Conversely a tail class gets a small (or negative) offset, so the network is forced to push its underlying η_y *up* to get the same training probability. So the bare logits η — the ones I use at test — end up *relatively boosted* for the tail and *relatively suppressed* for the head, which is precisely the correction the balanced test wants. Good, the sign is right: train with +log n on the logits, test without it.

I want to double-check this is the *right* correction and not just *a* correction, by coming at it from a completely different angle — what decision rule actually minimizes the balanced error, ignoring all this parameterization? Balanced error is (1/k) Σ_k Pr[ŷ ≠ k | y = k], the average per-class error under a uniform test prior. The Bayes-optimal rule for that is to predict argmax_y p(y|x) where p is computed *under the uniform prior*. But all I can estimate from data is the training posterior p̂(y|x) ∝ p(x|y) p̂(y). To recover the uniform-prior posterior I divide out the training prior: p(y|x) ∝ p̂(y|x) / p̂(y) ∝ p̂(y|x) / n_y. So the balanced-error-optimal decision is argmax_y p̂(y|x)/n_y — divide the training posterior by the class count. In log-space that's argmax_y [ log p̂(y|x) − log n_y ]. And if my training logits are modeling log p̂(y|x) up to a constant, the balanced-optimal test rule is argmax_y [η_y − log n_y]... which is the *same family* of corrections, a log-count offset on the logits. The two views are duals: I can either train η to be the balanced posterior by *adding* log n during training and testing on bare η, or train η to be the training posterior and *subtract* log n at test. Both land on a log-count shift, and the offset is log of the *prior* — not 1/n, not n^{1/4}, but exactly the log of the count — because that's what dividing out a prior costs in log-space. That dual view also tells me the rule is the consistent one: as data grows, argmax of the divided-out posterior converges to the Bayes-optimal balanced classifier. I find it reassuring that the probabilistic label-shift derivation and the "what minimizes balanced error" argument agree on the exact functional form. That's the signature of having found the right object rather than a lucky heuristic.

Still, agreement of two intuitions isn't a generalization guarantee. Let me try to derive the same loss from margin theory, because the long-tail literature lives there and I want to know whether the *theory* prefers this exact shift or merely something like it. Margin bounds say a classifier's per-class test error is bounded by something that *decreases* with the training margin γ_j of that class and *increases* as the class gets rarer. Generalizing the linear-prediction margin bound (Kakade, Sridharan, Tewari) to the balanced, multi-class case, the per-class error obeys roughly

  err_j ≲ (1/γ_j)·√(C/n_j) + (log n)/√n_j,

and the balanced error is the average over j. The first term is the one I can control through the margin; the second is a lower-order n_j term. So to minimize the balanced bound I minimize Σ_j (1/γ_j)√(C/n_j). But I can't make every margin huge — enlarging one class's margin eats into another's, so there's a budget; constrain Σ_j γ_j = β. Minimizing Σ_j (1/γ_j)√(C/n_j) subject to Σ_j γ_j = β is a clean Cauchy-Schwarz problem. By Cauchy-Schwarz, Σ_j (a_j/γ_j) · Σ_j γ_j ≥ (Σ_j √a_j)^2 with a_j = √(C/n_j), and equality — the minimum — when γ_j ∝ √a_j = (C/n_j)^{1/4} ∝ n_j^{-1/4}. So

  γ*_j = β n_j^{-1/4} / Σ_i n_i^{-1/4}.

The optimal margin grows as the class gets rarer (n_j^{-1/4} is large for small n_j) — rare classes deserve bigger margins, which matches the intuition that they're the ones at risk. This is exactly the trade-off LDAM found; LDAM enforces it by *subtracting* a margin Δ_j = C/n_j^{1/4} from only the true class's logit inside a Softmax. But that's a binary-derived, true-class-only construction. I want the margin enforced as a correction to the *whole* posterior. A class-j example is safe at margin γ_j when l_j(θ) + γ_j <= t, which is the same as forcing the raw loss l_j(θ) to sit γ_j below the threshold t. So the margin loss I need to minimize is l̂*_j = l_j + γ*_j. Can I realize that with another shifted Softmax? Try the parameterization

  φ̂_j = e^{η_j − log γ*_j} / Σ_i e^{η_i − log γ*_i},   l̂_j = −log φ̂_j,

i.e. shift each logit by −log γ*_j. Since γ*_j ∝ n_j^{-1/4}, we have −log γ*_j = (1/4) log n_j + const, and the const cancels in the Softmax, so this is φ̂_j = n_j^{1/4} e^{η_j} / Σ_i n_i^{1/4} e^{η_i} — the *same family* as before but with exponent 1/4 instead of 1.

Now I have to check whether −log φ̂_j really realizes that margin loss, or only approximately. The direct target −log φ̂*_j = −log φ_j + γ*_j cannot hold for every class, because γ*_j > 0 would make every φ̂*_j smaller than φ_j and then the probabilities would sum to less than one. I need a class-independent base shift γ_base to absorb the normalization: −log φ̂*_j = −log φ_j − γ_base + γ*_j. That is just a threshold shift, and γ* does not depend on the threshold t, so it leaves the margin allocation unchanged. Compute the difference l̂_j − (l_j − γ_base):

  l̂_j − (l_j − γ_base) = log φ_j − log φ̂_j + γ_base
    = log[ Σ_i e^{η_i − log γ*_i + log γ*_j} ] − log[ Σ_i e^{η_i} ] + γ_base.

By the mean-value theorem, log A − log B = (A − B)/α for some positive α between A and B, so this is (Σ_i e^{η_i}(γ*_j/γ*_i) − Σ_i e^{η_i})/α + γ_base. Pull out γ*_j: Σ_i (γ*_j/γ*_i) e^{η_i} = γ*_j Σ_i (1/γ*_i) e^{η_i}, and lower-bound Σ_i (1/γ*_i) e^{η_i} by Cauchy-Schwarz applied to the split e^{η_i} = e^{η_i/2}·e^{η_i/2}: Σ_i (1/γ*_i) e^{η_i} ≥ (Σ_i e^{η_i/2})^2 / Σ_i γ*_i = (Σ_i e^{η_i/2})^2 / β. And (Σ_i e^{η_i/2})^2 = λ Σ_i e^{η_i} with 1 ≤ λ ≤ k, where λ is close to 1 when the score mass concentrates on one class. Putting it together, the difference is bounded below by (γ*_j (λ/β) Σe^{η_i} − Σe^{η_i})/α + γ_base. With the simplifying choices β = 1 and γ_base = 1, and near a concentrated solution where λ is near 1 and α is on the scale of the score sum, this behaves like γ*_j. It is not an identity; it is an approximation that says the shifted Softmax with exponent 1/4 approximately enforces the bound-optimal margins. So the margin-bound route lands on n_j^{1/4} e^{η_j}, the *same loss family*, just with exponent 1/4.

I now have two derivations on the table that both produce φ̂_j ∝ n_j^a e^{η_j} — the same additive log-count shift on the logits — but they disagree on the exponent: the probabilistic label-shift derivation forces a = 1 (you multiply the prior back in, full strength), and the margin-bound derivation suggests a = 1/4. Which do I trust? The label-shift derivation is exact — it's an identity, given the premise that train and test share the likelihood. The margin-bound derivation is a chain of relaxations: a continuous surrogate for the 0-1 error, a *bound* (not the true error), a binary-hinge margin lineage adapted to a multi-class Softmax loss, and a mean-value-theorem approximation to even realize the margins as a Softmax. That relaxed route should not override the exact posterior identity — minimizing the bound's preferred margin allocation (giving 1/4) is not the same as directly matching the balanced posterior (giving 1). When two derivations of the same functional form disagree only in an exponent, and one is an identity while the other is a loose bound, I take the identity. So a = 1: shift by log n_j, not (1/4) log n_j. The bound is valuable not for its exact constant but for *independently confirming the functional form* — a per-class additive log-count shift — and for explaining *why* rarer classes need help.

Let me sanity-check the loss does the right thing at the extremes. If all classes are balanced, n_j = n/k for all j, then log n_j is a class-independent constant and cancels in the Softmax — the loss reduces to ordinary cross-entropy. Good: with balanced data there's nothing to correct and the method correctly does nothing. As the imbalance grows, log n_j spreads the offsets apart, and because the correction is *additive* and *bounded* (it's log of a count, growing only logarithmically with n_j), it does not blow up the way 1/n_j reweighting does — a class seen once gives offset log 1 = 0, a class seen ten thousand times gives offset log 10^4 ≈ 9.2, a spread of ~9 in logit units, not a factor of 10^4 in gradient magnitude. That logarithmic, additive nature is exactly why this avoids the abnormal-gradient instability that reweighting suffers. And it's a per-*logit* shift inside the normalizer, which is the thing a scalar loss reweight provably cannot reproduce — the prior enters the posterior multiplicatively *inside* the Softmax sum, n_i e^{η_i}, so it must be corrected inside the sum, not as an outer multiplier on the loss.

There's a second-order concern I should think through, because it bites if I naively combine this loss with class-balanced sampling. Suppose I keep this loss *and* also resample so each class appears B/k times per batch. At convergence, set the per-class gradient to zero. The gradient of −log φ̂_j with respect to θ_j is, for a positive sample of class j, f(x)(φ̂_j − 1), and for a negative sample, f(x)φ̂_j — standard Softmax gradients, just with φ̂ the shifted posterior. With class-balanced sampling, the zero-gradient condition for class j can be written as an empirical expectation over the original training split:

  (1/n_j) E_{y=j}[f(x)(1 − φ̂_j)] − Σ_{i≠j} (1/n_i) E_{y=i}[f(x)φ̂_j] = 0.

As the training loss converges, φ̂_y → 1. In that limit the ratio φ̂_j/φ_j tends to 1 for a positive sample of class j, and for a negative sample from class i it tends to n_j/n_i. Substituting those ratios gives

  (1/n_j) E_{y=j}[f(x)(1 − φ_j)] − Σ_{i≠j} (n_j/n_i^2) E_{y=i}[f(x)φ_j] ≈ 0.

Divide the whole equation by n_j, which does not change the zero, and the effective condition becomes

  (1/n_j^2) E_{y=j}[f(x)(1 − φ_j)] − Σ_{i≠j} (1/n_i^2) E_{y=i}[f(x)φ_j] ≈ 0.

So combining the log-count loss with class-balanced sampling *double-counts* the rebalancing — the optimization behaves as though it has a 1/n_j^2 class factor rather than the usual inverse-frequency 1/n_j. That is an over-balance that tips the optimization toward the tail past the intended distribution. The lesson isn't to abandon the loss; it's that the loss already encodes the full rebalancing, so I must *not* stack a balanced sampler on top of it naively. The clean recipe is the loss with ordinary instance-balanced sampling; if resampling is needed for optimization on an extremely imbalanced dataset, the sampling rate has to be *learned* to avoid exactly this over-balance, rather than hand-set to uniform-per-class. For the loss itself, the takeaway I keep is: it is a complete, self-contained correction, and it composes badly with a second hand-crafted rebalancer.

Many detection and segmentation heads don't use a multi-class Softmax — they use k independent binary logistic regressions, one per class, and predict argmax_j η_j. The same prior-correction idea should transfer by Bayes' theorem, but the algebra is different because now I'm correcting each binary posterior φ_j = e^{η_j}/(1 + e^{η_j}) separately. The binary link is η_j = log(φ_j/(1 − φ_j)). Decompose both φ_j and 1 − φ_j by Bayes for train and test: φ_j = p(x|y=j)p(y=j)/p(x), 1 − φ_j = p(x|y≠j)p(y≠j)/p(x), and similarly with hats for training. Substituting into the link and simplifying — the likelihoods and evidence cancel as before — I get η_j = log[ (φ̂_j/(1 − φ̂_j)) · (p(y=j)/p̂(y=j)) · (p̂(y≠j)/p(y≠j)) ]. With test prior p(y=j) = 1/k and train prior p̂(y=j) = n_j/n, the prior ratios become (n/k)/n_j and (n − n_j)/(n − n/k), so

  η_j − log[ ((n/k)/n_j) · ((n − n_j)/(n − n/k)) ] = log(φ̂_j/(1 − φ̂_j)),

i.e. shift each binary logit by −log[ ((n/k)/n_j)·((n − n_j)/(n − n/k)) ] during training, then

  φ̂_j = e^{η_j − log[...]} / (1 + e^{η_j − log[...]}).

Same principle — a per-class log-prior offset — but the binary version carries an extra (n − n_j)/(n − n/k) factor from the "not-class-j" prior. One practical wrinkle for detection: suppressing background-sample gradients causes a flood of false positives (the equalization-loss observation), so I'd apply this shift only to foreground classes and leave background on the plain sigmoid, dropping the constant for it. But the multi-class Softmax form is the core, and that's what I'll write.

So let me land it as the code I'd actually ship, filling the one open slot in the loss module — how the per-class counts enter the loss. The whole method is: add log(counts) to the logits, then standard cross-entropy. At test, do nothing.

```python
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def balanced_softmax_loss(labels, logits, sample_per_class, reduction='mean'):
    """Compute the corrected softmax loss for long-tailed training.

    Trains the logits to model the BALANCED-test posterior while training on the
    long-tailed split, by shifting each logit by + log(n_class) before the
    standard Softmax cross-entropy. Derived from the label-shift identity
    phi_hat_j = n_j e^{eta_j} / sum_i n_i e^{eta_i}: the imbalanced training
    posterior is the balanced one with the per-class prior (the count) multiplied
    back in, which is additive in log-space.
    """
    # spc[j] = n_j, the training count of class j; broadcast over the batch.
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)   # [batch, k]
    # eta_j + log n_j : the only change from ordinary cross-entropy.
    logits = logits + spc.log()
    # -log Softmax(eta + log n)_y, log-sum-exp stable; no separate normalization.
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class BalancedSoftmax(_Loss):
    """Loss module holding the per-class training counts n_1..n_k."""

    def __init__(self, sample_per_class):
        super().__init__()
        # n_1, ..., n_k : a fixed property of the long-tailed training split.
        self.sample_per_class = torch.as_tensor(sample_per_class)

    def forward(self, logits, labels, reduction='mean'):
        return balanced_softmax_loss(labels, logits, self.sample_per_class, reduction)


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(dim=-1)
```

The causal chain, end to end: ordinary cross-entropy on a long-tailed split fits the *training* posterior p̂(y|x) ∝ p(x|y)·(n_y/n), but the balanced test scores me against the *balanced* posterior p(y|x) ∝ p(x|y)·(1/k); the Softmax is a faithful estimator of the wrong thing, because the head-heavy training prior is baked in and gone at test. Reweighting and resampling attack a class-constant scalar on the loss, which is model-blind, can be invariant to the converged max-margin solution on separable data, and blows up gradients at severe imbalance — so they treat the wrong object. The right object is the prior inside the Softmax. Demanding that the logits η model the balanced posterior and asking what the training posterior is in terms of that same η — via the exponential-family canonical link, adding −log(φ/φ̂) and summing to normalize — yields the identity φ̂_j = n_j e^{η_j} / Σ_i n_i e^{η_i}, so the training posterior is the balanced one with the count multiplied in, i.e. an additive +log n_j on each logit. The same shift falls out independently as the Bayes-optimal rule for balanced error (divide the training posterior by its prior), and the margin-bound route confirms the functional form with the looser exponent 1/4; the exact identity fixes the exponent at 1. The shift is additive, logarithmic, and per-logit, so it corrects inside the normalizer where a scalar reweight cannot, and it stays bounded instead of exploding. It must not be stacked with a hand-set balanced sampler (that over-balances to 1/n_j^2); it extends to binary-logistic heads with an extra not-class-j prior factor. And the implementation is one line — add log(counts) to the logits, run standard cross-entropy, train on instance-balanced batches, and at test use the bare logits.
