Let me start from what actually goes wrong, because the symptom is concrete: I train a perfectly ordinary convolutional classifier with softmax cross-entropy on a long-tailed dataset, and at test time the rare classes are a disaster — the network would rather guess a head class than ever commit to a tail one. I want to understand *why* before I reach for any fix, because the fix should fall out of the diagnosis, not be a patch I bolt on.

So what am I minimizing? Empirical risk: R_emp(f) = (1/N) Σ_{i=1}^N L(f(x_i), y_i), one cross-entropy term per training example, averaged. That average is a sum, and every example contributes one term to it, so the value of the objective is dominated by whatever there are the most of. If 90% of my examples are five head classes and the remaining tail classes are scattered across the last 10%, then 90% of the terms in the sum come from those five classes. The optimizer doesn't care about classes; it cares about the average, and it can crush the average by getting the head examples right. The tail examples are a rounding error in the loss. That's the whole story of the bias: the model is faithfully minimizing exactly what I told it to minimize, and what I told it to minimize barely contains the tail.

Let me make that precise instead of hand-waving, because the precise version is going to tell me what to change. R_emp is a Monte-Carlo estimate of an expectation — as N grows it converges to E_{(x,y)~p}[L(f(x),y)], where p is the distribution the data was drawn from. Now group that expectation by class. The pair (x,y) decomposes: first draw the label y from the class prior p(y=c), then draw x from the class-conditional p(x|y=c). So

  E_{(x,y)~p}[L] = Σ_{c=1}^C p(y=c) · E[ L | y=c ].

There it is, written out: the objective is a *weighted* sum of the per-class conditional losses, and the weights are the class priors p(y=c). And the empirical class prior is just the count fraction, p(y=c) = n_c/N. So plain cross-entropy is optimizing

  Σ_c (n_c/N) · E[L | y=c],

with each class's conditional loss multiplied by its count fraction. A class with n_c = 5000 enters the objective a thousand times more heavily than a class with n_c = 5. The model isn't broken; it's solving a problem whose objective puts a thousand-to-one priority on the head. The bias is baked into the *prior* sitting in front of each class term.

Now — what do I actually want? Not this. At test time the data is balanced: every class appears equally, and accuracy on a rare class counts exactly as much as accuracy on a common one. So the quantity I care about is the loss under a *uniform* class prior, q(y=c) = 1/C:

  E_q[L] = Σ_c q(y=c) · E[L | y=c] = Σ_c (1/C) · E[L | y=c],

every class's conditional loss weighted equally. Staring at these two side by side, the whole problem is suddenly trivial to state: I have an estimator of Σ_c (n_c/N) E[L|c], and I want Σ_c (1/C) E[L|c]. Same conditional losses E[L|c] — those are properties of the model and the data, I don't get to change them — but in front of them, the *wrong* prior. I'm minimizing an expectation over the wrong distribution. I don't have a model problem; I have a distribution-mismatch problem. The training prior is n_c/N and the target prior is 1/C, and I need to convert one into the other.

Can I? I'm holding samples drawn under p (the skewed training distribution) and I want an expectation under q (the balanced target). This is exactly the situation importance sampling is built for. The identity is one line: for any function g,

  E_q[g] = Σ_c q(c) g_c = Σ_c p(c) · (q(c)/p(c)) · g_c = E_p[ (q(y)/p(y)) · g ].

I just multiply and divide by p(c). Whenever p(c) is nonzero — which it is, every class has at least one example — the ratio q(c)/p(c) is well defined, and the expectation under q is the expectation under p of the *reweighted* quantity (q/p)·g. So if I weight each example's loss by the ratio of the target prior to the training prior, the expectation of my reweighted average is the balanced loss I actually want. Let me write the reweighted empirical risk and check it really is unbiased for E_q[L]:

  R_w(f) = (1/N) Σ_{i=1}^N w_{y_i} · L(f(x_i), y_i),   with w_c = q(c)/p(c).

Take the expectation. E[R_w] = E_{(x,y)~p}[ w_y L ] = Σ_c p(c) · w_c · E[L|c] = Σ_c p(c) · (q(c)/p(c)) · E[L|c] = Σ_c q(c) E[L|c] = E_q[L]. The p(c) cancels exactly, and what's left is the balanced expectation. So R_w is an unbiased estimator of the thing I want to minimize. The reweighting is not a heuristic nudge; it is the unique multiplier that makes the cancellation happen and turns my skewed-prior estimator into a balanced-prior one. Anything other than q/p would leave a residual p(c) factor and estimate some other, unmotivated objective.

Now substitute the actual numbers. The target prior is uniform, q(c) = 1/C. The training prior is the count fraction, p(c) = n_c/N. So the weight is

  w_c = q(c)/p(c) = (1/C) / (n_c/N) = N / (C · n_c).

The weight on class c is the total number of examples N, divided by the number of classes C times that class's count n_c. It is inversely proportional to the class count — a rare class (small n_c) gets a large weight, a common class (large n_c) gets a small one — which is exactly the qualitative shape the diagnosis demanded: pull the objective's attention back toward the tail. But I didn't pick "inversely proportional" because it sounded right; it dropped out of the importance ratio as the only weight that makes the reweighted risk unbiased for the balanced loss. The constant out front, N/C, isn't decoration either — let me see what it does.

Pin down the scale. I worry that multiplying every loss term by some weight could secretly change the *magnitude* of the objective, and with a fixed learning rate the magnitude of the loss sets the size of the gradient steps — silently scaling the loss is silently changing the optimizer, which I'm not allowed to do. So compute the total weighted mass across all examples: Σ_c n_c · w_c = Σ_c n_c · (N/(C·n_c)) = Σ_c (N/C) = C · (N/C) = N. The total reweighted mass is N — exactly the number of examples. Which means the *average* weight per example is N/N = 1. The N/C constant is precisely what normalizes the importance ratio so that, on average, examples are weighted by one: the weighted objective sits on the same overall scale as the unweighted cross-entropy, just with the per-class emphasis redistributed. So that constant earns its place — it keeps the optimization dynamics unchanged while flipping the priors. Good. The bare proportionality is w_c ∝ 1/n_c; the N/C is the scale that keeps the loss magnitude honest.

Let me look at the same weight from a completely different angle, because if it's the right object it should make sense more than one way. Forget importance sampling for a moment and just ask: how much does each *class* contribute to the weighted objective in total? A class-c example contributes w_c · L; class c has n_c such examples; so the total contribution of class c is

  n_c · w_c · (mean L over class c) = n_c · (N/(C·n_c)) · (mean L) = (N/C) · (mean L),

and the n_c cancels. Every class contributes (N/C)·(its mean loss) — the *same* count-independent factor in front, regardless of whether the class has five examples or five thousand. So inverse-frequency weighting is precisely the rule that makes every class contribute equally to the loss: it cancels the empirical class prior, and the network is trained as if it had seen the same number of examples from each class. That's the cost-sensitive reading. Elkan's foundations of cost-sensitive learning say a per-class weight plays the role of a misclassification cost — declare a class costlier and you up-weight its examples, equivalent to resampling more of them — and here the cost I've assigned is "every class is equally important," i.e. cost inversely proportional to how often you see the class. The importance-sampling derivation and the equal-contribution derivation are the same weight seen twice, which is the kind of agreement that tells me I've found the natural object and not a coincidence.

And it connects to something even older that I should make explicit, because it's the same move in a different dialect. When a statistician fits a logistic model to data collected under a class prior that doesn't match the population they care about — choice-based or stratified sampling, where you deliberately over-collect a rare outcome — the consistent fix is the weighted exogenous-sampling likelihood: weight each observation by (target prior)/(observed prior), w_1 = τ/ȳ for the positives and w_0 = (1−τ)/(1−ȳ) for the negatives, where τ is the population fraction and ȳ the fraction in your sample. Maximizing that weighted log-likelihood is consistent for the *target* population rather than the sampled one. That is exactly q/p, two classes at a time. My N/(C·n_c) is the multi-class specialization of that same correction with the target prior set to uniform — τ_c = 1/C — and the observed prior set to the training count fraction — ȳ_c = n_c/N. So the rule isn't an image-classification trick at all; it's the textbook prior-correction / covariate-shift-on-the-label correction written in the vocabulary of a long-tail loss. The thing the cross-entropy needed was never a new loss — it was the importance weight that makes its expectation point at the balanced prior.

Before I commit, let me think about normalization, because this is where it is easy to lose a constant and quietly change the statement. If I multiplied every w_c by some positive constant κ, the minimizer of an exact full-risk objective would have the same class tradeoff, but the empirical objective I wrote is no longer an unbiased estimator of E_q[L]; it estimates κ E_q[L]. Its gradient is also κ times as large, which silently changes the effective step size under a fixed learning rate. So the scale is not arbitrary for the canonical empirical risk. The importance ratio q(c)/p(c) fixes it. With q(c)=1/C and p(c)=n_c/N, the scale is exactly N/C, and that same constant is what made Σ_c n_c w_c = N. If I want the standard average loss to keep its ordinary scale and its expectation to be exactly the balanced risk, I should not renormalize it again. The canonical rule is the plain ratio N/(C·n_c).

Now I have to stress-test this, because a weight that is unbiased can still be a noisy estimator. Look at the weight as imbalance gets severe. If the biggest class has 5000 images and the smallest has 5, then w differs by a factor of 1000 across classes — the tiny class's five images are each up-weighted a thousand times relative to a head image. Is that a problem? The reweighting is unbiased *in expectation*: E[R_w] = E_q[L] holds exactly. But "in expectation" hides the variance, and the per-class conditional loss E[L|c] is itself being *estimated* from that class's samples. For a class with five images, my estimate of E[L|c] is built from five noisy, probably near-duplicate, possibly mislabelled examples — it has enormous variance. And then I multiply that high-variance estimate by a weight of ~1000. Importance weighting corrects the *bias* of the prior but does nothing to *repair* a high-variance estimate from a tiny sample; if anything it amplifies it, because the variance of w_y·L scales with w^2. So the unbiased weighted objective, on extreme long-tail data, can be driven by a handful of unreliable, heavily amplified tail gradients. The honest read is: this is the exact prior correction when each class-conditional loss is estimable, and under extreme imbalance the variance of the tail estimate becomes the bottleneck. The canonical rule remains the clean q/p ratio, and what I would want to validate is balanced test accuracy across imbalance ratios; I will not pretend to know the numbers.

So the landing form is settled and it's almost embarrassingly small: a per-class weight equal to the total count over the number-of-classes times the class count. It's pure in exactly the way required — it touches only n_c, N, and C, never an image, never a model parameter, never a test label. It hands a length-C vector straight to a weighted cross-entropy, leaving the data, sampler, backbone, optimizer, and metric untouched. Let me write the function that fills the one open slot: total samples divided by number of classes times per-class count.

```python
import torch


def compute_class_weights(class_counts, num_classes, config):
    """Inverse class frequency reweighting.

    Derivation: plain cross-entropy minimizes the empirical risk, an estimate of
    E_{(x,y)~p}[L] with p(y=c) = n_c / N the training class prior. The balanced test
    targets the uniform prior q(y=c) = 1 / C. The importance-sampling identity
    E_q[L] = E_p[(q/p) L] makes the unique unbiased reweighting the ratio
        w_c = q(c) / p(c) = (1/C) / (n_c / N) = N / (C * n_c),
    i.e. weight inversely to the class count. The N/C constant fixes the average weight
    per example to 1, so the loss magnitude (hence the effective step size) is unchanged.

    Pure: depends only on the counts, num_classes, and the total. Returns a length-C
    tensor for nn.CrossEntropyLoss(weight=...). Every class must be present.
    """
    counts = class_counts.float()
    if torch.any(counts <= 0):
        raise ValueError("balanced class weights require every class count to be positive")
    total = counts.sum()                       # N = total training samples
    # w_c = N / (C * n_c): the importance ratio (uniform target) / (empirical prior)
    weights = total / (num_classes * counts)
    return weights
```

That's the whole method, and I can recover everything from one line of bookkeeping. The trouble was never the network; it was that average cross-entropy estimates the loss under the *training* class prior n_c/N, while the balanced test scores the loss under the *uniform* prior 1/C — the right objective sitting behind the wrong distribution. Importance sampling converts one into the other by weighting each example with the ratio of target prior to training prior, q(c)/p(c); substituting the uniform target and the count fraction makes that ratio N/(C·n_c), inversely proportional to the class count, with the N/C constant making the total weighted mass equal to N, the average example weight equal to one, and E[R_w] exactly equal to E_q[L]. The same weight reads, equivalently, as "make every class contribute equally to the loss" — the cost-sensitive stance that each class is equally important — and as the multi-class case of the classical weighted-likelihood prior correction with a uniform target prior. It is unbiased for the balanced risk and assumption-free in the counts, which is exactly why it is the natural count-only correction; its one weakness is that under extreme imbalance it amplifies the high variance of the few tail samples rather than removing it, which is a variance problem, not a bias problem. And it lands as a pure function of the counts that drops into the weighted cross-entropy without disturbing anything else in the pipeline.
