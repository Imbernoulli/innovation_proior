Label smoothing landed as the post-mortem predicted, and seeing exactly *how* tells me where the ceiling still is. VGG-16-BN/CIFAR-100 rose to $74.67$, the best of the three losses on that pair and clearly above Poly-1's $74.06$ and focal's $74.18$; MobileNetV2/FashionMNIST stayed high at $94.82$, even edging above Poly-1; the mean, $80.283$, is the best on the ladder. But ResNet-56/CIFAR-100 went the *other* way to $71.36$, actually *below* its focal ($71.67$) and Poly-1 ($71.56$) marks. So the strongest baseline overall still leaves the hardest pair behind, and the per-pair pattern is the clue: label smoothing helped where the failure was plain overconfidence (VGG, the non-residual net that overfits readily) but slightly *hurt* the deep residual ResNet-56. Why would the same mechanism hurt there? Label smoothing softens the *target* by a fixed, uniform $\varepsilon/C$ on every wrong class, identically for every example — a static, example-blind edit that cannot tell a clean, confidently-correct example from a mislabeled or ambiguous one. CIFAR-100 is not pristine, and on a deep residual net with the capacity to fit its tail of hard and near-mislabeled images, two things still bite that a uniform target floor does not address. First, the loss is still *convex and unbounded* in the activations: an example the net becomes confident about the *wrong* way still incurs a $-\log$ that grows without bound, so its gradient dominates and drags the boundary toward it. Second, the softmax is still *short-tailed*: its exponential decay saturates probabilities to $0/1$ fast, so a boundary-ambiguous example is still chased to a near-extreme probability and the classifier distorts itself locally to fit it. Both are about how the loss and the probabilities behave on the outlier and ambiguous examples — and label smoothing leaves both untouched.

So I attack the loss and the softmax at the source, keeping the two diseases distinct because they call for two cures. For large-margin outliers I want the per-example loss *bounded*: past some point, telling the model "you are very wrong about this one" should stop adding loss, so one mislabeled point cannot contribute unboundedly. For small-margin boundary noise I want the probability assignment *heavy-tailed*: a point far on the wrong side should still be allowed a non-negligible probability rather than being forced to $0/1$. The temptation is two unrelated hacks — clip the loss, fatten the softmax by hand — but I should respect *why* softmax and cross-entropy belong together and deform that pairing, so both cures fall out of one construction with the good properties intact. Cross-entropy is the KL divergence from the one-hot label to the softmax, and KL is the Bregman divergence induced by the negative entropy $F(y) = \sum_i (y_i \log y_i - y_i)$: $\Delta_F(y, \hat y) = F(y) - F(\hat y) - (y - \hat y)\cdot\nabla F(\hat y)$, nonnegative, zero iff $y = \hat y$. The softmax is exactly the gradient of the convex *dual* of that same $F$ on the simplex, so the logistic loss is the *matching loss* of the softmax — they are the two dual halves of one convex object. Generalize $F$ with a temperature and let both the transfer function and the divergence re-derive from it together, keeping the duality by construction.

The two cures point at the two operations to bend: the $\log$ in the loss and the $\exp$ in the softmax. I propose the **robust bi-tempered logistic loss**, built on a one-parameter deformation of each. The tempered logarithm $\log_t(x) = (x^{1-t} - 1)/(1 - t)$ is monotone increasing and concave, recovers ordinary $\log$ as $t \to 1$, and — the key property — for $0 \le t < 1$ is *bounded below* by $-1/(1-t)$. Ordinary $\log$ runs to $-\infty$ as its argument hits zero, the unboundedness that lets a wrong-confident example blow up; a $\log_{t_1}$ with $t_1 < 1$ floors at a finite value, so a loss built on it is bounded — cure one. Its inverse, the tempered exponential $\exp_t(x) = [1 + (1-t)x]_+^{1/(1-t)}$, recovers $\exp$ as $t \to 1$ but for $t > 1$ has a heavier, polynomial tail; a softmax built on $\exp_{t_2}$ with $t_2 > 1$ does not saturate as fast and can hold a near-boundary example at a non-extreme probability — cure two. Two temperatures, two cures, inverse functions, sitting naturally on the two dual halves.

Build the convex function so its gradient is $\log_t$. Integrating $\log_t$ elementwise and collecting constants gives $F_t(y) = \sum_i\big(y_i\log_t y_i + \tfrac{1}{2-t}(1 - y_i^{2-t})\big)$, with Hessian $\nabla^2 F_t(y) = \mathrm{diag}(y^{-t}) \succeq 0$ on the positive orthant, so it is strictly convex and induces a legitimate Bregman divergence reducing to the negative entropy at $t = 1$. Plugged into the Bregman definition, after the algebra,

$$\Delta_{F_t}(y, \hat y) = \sum_i\Big(y_i\log_t y_i - y_i\log_t \hat y_i - \tfrac{1}{2-t}\,y_i^{2-t} + \tfrac{1}{2-t}\,\hat y_i^{2-t}\Big),$$

which is the $\beta$-divergence axis with $\beta = 2 - t$ (KL at $t = 1$, squared Euclidean at $t = 0$). The transfer function repeats the dual-gradient construction with $F_t$: the link $\log_t(\hat y_i) = \hat a_i - \lambda$ with $\lambda$ a Lagrange multiplier enforcing $\sum_i \hat y_i = 1$, inverted to $\hat y_i = \exp_{t_2}(\hat a_i - \lambda_{t_2}(\hat a))$ — the tempered softmax, heavy-tailed for $t_2 > 1$. There is a real price for leaving $t = 1$: $\lambda_{t_2}$ has *no closed form*, because $\exp_t(a+b) \ne \exp_t(a)\exp_t(b)$ — the tempered exponential does not turn sums into products, so the partition function does not factor as $\lambda = \log\sum\exp(\hat a_i)$ does for ordinary softmax. I solve $\sum_i \exp_{t_2}(\hat a_i - \lambda) = 1$ per example by a short fixed-point iteration: subtract the max for stability, then repeatedly rescale the shifted activations by the tempered partition $Z^{1-t_2}$ until it settles — a handful of iterations, fully differentiable, so autograd handles the gradient through it.

Put the halves together with a deliberate *mismatch*: score the heavy-tailed tempered softmax ($t_2 > 1$) with the bounded tempered divergence ($t_1 < 1$). For a one-hot true class $c$, only the $i = c$ term has $y_i = 1$, so

$$L = -\log_{t_1}(\hat y_c) - \tfrac{1}{2 - t_1}\Big(1 - \sum_i \hat y_i^{2 - t_1}\Big),\qquad \hat y = \exp_{t_2}(\hat a - \lambda_{t_2}(\hat a)),$$

the first term the bounded tempered log-loss on the true class, the second the normalization over all classes that makes this a genuine Bregman divergence and not just a clipped log. When $t_1 = t_2$ the matching duality holds and the loss is convex; mismatching $t_1 < 1 < t_2$ makes it bounded *and* heavy-tailed at once — exactly the pair of properties ResNet-56's hard examples needed. The loss is non-convex in the activations now, and that is fine: convexity in the last layer was never convexity in the network's real parameters, and it was the very thing buying the outlier-sensitivity I am removing. Two properties survive the mismatch. *Properness*: the expected Bregman divergence differs from $\sum_i(-\eta_i\log_{t_1} p_i + \tfrac{1}{2-t_1}p_i^{2-t_1})$ only by terms depending on the true posterior and not the model, so the empirical mean over sampled labels is an unbiased estimator and minimizing it recovers $P(y|x)$. This is where building on the *Bregman* rather than the *Tsallis* divergence is load-bearing — the Tsallis route uses $\log_t(\hat y / y)$, and since $\log_t(a/b) \ne \log_t a - \log_t b$ its unbiased estimator would need the unknown posterior inside the logarithm, biasing it. *Bayes-risk consistency*: the repeated normalization term does not affect the classwise argmin, and since $-\log_{t_1}$ is monotone decreasing for $0 \le t_1 < 1$, $\arg\min_i L_i = \arg\max_i \hat a_i$, so the loss still picks the Bayes-optimal class.

This is the natural successor to label smoothing, not a jump sideways. Label smoothing curbed overconfidence by a static, uniform, example-blind edit to the *target*; the bi-tempered loss curbs the same overconfidence *and* the outlier-sensitivity at the source — a bounded loss and a heavy-tailed softmax that act per example, hardest on exactly the wrong-confident and near-boundary examples that label smoothing's uniform floor could not distinguish. It even *subsumes* the rung I stand on: the tempered loss admits an optional uniform label-smoothing of its target, so label smoothing is reachable inside this family. The two temperatures are the only knobs, and each cures one disease: pure boundary noise wants tail-only ($t_1 = 1$, $t_2$ large), pure large-margin outliers want bounded-only ($t_1$ small, $t_2 = 1$), and a real, moderately-noisy CIFAR-100 net wants both engaged but gently so training stays stable. A mild $(t_1, t_2) = (0.8, 1.2)$ sits in the safe, effective range for a residual net, with $(1, 1)$ recovering plain cross-entropy exactly. The bar is concrete: label smoothing's marks are $71.36 / 74.67 / 94.82$ (mean $80.283$), and the falsifiable claim is sharp — bound the loss and heavy-tail the softmax, and ResNet-56/CIFAR-100, the hard residual pair every static reshaping left behind, should finally rise above $71.36$, while VGG-16-BN holds $\ge 74.67$ and MobileNetV2/FashionMNIST stays near $94.82$. I would sweep $(t_1, t_2)$ per pair over $t_1 \in [0.5, 1)$, $t_2 \in (1, 4]$, since the right amount of boundedness and tail-heaviness depends on each pair's label ambiguity, and watch that the fixed-point normalization stays stable.

```python
# EDITABLE region of pytorch-vision/custom_loss.py — finale: robust bi-tempered logistic loss
def compute_loss(logits, targets, config):
    """Robust bi-tempered logistic loss.

        L = Delta_{F_t1}( one_hot(y), tempered_softmax(logits, t2) )

    t1 < 1 bounds the per-example loss (resists wrong-confident outliers);
    t2 > 1 heavy-tails the softmax (resists boundary-ambiguous examples);
    t1 = t2 = 1 recovers softmax cross-entropy.
    """
    t1, t2, num_iters = 0.8, 1.2, 5
    if t1 == 1.0 and t2 == 1.0:
        return F.cross_entropy(logits, targets)

    def log_t(u, t):
        return torch.log(u) if t == 1.0 else (u.pow(1.0 - t) - 1.0) / (1.0 - t)

    def exp_t(u, t):
        if t == 1.0:
            return torch.exp(u)
        return torch.clamp(1.0 + (1.0 - t) * u, min=0.0).pow(1.0 / (1.0 - t))

    def tempered_softmax(a, t):
        if t == 1.0:
            return F.softmax(a, dim=-1)
        mu = a.max(dim=-1, keepdim=True).values            # shift for stability
        shifted = a - mu
        z = shifted
        for _ in range(num_iters):                          # fixed point for the normalizer
            partition = exp_t(z, t).sum(dim=-1, keepdim=True)
            z = shifted * partition.pow(1.0 - t)
        partition = exp_t(z, t).sum(dim=-1, keepdim=True)
        norm = -log_t(1.0 / partition, t) + mu
        return exp_t(a - norm, t)

    y = F.one_hot(targets, config['num_classes']).to(logits.dtype)  # one-hot target
    p = tempered_softmax(logits, t2)                               # heavy-tailed probs
    if t1 == 1.0:
        return (y * (torch.log(y + 1e-10) - torch.log(p))).sum(dim=-1).mean()
    term1 = (y * (log_t(y, t1) - log_t(p, t1))).sum(dim=-1)        # bounded log-loss
    term2 = (1.0 / (2.0 - t1)) * (y.pow(2.0 - t1) - p.pow(2.0 - t1)).sum(dim=-1)  # normalizer
    return (term1 - term2).mean()
```
