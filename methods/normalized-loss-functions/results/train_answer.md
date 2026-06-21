We want to train a high-capacity softmax classifier when some fraction of the training labels are wrong, and we want to do it by changing only the per-example loss — no noise-transition matrix to estimate, no clean validation set, no sample filtering, no auxiliary network, no special optimizer schedule. The model emits logits $z$ and probabilities $p(k\mid x)=\mathrm{softmax}(z)_k$, the targets are one-hot, and the training loop just evaluates a loss on $(\text{logits}, \text{labels})$ and backpropagates the minibatch mean. The one theoretical lever we have is a sufficient condition for noise tolerance: if a loss has a constant class-sum, $\sum_{j=1}^K L(f(x),j)=C$ with $C$ independent of $x$ and $f$, then under uniform noise — true class kept with probability $1-\eta$, each wrong class drawn with probability $\eta/(K-1)$ — the noisy risk is an affine transform of the clean risk,
$$R_\eta(f) = \Big(1 - \tfrac{\eta K}{K-1}\Big)\,R(f) + \frac{C\,\eta}{K-1}.$$
When $\eta < (K-1)/K$ the coefficient on $R(f)$ is positive, so every clean-risk minimizer remains a noisy-risk minimizer and the global optimum survives the corruption. The trouble is that the losses satisfying this are exactly the ones that train badly. Cross entropy, $\mathrm{CE}=-\log p_y$, fits deep networks well because it delivers large updates on hard examples, but it has no constant class-sum and will memorize corrupted labels. Mean absolute error is symmetric — its class-sum is constant — but its gradients vanish on badly classified samples, so it underfits anything hard. Reverse cross entropy is also symmetric once $\log 0$ is replaced by a finite negative constant, but on its own it has the same insufficient-learning problem. The known compromises do not resolve the tension: generalized cross entropy interpolates between CE and MAE but is fully robust only at the MAE end, and symmetric cross entropy bolts a reverse term onto ordinary CE but leaves the non-robust CE term inside the objective. The open problem is to get CE-like fitting behavior together with theorem-level robustness without ever putting a non-robust term back in.

I propose to stop hunting for robust losses one at a time and instead manufacture them, and then to combine two manufactured pieces into an active–passive objective; the concrete instance I build is NCE+RCE, normalized cross entropy plus reverse cross entropy. The first idea is normalization: the constant-sum condition is itself a construction recipe. For any nonnegative loss with positive class-sum, define
$$L_{\mathrm{norm}}(f(x),y) = \frac{L(f(x),y)}{\sum_{j=1}^K L(f(x),j)},$$
and then $\sum_y L_{\mathrm{norm}}(f(x),y)=1$ identically, so the affine-risk argument applies with $C=1$. This does not merely discover one more robust loss; it turns any suitable base loss into a symmetric one. (The clean uniform-noise guarantee is the strong statement; for class-conditional noise the guarantee narrows and needs extra assumptions — zero clean risk, bounded normalized losses on off-target classes, and a diagonally dominant transition — and that caveat stays attached.) Applying normalization to cross entropy gives, with $p_k=p(k\mid x)$, one-hot $q$, and label $y$,
$$\mathrm{NCE} = \frac{-\sum_k q_k \log p_k}{-\sum_j \log p_j} = \frac{-\log p_y}{-\sum_j \log p_j},$$
a ratio in $[0,1]$ where I keep the numerator as the positive $-\log p_y$ and the denominator as the positive class-sum $-\sum_j \log p_j$; the signs are load-bearing, and equivalently this equals $\log p_y / \sum_j \log p_j$ since both logs are nonpositive.

NCE is robust, but robustness alone is not the criterion — empirically NCE (and normalized focal loss) underfit on hard noisy data even as CE and focal loss overfit it. The reason is visible in the denominator: it contains $Q=-\sum_{k\neq y}\log p_k$, so the loss can be improved by moving the non-target distribution even while $p_y$ is held fixed, which bleeds off the useful pressure on the labeled class. To name the fix I split losses by writing $L(f(x),y)=\sum_k \ell(f(x),k)$. A loss is active if $\ell(f(x),k)=0$ for all $k\neq y$, acting only at the labeled coordinate — CE, focal loss, NCE, normalized focal loss are active. A loss is passive if it has a nonzero off-label component, explicitly penalizing probability on wrong classes — MAE, normalized MAE, RCE, normalized RCE are passive. The underfitting of a robust active term is exactly its blindness to the off-label route the normalization opens; a passive term controls that route directly. And the two compose cleanly: if both terms are noise-tolerant, any positive linear combination is too, since $\sum_y(\alpha L_A+\beta L_P)(f,y)=\alpha C_A+\beta C_P$ is still constant. So an active–passive objective keeps the theorem while pulling in two complementary optimization directions.

For the active term I take NCE, the robust counterpart of CE that retains a CE-like dependence on the labeled probability. For the passive term I take RCE, which is already symmetric and therefore needs no normalization in this pairing. With $q_y=1$ and the off-label entries clamped so $\log q_k = A < 0$,
$$\mathrm{RCE} = -\sum_k p_k \log q_k = -A\sum_{k\neq y} p_k = -A(1-p_y),$$
whose class-sum over candidate labels is $\sum_y \mathrm{RCE}(f,y) = -A(K-1)$, a positive constant since $A<0$. (Had I chosen to normalize RCE the correct scaling would be division by $-A(K-1)$, giving $(1-p_y)/(K-1)$; here the unnormalized form is fine because scalar multiples of symmetric losses stay symmetric.) The gradient signs confirm the passive role: $\partial\,\mathrm{RCE}/\partial z_y = A\,p_y(1-p_y) \le 0$, so descent raises the labeled logit, while for a wrong logit $\partial\,\mathrm{RCE}/\partial z_j = -A\,p_y p_j \ge 0$, so descent lowers wrong logits — and the whole term is gated by $p_y$, strong only once the model already grants the provided label some probability and weak on examples it firmly rejects. The objective is therefore
$$L = \alpha\,\mathrm{NCE} + \beta\,\mathrm{RCE},$$
with uniform-noise class-sum $\alpha\cdot 1 + \beta\cdot(-A)(K-1)$, constant, so the same affine-risk proof carries through. Here $\alpha$ and $\beta$ are balancing weights, not a universal constant: equal weights $\alpha=\beta=1$ are the natural start and suffice on an easier setting like CIFAR-10, but many-class, harder settings need a larger active weight and a smaller passive weight to lean on the fitting term — for instance CIFAR-100 with $\alpha=10,\beta=0.1$ and WebVision-mini with $\alpha=50,\beta=0.1$ — so the implementation exposes $\alpha$ and $\beta$ rather than baking in one pair. In the reference code the NCE branch uses `F.log_softmax`, a one-hot label, numerator $-(\text{one\_hot}\cdot\text{log\_probs}).\text{sum}$ and denominator $-\text{log\_probs}.\text{sum}$; the RCE branch uses `F.softmax` with probabilities clamped at `1e-7` and one-hot labels clamped at `1e-4` (this clamp is the finite $A=\log(10^{-4})$), and the final scalar is the mean of each scaled term.

```python
import torch
import torch.nn.functional as F


class RobustLoss:
    """Active Passive Loss: normalized cross entropy plus reverse cross entropy."""

    def __init__(self, num_classes, alpha=1.0, beta=1.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def _one_hot(self, labels, logits):
        return F.one_hot(labels, self.num_classes).to(dtype=logits.dtype, device=logits.device)

    def _nce(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        one_hot = self._one_hot(labels, logits)
        numerator = -(one_hot * log_probs).sum(dim=1)
        denominator = -log_probs.sum(dim=1)
        return (numerator / denominator).mean()

    def _rce(self, logits, labels):
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = self._one_hot(labels, logits).clamp(min=1e-4, max=1.0)
        return (-(probs * torch.log(one_hot)).sum(dim=1)).mean()

    def compute_loss(self, logits, labels, epoch):
        return self.alpha * self._nce(logits, labels) + self.beta * self._rce(logits, labels)
```
