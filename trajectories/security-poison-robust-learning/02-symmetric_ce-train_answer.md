The bootstrap numbers sharpened the diagnosis exactly along the line I worried about. On FashionMNIST the soft bootstrap was a clean no-op ($0.9480 / 0.0076 / 0.9702$), and on CIFAR-10 nearly flat ($0.9025 / 0.1142 / 0.8942$), a hair below cross entropy. But on CIFAR-100 it *failed*: $\text{poison\_fit} = 0.5642$, $\text{robust\_score} = 0.5586$ — worse than plain cross entropy ($0.4532 / 0.6187$). More than half of the poisoned VGG examples are predicted as their flipped label. This is precisely the circularity I derived: the target $t = \beta e_y + (1-\beta)\hat p$ can only distrust a label to the degree the model already does, and on a hard, low-signal 100-class problem the model takes many epochs to build confident correct predictions, so for a long window $\hat p$ leans toward the class it is being trained on — the flipped one — and the detached self-vote *endorses* the memorized lie. The lesson is unambiguous: the next rung's resistance to noise cannot come from the model's own belief. It has to be a property of the *loss* that holds independently of what the model predicts.

So I go back to the only loss-level criterion that gives a model-independent guarantee: a loss is noise-tolerant if it is *symmetric*, $\sum_{k=1}^{K} L(f,k) = C$ for a constant independent of $x$ and $f$. The algebra is the template for everything that follows. Under uniform noise at rate $\eta$ the noisy risk is

$$R^{\eta}(f) = \mathbb{E}_x \mathbb{E}_{y|x}\Big[(1-\eta)L(f,y) + \tfrac{\eta}{K-1}\sum_{k\neq y}L(f,k)\Big],$$

and because the total over all classes is constant, $\sum_{k\neq y}L = C - L(f,y)$, which substitutes to give $R^{\eta}(f) = \frac{C\eta}{K-1} + \big(1 - \frac{\eta K}{K-1}\big)R(f)$ — the noisy risk is a positive affine map of the clean risk (positive multiplier as long as $\eta < 1 - 1/K$), so its minimizer is the clean minimizer. No dependence on data, on a clean validation set, or on the model's state. That is the independence the bootstrap lacked. Cross entropy fails outright ($\sum_k -\log f_k$ is unbounded), and mean absolute error passes but its gradient $-2f_y(1-f_y)$ vanishes as $f_y\to 0$, so it refuses to push the very hard examples and trains too slowly for a 100-class problem — and CIFAR-100 just reminded me how unforgiving it is about convergence. So I cannot simply *swap in* a symmetric loss; I would lose exactly the convergence the hard benchmark most needs.

That argues for a different shape than the bootstrap's. The bootstrap deformed the target; here I want to *keep cross entropy for what it is good at* — dense, well-conditioned gradients — and *add* a separate symmetric term that supplies tolerance, rather than trade one for the other. So I propose **Symmetric Cross Entropy**, the objective $\ell = \alpha\cdot\text{CE} + \beta\cdot\text{RCE}$, where the second term is Reverse Cross Entropy. Where this term comes from: cross entropy is $H(q,f) = -\sum_k q_k \log f_k$, corresponding to $\mathrm{KL}(q\|f)$, which trusts $q$ as ground truth — but $q$ is the untrustworthy object. So measure the discrepancy the other way too, $\mathrm{KL}(f\|q)$, and symmetrize; the cross-entropy analogue of the mirror term is the same formula with the arguments swapped,

$$\ell_{\text{rce}} = -\sum_{k=1}^{K} f_k \log q_k.$$

Now $q$ is the one-hot label, so $\log q_{k\neq y} = \log 0 = -\infty$ and the term is ill-defined. The fix is to clamp — define $\log 0 := A$ for a finite negative constant, realized by flooring the one-hot at a small positive value before the log — and that clamp is not numerical hygiene, it is the load-bearing piece. With $\log q_y = 0$ and $\log q_{k\neq y} = A$,

$$\ell_{\text{rce}} = -A\sum_{k\neq y} f_k = -A(1 - f_y),$$

so $\ell_{\text{rce}}(f,i) = -A(1-f_i)$ and $\sum_i \ell_{\text{rce}}(f,i) = -A(K-1)$, a constant independent of $x$ and $f$. So RCE is symmetric with $C = -(K-1)A$ and, by the affine-risk argument, noise-tolerant for $\eta < 1 - 1/K$. (Note $-A(1-f_y)$ is MAE up to the multiplier: $A=-2$ recovers MAE exactly, so RCE is a one-parameter generalization of a known robust loss — confirmation I am in the right family.)

What makes this the right answer to the bootstrap's failure is that RCE's robustness holds for *every* $f$ — the symmetry is a property of the loss summed over classes, true at initialization and convergence alike, never waiting on the model learning to disagree. And its gradient actually *helps* the under-learned examples rather than adding inert tolerance. Differentiating $\text{CE} + \text{RCE}$ through the softmax Jacobian: on the labeled-class logit the RCE contribution is $A f_y(1-f_y) \le 0$ (since $A<0$), which *adds* to cross entropy's downhill push and is largest in magnitude at $f_y\approx 0.5$ — exactly the half-learned examples — tapering as $f_y\to 1$ so already-learned points are not hammered into overfitting; on a wrong-class logit it is $-A f_j f_y \ge 0$, suppressing residual wrong-class mass but *scaled by $f_y$*, so when the model is not buying the label at all ($f_y\approx 0$, what a flipped label looks like from the inside) the suppression goes quiet. RCE *self-gates* — it sharpens predictions the model already half-believes and stays silent on the examples it implicitly flags as mislabeled, the structural opposite of the bootstrap that *endorsed* whatever the model believed.

On the coefficients: $\ell = \alpha\cdot\text{CE} + \beta\cdot\text{RCE}$ uses two separate weights, not a single $\alpha, 1-\alpha$ mix, because they do different jobs — $\alpha$ scales CE, the source of overfitting (lowering it eases memorization), and $\beta$ scales RCE, the robust pressure (raising it adds tolerance). The generic recipe tunes these per dataset, but the harness fixes one objective across all three benchmarks and exposes no per-benchmark hook. Since CIFAR-100's hard convergence is the binding constraint, I keep $\alpha = 1.0$ — the full cross-entropy gradient for convergence — and add a moderate $\beta = 0.5$ of RCE for tolerance, not a large $\beta$ that would slide toward MAE's vanishing gradient. The constant $A$ is not free either: it is realized as $A = \log(10^{-4}) \approx -9.21$ by clamping the one-hot up to $10^{-4}$ before its log, and the prediction inside RCE is clamped at $10^{-8}$ for stability. No per-dataset tuning, no use of the epoch counter — the contract exposes neither.

```python
class RobustLoss:
    """Cross-entropy plus reverse-CE penalty."""

    def __init__(self):
        self.alpha = 1.0
        self.beta = 0.5

    def compute_loss(self, logits, labels, epoch):
        ce = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
        one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float().clamp_min(1e-4)
        rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
        return self.alpha * ce + self.beta * rce
```
