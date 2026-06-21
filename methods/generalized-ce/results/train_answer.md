The problem is training a deep softmax classifier when some fraction of the training labels have been flipped to wrong classes. High-capacity networks can memorize arbitrary label assignments, so if we simply minimize ordinary cross entropy the model will eventually fit the corrupted labels and clean-test accuracy will collapse. The constraint is minimal intervention: we want robustness to come only from swapping the loss function, with no extra networks, confusion matrices, clean validation sets, or changes to the optimizer or architecture.

The two standard losses point to opposite failures. Categorical cross entropy, L = -log f_y, has gradient -(1/f_y) ∇_θ f_y, so samples where the model assigns low probability to the given label get an amplified gradient. On clean data this hard-example weighting is beneficial, but under label noise the poisoned examples are exactly the low-confidence ones, so cross entropy drives memorization of the corrupted labels. Mean absolute error, L_MAE = 2 - 2 f_y (equivalently the unhinged loss 1 - f_y), is symmetric and therefore provably noise-tolerant under uniform noise, but its flat gradient gives no extra pull on hard examples, causing slow optimization and poor final accuracy on difficult datasets.

The method I propose is Generalized Cross Entropy (GCE), also called the L_q loss. It is defined as L_q(f(x), e_y) = (1 - f_y^q) / q for q in (0, 1]. This single parameter continuously interpolates between the two baselines: taking q -> 0 recovers categorical cross entropy by L'Hopital's rule, and q = 1 gives the unhinged/MAE loss. The per-sample gradient is ∂L_q/∂θ = -f_y^{q-1} ∇_θ f_y = f_y^q · (-(1/f_y) ∇_θ f_y). Read as f_y^q times the cross-entropy gradient, it down-weights low-confidence samples, which are the likely-noisy ones. Read as f_y^{q-1} times the MAE gradient, it up-weights hard samples relative to MAE, preserving enough learning signal to train deep networks on hard data. Thus q is a direct robustness-versus-learnability knob: larger q is more noise-robust but harder to optimize, smaller q trains more easily but is less robust.

The noise-tolerance guarantee follows from bounding the symmetry defect. A loss is symmetric if the sum over classes of L(f(x), j) is constant; symmetric losses are noise-tolerant because the noisy risk becomes an affine increasing transform of the clean risk. L_q is not exactly symmetric except at q = 1, but for q in (0,1] its class-sum is bounded between (c - c^{1-q})/q and (c - 1)/q. Pushing these bounds through the uniform-noise risk expansion shows that the clean-risk gap between the noisy-optimum classifier and the true clean-optimum classifier is controlled by a factor proportional to c^{1-q} - 1, which vanishes at q = 1 and widens as q approaches 0. The same tradeoff appears on both the gradient and risk sides, confirming that q is the right interpolation parameter. Empirically, q = 0.7 strikes a good compromise on standard image benchmarks: it suppresses overfitting to label noise while keeping convergence close to that of cross entropy.

```python
import torch


class RobustLoss:
    """Generalized Cross Entropy (L_q) loss for classification under label noise.

    L_q(f(x), e_y) = (1 - f_y^q) / q, averaged over the minibatch. f_y is the
    softmax probability assigned to the given (possibly corrupted) label. q in
    (0, 1] interpolates categorical cross entropy (q -> 0) and MAE/unhinged loss
    (q = 1). The per-sample gradient is f_y^q times the CCE gradient, which
    down-weights low-confidence (likely-noisy) samples while preserving enough
    pull on hard examples to train deep networks.
    """

    def __init__(self):
        self.q = 0.7  # robustness <-> learnability tradeoff; tune by validation accuracy

    def compute_loss(self, logits, labels, epoch):
        probs = torch.softmax(logits, dim=1)                  # f(x): class probabilities
        p = probs.gather(1, labels[:, None]).clamp_min(1e-8)  # f_y, floored for stability
        return ((1.0 - p.pow(self.q)) / self.q).mean()        # (1 - f_y^q)/q, mean over batch
```
