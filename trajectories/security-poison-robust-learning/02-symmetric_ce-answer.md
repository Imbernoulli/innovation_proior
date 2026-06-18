**Problem (from step 1).** The bootstrap's resistance to noise was sourced from the model's own (detached) prediction, which the corrupt labels distort — so on CIFAR-100, where the model memorizes fastest, the self-vote *endorsed* the flipped labels (`poison_fit 0.5642`, worse than plain cross entropy). Robustness cannot come from the model's belief; it must be a property of the loss that holds independently of what the model predicts.

**Key idea.** Use the formal criterion: a loss is noise-tolerant if it is *symmetric* (`Σ_k L(f,k) = C`). Keep cross entropy for its convergence-driving gradients and *add* a symmetric robust term — Reverse Cross Entropy, the mirror of CE with prediction and label swapped, `ℓ_rce = −Σ_k f_k log q_k`. The one-hot label forces `log 0`, defined as a finite constant `A` by clamping the label up to a small floor; that very clamp makes `Σ_k ℓ_rce(f,k) = −(K−1)A` constant, hence RCE symmetric and provably tolerant for `η < 1 − 1/K`. The full objective is `α·CE + β·RCE`.

**Why it works.** RCE's symmetry holds for *every* `f`, so its noise tolerance does not wait on the model learning to disagree — the structural opposite of the bootstrap. Its gradient also helps convergence: it adds `A f_y(1−f_y) ≤ 0` to the labeled-class logit (largest at `f_y ≈ 0.5`, the half-learned examples), and `−A f_j f_y ≥ 0` on wrong-class logits, *gated by `f_y`* so it suppresses wrong-class mass only on examples the model already half-believes and stays silent on likely-mislabeled ones. CE supplies the dense gradients that a hard problem like CIFAR-100 needs to converge at all.

**Hyperparameters.** `α = 1.0` (full CE — the harness fixes one objective across all three benchmarks and CIFAR-100's hard convergence is the binding constraint, so I do not starve the CE gradient), `β = 0.5` (moderate robust term; a large `β` slides toward MAE's vanishing gradient). `A = log(1e-4) ≈ −9.21`, fixed by `one_hot.clamp_min(1e-4)`; the prediction inside RCE is `softmax(logits).clamp_min(1e-8)`. No per-dataset tuning and no `epoch` use — the contract exposes neither.

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
