The generalized cross entropy numbers confirm the source-attenuation hypothesis and, in the same breath, expose its limit. On CIFAR-100 $\text{poison\_fit}$ fell to $0.0264$ — memorization almost entirely shut off, far below the symmetric loss's $0.18$ and cross entropy's $0.45$ — but $\text{test\_acc}$ dropped to $0.6674$, *below* plain cross entropy's $0.6907$ and the symmetric loss's $0.7032$. The $\text{robust\_score}$ rose to $0.8205$ only because the poison_fit collapse dominates the average. That is an underfitting signature, and it is structural, not noise: $q=0.7$ slid the loss most of the way to the MAE end, where the gradient flattens and the hard, under-learned classes never reach their clean-label ceiling. The pattern across the whole ladder is now legible — every rung that bought more robustness paid in fitting, because each blended a robust (symmetric) loss with a non-robust fitting term and tuned the balance. The gap left is an underfitting gap, and I want to close CIFAR-100's poison_fit *without* sacrificing the clean accuracy GCE gave up.

The trap I keep falling into is treating symmetry as a property a loss either has or lacks, then mixing a symmetric (robust, underfitting) loss with a non-symmetric (fitting, memorizing) one and tuning the blend — GCE with the single knob $q$, the symmetric loss with $\alpha\,\text{CE}+\beta\,\text{RCE}$. Every such blend reintroduces the non-robustness in proportion to how much fitting I want back. The constructive question I have not asked: instead of *finding* a symmetric loss, can I *make* any loss symmetric? Read the condition $\sum_{k=1}^K L(f,k)=C$ literally — the only thing between an arbitrary loss and symmetry is its class-sum, which is a quantity I can compute. So divide it out: the normalized loss

$$L_{\text{norm}}(f,y) = \frac{L(f,y)}{\sum_{j} L(f,j)}$$

has class-sum identically $1$ for every $x$ and $f$, hence is noise-tolerant for *any* base loss by the same affine-risk theorem. Normalize cross entropy and it becomes robust while keeping its labeled-class up-push. But I have to check whether normalization smuggles the underfitting back in — and it does, which is what unlocks the finale.

Name the structural axis I have been circling. At the level of the softmax outputs, a classification loss does one of two things. It can push *up* the probability at the labeled position — all $-\log f_y$ does is maximize $f_y$, saying nothing about how residual mass is spread; call that **active**. Or it can push *down* the probability at the non-labeled positions — $2(1-f_y)=2\sum_{k\neq y}f_k$ is manifestly a penalty on wrong-class mass; call that **passive**. Cross entropy, normalized or not, is active; MAE and RCE are passive. And each kind alone underfits the half it ignores: an active loss has no direct lever on how wrong-class mass is distributed, so under noise it can be "satisfied" (high $f_{\tilde y}$) while the true class still carries mass; a passive loss has no term maximizing the labeled class, so it commits slowly. GCE's CIFAR-100 underfitting *is* this — pushed toward MAE it leans passive, its active up-push weakens, and the hard classes never reach their ceiling.

So I propose **Active Passive Loss**, the combination of a normalized *active* loss with a *passive* loss, $\alpha\,L_{\text{active}} + \beta\,L_{\text{passive}}$, and the concrete instance is **NCE + RCE**. The combination is *still symmetric*, because a positive combination of symmetric losses is symmetric ($\sum_k(\alpha L_1+\beta L_2)(f,k)=\alpha C_1+\beta C_2$), so robustness is fully preserved — and decisively, *both* terms are robust, so unlike the symmetric loss's $\alpha\,\text{CE}+\beta\,\text{RCE}$ there is no non-robust active term reintroducing memorization. Meanwhile the two complementary fitting behaviors — active pushing the right class up, passive pushing the wrong classes down — *jointly* concentrate the prediction the way neither can alone, which is exactly the fitting GCE sacrificed. The tolerance is now structural (every term symmetric) rather than a balance point on a single knob.

This is why it is genuinely different from the symmetric-loss rung even though $\alpha\,\text{NCE}+\beta\,\text{RCE}$ looks like the same shape. There the active term was *plain* cross entropy — un-normalized, non-symmetric, full $1/f_y$ amplification intact — so RCE had to act as a counterweight against an active term that was *itself* generating memorization, and $\beta=0.5$ was a truce, not a cure. Here the active term is *normalized* cross entropy, which is symmetric and carries no memorization pressure to counterweight: RCE is no longer fighting the active term, it is *completing* it. Both pull in the same robust direction — NCE concentrates the labeled class without the $1/f_y$ blow-up on poisoned examples (the class-sum denominator damps exactly the low-confidence updates), RCE drives down wrong-class mass — and neither needs holding back to protect against the other. That is why I can take $\alpha=\beta=1$ without a tuned truce: there is no opposing force to balance. The symmetric loss could not, because its full-weight CE term would simply memorize; it *needed* a small $\alpha$. Normalization is what frees the coefficients.

The two terms concretely. NCE is the active term,

$$L_{\text{NCE}} = \frac{\sum_k q_k \log f_k}{\sum_j \log f_j},$$

computed from $\mathrm{log\_softmax}$ with the denominator the class-sum $\sum_j -\log f_j$; the denominator is load-bearing, large exactly when the prediction is diffuse and shrinking as the model commits, so it rescales the per-sample loss by the model's overall uncertainty rather than by the single labeled probability — which removes the $1/f_y$ over-weighting of poisoned (low-$f_y$) examples while keeping a real gradient on genuinely hard clean ones. That restores cross entropy's hard-example pull, the dense gradients CIFAR-100 needs to lift the hard classes back to their ceiling, exactly what GCE gave up at $q=0.7$. RCE is the passive term, the same $-A(1-f_y)$ from the symmetric-loss rung (with $A=\log 0$ realized by clamping the one-hot to a small floor), already symmetric so it serves as the passive piece directly; its logit gradient self-gates, adding $A f_y(1-f_y)\le 0$ on the labeled logit (largest at $f_y\approx 0.5$, the half-learned classes GCE under-pushed) and $-A f_j f_y\ge 0$ on wrong logits scaled by $f_y$, so it suppresses wrong-class mass only on examples the model already half-believes and goes silent on the ones it flags as mislabeled. NCE supplies the up-push and the convergence, RCE the robust down-push, both noise-tolerant, no compromise term — the structural completion of the ladder: the symmetric loss kept a non-robust CE, GCE removed the non-robustness but slid into underfitting, and normalization plus active-passive removes the non-robustness *and* the underfitting at once.

It fits the contract exactly, as every rung had to: a per-minibatch scalar from logits, labels, epoch — and NCE+RCE is a pure function of the current batch's logits and labels, two normalized/symmetric terms summed, needing no sample indices, no model access, no cross-epoch state (which is why heavier noisy-label methods that co-train two networks or track per-sample loss would not fit this slot). The coefficients are the canonical untuned $\alpha=\beta=1.0$, equal weight on the up-push and down-push, since the harness fixes one global objective across all three benchmarks. The numerical guards: $\mathrm{log\_softmax}$ for NCE's stability, the softmax prediction clamped at $10^{-7}$ and the one-hot label at $10^{-4}$ for RCE (so $A=\log(10^{-4})$). The one adaptation from the usual $\texttt{\_\_init\_\_(num\_classes)}$ signature is that I read $\texttt{num\_classes}$ from $\texttt{logits.shape[1]}$, because the scaffold constructs the module with no arguments and the harness does not pass the class count.

```python
class RobustLoss:
    """Active Passive Loss: normalized cross-entropy + reverse cross-entropy."""

    def __init__(self):
        self.alpha = 1.0
        self.beta = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.shape[1]
        one_hot = F.one_hot(labels, num_classes).float()

        # Active term: normalized cross entropy (symmetric by construction)
        log_probs = F.log_softmax(logits, dim=1)
        nce_num = -(one_hot * log_probs).sum(dim=1)          # -log f_y
        nce_den = -log_probs.sum(dim=1)                      # sum_j -log f_j (class-sum)
        nce = (nce_num / nce_den).mean()

        # Passive term: reverse cross entropy (already symmetric; A = log(1e-4))
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        rce_target = one_hot.clamp(min=1e-4, max=1.0)
        rce = -(probs * torch.log(rce_target)).sum(dim=1).mean()

        return self.alpha * nce + self.beta * rce
```
