A classifier trained on a private dataset leaks which samples it was trained on, and I want a training-time procedure that drives a strong membership-inference attacker down toward a coin flip — an attack AUC near $0.5$ — without sacrificing test accuracy, cheaply, and robustly against attacks I did not anticipate. The reason this is hard is that the signal the attacker exploits is overfitting, and the obvious cure for overfitting costs utility. But before reaching for a cure I want to know precisely what quantity the attacker reads, because that is exactly what I have to flatten. There is a clean answer. Modeling the trained parameters as a posterior $P(\theta\mid z_1,\dots,z_n)\propto\exp\!\big(-\tfrac1T\sum_i m_i\,\ell(\theta,z_i)\big)$ and asking for the Bayes-optimal guess of a single sample's membership bit, the dependence on $\theta$ collapses: the score reduces to $s(z_1,\theta,p)=\tfrac1T\big(\tau_p(z_1)-\ell(\theta,z_1)\big)$, where $\tau_p(z_1)$ is the typical loss the sample incurs under models that did not train on it. The optimal attacker, even holding the full white-box model, reads off exactly one number — the per-sample loss $\ell(\theta,z_1)$ — and calls the point a member when its loss falls below a threshold. The weights, activations and gradients add nothing once the loss is known. A second result confirms this is the right target: Yeom and colleagues showed the membership advantage of the simplest bounded-loss adversary equals $R_{\text{gen}}/B$, the generalization gap, so with $0$–$1$ loss membership advantage literally is the gap between typical member and non-member loss. So my entire job is a statement about distributions. Let $P$ be the distribution of per-sample loss over members and $Q$ over non-members; if a threshold separates them the attacker wins, and if they overlap it cannot. Plain cross-entropy is what manufactures the separation: with $\ell_{\text{CE}}=-\sum_c y^c\log p^c$ and $p=\mathrm{softmax}$, the logit gradient is $p-y$, which vanishes as $p\to y$, so nothing stops the optimizer from grinding member losses to $\sim 0$ while never-optimized non-members sit higher — two well-separated humps, a loss-thresholding AUC around $0.84$ on CIFAR-10. The existing toolkit — early stopping, dropout, label smoothing, confidence penalty, distillation — either only slides along the same accuracy-versus-leakage curve or applies the *same* mean-shifting pressure to every sample: it translates the member hump to a higher loss but barely changes its spread, and slides $Q$ along with it, so the two distributions stay just as separable, only relocated. Adversarial regularization and MemGuard defend a surrogate attacker and collapse against a different metric attack; DP-SGD pays for a worst-case guarantee with visible accuracy loss and expensive per-sample clipping. None of them does the thing the distributional analysis says is necessary.

I propose RelaxLoss, which makes $P$ and $Q$ overlap with two coupled moves derived directly from that analysis. The first is to stop driving member loss to zero and instead aim its *mean* at a target $\alpha>0$ — a level that is achievable for non-members too — using a thermostat. As long as the batch mean loss $L$ is above $\alpha$ the model is genuinely undertrained, so I descend ordinarily; once $L$ drops below $\alpha$ I want to push it back up, which is gradient *ascent*, $\theta\leftarrow\theta+\tau\nabla L$. Both directions fold into one objective the fixed optimizer can simply minimize, $$\mathcal{L}=\lvert L-\alpha\rvert,$$ whose derivative is $+\nabla L$ when $L>\alpha$ (descent) and $-\nabla L$ when $L<\alpha$ (which under the minimizer's subtract-the-gradient rule becomes ascent), and whose magnitude even tapers the correction as $L$ nears $\alpha$. The crucial point is that this ascent does more than steer the mean — it spreads $P$ for free, and that is the lever the mean-shifting regularizers lack. Taylor-expanding one sample's loss change under the batch ascent step gives $\Delta\ell_i=\tfrac{\tau}{B}\lVert\nabla\ell_i\rVert^2+\tfrac{\tau}{B}\sum_{j\neq i}\langle\nabla\ell_i,\nabla\ell_j\rangle+o(\tau)$, and under the same-norm, non-negatively-aligned local approximation this collapses to $$\Delta\ell_i\approx c_1\lVert\nabla\ell_i\rVert^2+c_2,\qquad c_1=\tfrac{\tau}{B}\big(1+\textstyle\sum_{j\neq i}\cos a_{ij}\big)>0.$$ For cross-entropy $\nabla_\theta\ell_{\text{CE}}=J_\theta^{\top}(p-y)$, so as a sample's loss shrinks $p\to y$ and $\lVert\nabla\ell\rVert^2\to 0$: the gradient norm is small for low-loss samples and large for high-loss ones, i.e. $\mathrm{Cov}(\ell,\lVert\nabla\ell\rVert^2)>0$. Chaining the two, $\mathrm{Cov}(\ell,\Delta\ell)=c_1\,\mathrm{Cov}(\ell,\lVert\nabla\ell\rVert^2)>0$, so high-loss samples get pushed up *harder* than low-loss ones, and since $\mathrm{Var}(\ell+\Delta\ell)=\mathrm{Var}(\ell)+\mathrm{Var}(\Delta\ell)+2\,\mathrm{Cov}(\ell,\Delta\ell)$ with a strictly positive covariance, $\mathrm{Var}(\ell+\Delta\ell)>\mathrm{Var}(\ell)$. The member hump fattens rather than merely sliding. That this lowers the attacker's ceiling follows from a chain of bounds: treating the attack as a member-versus-non-member hypothesis test, $\mathrm{AUC}\le-\tfrac12 D_{\text{TV}}^2+D_{\text{TV}}+\tfrac12$, $D_{\text{TV}}\le\sqrt2\,D_H$, and for Gaussians $P\sim\mathcal N(\mu_1,\sigma_1^2)$, $Q\sim\mathcal N(\mu_2,\sigma_2^2)$ with $c=\sigma_2/\sigma_1$, $$D_H^2(P,Q)=1-\sqrt{\tfrac{2c}{1+c^2}}\,\exp\!\Big(-\tfrac14\tfrac{(\mu_1-\mu_2)^2}{(1+c^2)\sigma_1^2}\Big).$$ Shrinking the mean gap $(\mu_1-\mu_2)^2$ raises the exponential toward $1$; raising $\sigma_1^2$ with $\sigma_2$ fixed grows the total-variance denominator and, since $Q$ starts wider so $c\ge1$, drives the prefactor $\sqrt{2c/(1+c^2)}$ toward its maximum at $c=1$. The two levers together shrink $D_H$, hence $D_{\text{TV}}$, hence the AUC ceiling in the regime I actually operate in.

Relaxing the loss has a cost I must neutralize: holding $p_{\text{gt}}$ below $1$ means that on a hard sample the leftover mass $1-p_{\text{gt}}$ can pile onto one competing class and flip the argmax, turning relaxed confidence into a misclassification. I must not re-sharpen $p_{\text{gt}}$ — that would undo the privacy gain — so I protect the margin at *fixed* $p_{\text{gt}}$. The argmax is safe whenever $p_{\text{gt}}$ beats the largest competitor, and with $1-p_{\text{gt}}$ mass to spread over $C-1$ classes the way to minimize the maximum competitor is to spread it perfectly evenly, $(1-p_{\text{gt}})/(C-1)$ each. So I build a soft target that keeps the model's *own* current $p_{\text{gt}}$ on the truth and flattens the rest, and train toward it. Two details are load-bearing. The target is read off the forward pass, so it must be treated as a constant — a stop-gradient on $t$ — otherwise the gradient would flow back through the $p_{\text{gt}}$ inside $t$ and nudge exactly the confidence I am trying to leave alone; the match is then a soft cross-entropy $\ell_{\text{soft}}=-\sum_c \mathrm{sg}[t^c]\log p^c$. And only the *misclassified* samples need this margin repair, so I gate the soft term by $(1-\text{correct}_i)$; flattening an already-correct sample would only perturb a settled prediction. I still want to keep pushing the loss up during this branch, so I subtract the per-sample cross-entropy, giving $\text{loss}_i=(1-\text{correct}_i)\,\ell_{\text{soft},i}-\ell_{\text{CE},i}$, where the $-\ell_{\text{CE},i}$ term is ascent that keeps the loss elevated and spreading while the gated soft term fixes the argmax. The two near-target operations are alternated by epoch parity so the flattening branch does not swamp the ascent: even epochs minimize $\lvert L-\alpha\rvert$ directly (descent above $\alpha$, variance-spreading ascent below); odd epochs descend ordinary cross-entropy while $L>\alpha$ and only switch to posterior flattening once $L\le\alpha$. The single knob is $\alpha$, fixed from the class count as $0.5$ when $C=100$ and $1.0$ otherwise, and the retained $\text{upper}$ parameter clamps $p_{\text{gt}}$ into $[0,\text{upper}]$ before building $t$ — at $\text{upper}=1.0$ it is a no-op. The whole causal chain in one breath: the optimal attack sees only how $P$ differs from $Q$; cross-entropy crushes $P$ to a tight spike near zero; relaxing the mean to $\alpha$ and holding it with ascent both closes the mean gap and fattens $P$'s tail, the two moves that lower the optimal-attack ceiling; and detached posterior flattening on the misclassified protects accuracy without re-sharpening confidence.

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """RelaxLoss: relax the target loss to alpha, hold it with gradient ascent
    (which also spreads the member-loss variance), and flatten the
    non-ground-truth posterior to keep the argmax correct at a fixed p_gt."""

    def __init__(self):
        # clamps p_gt before building the soft target; 1.0 is a no-op
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        # single hyperparameter: a target loss level reachable by non-members
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()

        if epoch % 2 == 0:
            # thermostat: descend |L - alpha| -> ascent when below alpha
            return (loss_ce - alpha).abs()

        if loss_ce > alpha:
            # still undertrained -> ordinary cross-entropy descent
            return loss_ce

        # posterior flattening
        probs = torch.softmax(logits, dim=1)
        p_gt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_gt = torch.clamp(p_gt, min=0.0, max=self.upper)
        p_else = (1.0 - p_gt) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = onehot * p_gt.unsqueeze(1) + (1.0 - onehot) * p_else.unsqueeze(1)
        soft_targets = soft_targets.detach()                 # constant target

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)     # soft cross-entropy

        correct = logits.argmax(dim=1).eq(labels).float()
        # flatten only misclassified samples; subtract per-sample CE (ascent)
        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
```
