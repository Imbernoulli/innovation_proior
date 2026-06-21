The confidence penalty confirmed both halves of my worry, and the CIFAR-100 number is the headline. On **resnet20-cifar10** the adaptivity paid off — attack AUC dropped to $0.7130$ (below smoothing's $0.7678$ and even ERM's $0.7275$) at preserved $0.7974$ accuracy, privacy_score up to $0.5844$. On **mobilenetv2-fmnist** a clean small win ($0.8671$). But on **vgg16bn-cifar100** the run *collapsed*: test_acc $0.01$ — chance on a 100-class problem — with a meaningless privacy_score of $0.01$, exactly the failure I flagged, the constant unscheduled entropy push preventing the high-capacity model from ever committing while the learning rate decayed. So I now have the full shape of the wall. Shifting confidence uniformly (smoothing) does nothing to separability; shifting it adaptively (confidence penalty) dents the attack on easy cases but never breaks it; and pushing a mean-control penalty hard enough to matter on the hardest case destabilizes training. Not one of these regularizers made the member and non-member distributions *overlap* — they only relocated, compressed, or destroyed the member one. I need a method that targets overlap directly, without the collapse risk.

So I restart from what the attacker actually reads, because that is the only thing worth fighting. There is a clean result that the attack is essentially one-dimensional. Model the trained parameters as a posterior $P(\theta \mid z_1,\dots,z_n) \propto \exp\!\big(-\tfrac{1}{T}\sum_i \ell(\theta, z_i)\big)$, single out one sample $z_1$, and ask for the Bayes-optimal membership guess; the dependence on $\theta$ collapses, and the optimal score is $s(z_1) = \tfrac{1}{T}\big(\tau(z_1) - \ell(\theta, z_1)\big)$, where $\tau(z_1)$ is the typical loss $z_1$ incurs under models that did *not* train on it. The optimal attacker — even white-box — reads exactly one number: the sample's loss $\ell(\theta, z_1)$, and the confidence the harness thresholds is its monotone proxy. A second result points the same way: Yeom and colleagues showed the simplest bounded-loss attacker has membership advantage equal to the *generalization gap* $R_{\mathrm{gen}} = \mathbb{E}_{\mathrm{test}}[\ell] - \mathbb{E}_{\mathrm{train}}[\ell]$. So the whole job reduces to a statement about *distributions*: let $P$ be the per-sample loss distribution over members and $Q$ over non-members; if a threshold separates them the attacker wins, if they overlap it cannot. Defense = make $P$ and $Q$ indistinguishable. That reframes every failure above: plain cross-entropy crushes $P$ to a tight spike near zero, well separated from $Q$ (the $0.84$–$0.87$ AUCs), and smoothing and the confidence penalty *shifted or compressed* $P$ but left it a recognizable, separable hump. I have only ever moved $P$ around; I need it to *bleed into* $Q$.

Two things make two humps hard to tell apart: the distance between their means and their widths. And here is the structural fact I had not used — the non-member distribution $Q$ already has the *larger* variance (never optimized, its losses scatter), while $P$ is a tight spike near zero: the member hump is narrow *and* low, doubly easy to threshold. So I want two moves at once — bring $P$'s mean *up* toward $Q$'s, *and* spread $P$ out so it stops being a spike. The prior rungs only ever attempted the first, and rigidly; the second lever, variance, is the one nobody pulled, and it is what produces overlap.

I propose **RelaxLoss** (Chen, Yu, Fritz, ICLR 2022). The mean part simplifies drastically. Manipulating full distributions is hopeless — estimating $Q$ would need a hold-out non-member set a real defender lacks — so I target the *mean*. Pick a target loss level $\alpha > 0$ that is *achievable for non-members too*, not the unreachable zero, and aim the average member loss at $\alpha$. While the batch mean loss is above $\alpha$ the model is genuinely undertrained, so descend; once it drops *below* $\alpha$, do not keep descending (that re-opens the gap) but push it back *up* to $\alpha$, which is gradient *ascent*. Both directions fold into one objective the fixed optimizer just minimizes:

$$\text{loss} = |L - \alpha|.$$

When $L > \alpha$, $\frac{d|L-\alpha|}{d\theta} = +\nabla L$ (descent); when $L < \alpha$, it is $-\nabla L$, which under the minimizer's subtract-the-gradient rule is ascent. One absolute-value loss is the whole thermostat: descend above $\alpha$, ascend below, settle at $\alpha$. Crucially this is *not* the confidence penalty's unbounded flattening — it does not keep pushing the output toward uniform once the loss is high enough; it *holds* the loss at a finite, non-degenerate $\alpha$. That is the relief valve CIFAR-100 lacked: the objective stops fighting once the member loss reaches $\alpha$, so it cannot drive the model to chance.

Now the part the prior rungs never reached: does ascent also *spread* $P$? Take one sample's loss change under the batch ascent step $\theta \to \theta + \tau\nabla L$ and Taylor-expand,

$$\Delta \ell_i = \frac{\tau}{B}\|\nabla \ell_i\|^2 + \frac{\tau}{B}\sum_{j\neq i}\langle \nabla \ell_i, \nabla \ell_j\rangle + o(\tau),$$

which under comparable gradient norms and non-negative alignment is $\Delta\ell_i \approx c_1\|\nabla\ell_i\|^2 + c_2$ with $c_1 > 0$. Connect the norm to the loss: for cross-entropy $\nabla_\theta \ell = J^T(p - y)$, so as a sample's loss shrinks, $p \to y$, $\|p - y\| \to 0$, and $\|\nabla\ell\|^2 \to 0$ — the gradient norm is small for low-loss samples and large for high-loss ones, so $\mathrm{Cov}(\|\nabla\ell\|^2, \ell) > 0$. Chaining, $\mathrm{Cov}(\ell, \Delta\ell) = c_1\,\mathrm{Cov}(\ell, \|\nabla\ell\|^2) > 0$: high-loss samples get pushed up *more* than low-loss ones, and that is variance-increasing, since $\mathrm{Var}(\ell + \Delta\ell) = \mathrm{Var}(\ell) + \mathrm{Var}(\Delta\ell) + 2\mathrm{Cov}(\ell, \Delta\ell) > \mathrm{Var}(\ell)$. The ascent step *fattens* $P$ — it stretches the tail, not just the mean. This is the lever the mean-shifting regularizers structurally cannot pull: smoothing and the confidence penalty push every member up by roughly the same amount, leaving $\mathrm{Var}(P)$ essentially untouched (which is *why* their distributions stayed separable), whereas ascent pushes the already-high losses up harder until $P$ overlaps the wider $Q$. Both levers — mean toward $\alpha$, variance up — fall out of the one absolute-value objective.

That this lowers the attack is not hand-waving. Treat the attack as a hypothesis test $P$ vs $Q$; the optimal attacker's AUC is bounded by the total-variation distance, $\text{AUC} \le -\tfrac12 D_{\mathrm{TV}}(P,Q)^2 + D_{\mathrm{TV}}(P,Q) + \tfrac12$, increasing in $D_{\mathrm{TV}}$ on $[0,1]$. Bound $D_{\mathrm{TV}} \le \sqrt{2}\,D_H$ by the Hellinger distance, which for Gaussians $P\sim N(\mu_1,\sigma_1^2)$, $Q\sim N(\mu_2,\sigma_2^2)$ has a closed form. Shrinking the mean gap $(\mu_1 - \mu_2)^2$ and raising the member variance $\sigma_1^2$ (with $\sigma_2$ fixed) both raise the Hellinger exponential factor toward $1$, and since $Q$ started wider ($\sigma_2 \ge \sigma_1$), fattening $P$ moves the variance ratio toward equality where the prefactor is maximized. Both levers shrink $D_H$, hence $D_{\mathrm{TV}}$, hence the AUC ceiling — exactly the two moves I derived, against exactly the metric the harness reports.

Now the cost, because relaxing member loss threatens utility in a specific way. If I let $p_{\mathrm{gt}}$ (the predicted probability on the true class) sit lower and the leftover mass $1 - p_{\mathrm{gt}}$ happens to concentrate on one competing class — as happens on hard samples near a decision boundary — that rival can exceed $p_{\mathrm{gt}}$ and the argmax flips: I have relaxed confidence into misclassification. The fix must *preserve* the privacy gain (do not re-sharpen $p_{\mathrm{gt}}$ toward $1$, or I undo the defense) while protecting the argmax. The argmax is safe as long as $p_{\mathrm{gt}}$ beats every competitor, so at *fixed* $p_{\mathrm{gt}}$ I maximize the margin by making the largest rival as small as possible — spread $1 - p_{\mathrm{gt}}$ *evenly* over the $K-1$ wrong classes, each getting $(1 - p_{\mathrm{gt}})/(K-1)$. Build that target and train toward it:

$$t^c = p_{\mathrm{gt}}\ \text{if}\ c\ \text{is the true class},\qquad t^c = \frac{1 - p_{\mathrm{gt}}}{K - 1}\ \text{otherwise}.$$

This is *posterior flattening*: do not change how confident I am in the truth, only equalize the doubt so no single rival can overtake. Two details are load-bearing. The target $t$ is built from the model's *own* current $p_{\mathrm{gt}}$, read off the forward pass, so it must be a *constant* — apply stop-gradient (detach) to $t$, or the gradient would flow back through the $p_{\mathrm{gt}}$ inside $t$ and start re-sharpening the very confidence I am leaving alone. And *which* samples: a correctly-classified sample's argmax is already fine, so flattening it spends effort on rivals that are not winning and perturbs a settled prediction — the samples that need margin protection are the *misclassified* ones, so I gate the soft cross-entropy by $(1 - \text{correct}_i)$. Meanwhile I keep pushing the loss up by subtracting the per-sample cross-entropy, so the per-sample objective is $\text{loss}_i = (1 - \text{correct}_i)\cdot \ell_{\mathrm{soft},i} - \ell_{\mathrm{CE},i}$, batch-mean: the $-\ell_{\mathrm{CE},i}$ term is ascent (keeps the loss elevated and spreading), the gated $\ell_{\mathrm{soft},i}$ fixes the argmax of exactly the samples that need it, at fixed $p_{\mathrm{gt}}$.

Assembled into the two-phase rule the scaffold lands, this is the first method that *needs* the `epoch` argument the loop hands me. On **even epochs** I use the thermostat directly, $\text{loss} = |L - \alpha|$ — descent above $\alpha$, ascent with its free variance-spreading below. On **odd epochs**, if $L > \alpha$ the model is still undertrained so I descend, $\text{loss} = L$; only when $L \le \alpha$ does the posterior-flattening branch run, $\text{loss} = \text{mean}_i[(1 - \text{correct}_i)\cdot\ell_{\mathrm{soft},i} - \ell_{\mathrm{CE},i}]$. Alternating keeps the flattening branch from swamping the ascent step. Two hyperparameters: `upper` clamps $p_{\mathrm{gt}}$ into $[0, \text{upper}]$ before building $t$, and with $\text{upper} = 1.0$ it is a no-op (it cannot clamp a probability below $1$), kept for faithfulness to the reference. The real knob is $\alpha$, a loss level reachable by non-members, selected here by class count read from `logits.size(1)`: $\alpha = 0.5$ when $K = 100$ (CIFAR-100) and $\alpha = 1.0$ otherwise (the 10-class CIFAR-10 and FashionMNIST cases). That per-dataset $\alpha$ is the one place the method adapts, read at call time from the logits, not configured externally — exactly what the fixed `compute_loss` signature allows.

The falsifiable expectations against the prior numbers. The decisive prediction is on **vgg16bn-cifar100**, where the confidence penalty collapsed to $0.01$: the thermostat holds the member loss at a finite $\alpha = 0.5$ rather than driving it degenerate, and the flattening branch protects the argmax of misclassified samples, so accuracy should *not* collapse — I expect it to recover to genuine, non-trivial accuracy (well above ERM's $0.5045$, plausibly into the low-$0.6$s as the relaxation acts as a regularizer) while the variance-spreading drags the AUC well down from the $0.86$–$0.87$ the mean-control rungs were stuck at, toward the mid-$0.6$s — several-fold above the wreckage and the ERM floor. On **resnet20-cifar10** I expect the attack to drop hardest of all (overlap, not a shift), plausibly near the $0.5$ coin-flip floor at roughly ERM-level accuracy, so privacy_score clears every prior rung's $\sim 0.57$–$0.58$ by a wide margin and approaches test_acc itself. On **mobilenetv2-fmnist**, where leakage was already mild, I expect the AUC at or near $0.5$ (the privacy term essentially zero), though the relaxation may cost some accuracy on an already-easy task. The clean signature distinguishing this method from the three that failed: the attack AUC moving *toward 0.5* — overlap — rather than the member distribution merely relocating, and CIFAR-100 *not* collapsing. If those hold, the variance lever is what broke the wall.

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """RelaxLoss training rule (Chen et al., ICLR 2022).

    Even epochs: loss = |mean_CE - alpha|   (drive loss toward target level)
    Odd  epochs: if mean_CE > alpha -> CE descent
                 else -> posterior flattening with sign-flipped CE
    """

    def __init__(self):
        self.upper = 1.0

    def compute_loss(self, logits, labels, epoch):
        num_classes = logits.size(1)
        alpha = 0.5 if num_classes == 100 else 1.0

        loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        loss_ce = loss_ce_full.mean()

        if epoch % 2 == 0:
            return (loss_ce - alpha).abs()

        if loss_ce.item() > alpha:
            return loss_ce

        probs = torch.softmax(logits, dim=1)
        confidence_target = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        confidence_target = torch.clamp(confidence_target, min=0.0, max=self.upper)
        confidence_else = (1.0 - confidence_target) / (num_classes - 1)

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        soft_targets = (
            onehot * confidence_target.unsqueeze(1)
            + (1.0 - onehot) * confidence_else.unsqueeze(1)
        )
        soft_targets = soft_targets.detach()

        log_probs = F.log_softmax(logits, dim=1)
        ce_soft = -(soft_targets * log_probs).sum(dim=1)

        pred = logits.argmax(dim=1)
        correct = pred.eq(labels).float()

        loss = (1.0 - correct) * ce_soft - loss_ce_full
        return loss.mean()
```
