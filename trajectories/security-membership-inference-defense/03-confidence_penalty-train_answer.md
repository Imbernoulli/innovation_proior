Label smoothing landed almost exactly where I feared. On **resnet20-cifar10** it bought nothing — accuracy slipped to $0.7878$ and the attack AUC actually *rose* to $0.7678$ (from ERM's $0.7275$), so privacy_score fell to $0.5200$, below ERM's $0.5697$. On **vgg16bn-cifar100** the AUC barely moved ($0.8686$ versus ERM's $0.8677$, the same wall), and although privacy_score ticked up to $0.1572$ it did so off a tiny accuracy bump, not any real defense — the privacy_gap even widened to $0.2866$. On **mobilenetv2-fmnist** it was a small net loss ($0.8460$ vs $0.8685$). This is the mean-shift failure made concrete: smoothing relocated the member-confidence distribution but did not change its *separability* from the non-members, so the attack — which thresholds confidence and cares only about separability — was untouched, and I paid a little accuracy for nothing. The lesson is precise: bounding confidence *uniformly* is not enough, because smoothing treats a sample the model has already memorized exactly like one it is unsure about. I need pressure applied *where the over-confidence actually is*, adaptively.

So I re-derive the regularizer from the symptom rather than the cure. The disease shows up in the *output distribution*: an overfit network puts almost all its softmax mass on one class — a low-entropy spike — and that spike is the high-confidence signal the attack reads. Smoothing medicated this by editing the *target*, which acts identically on every example. What I want instead is to penalize the spike *itself*, in proportion to how spiked each output is. Entropy is the scalar that measures the spike: for a softmax output,

$$H(p) = -\sum_i p_i \log p_i,$$

maximal at uniform, zero at a one-hot spike. Over-confidence is low entropy; the cure is to push entropy up. So I propose the **confidence penalty** (Pereyra et al., 2017): take the cross-entropy I already minimize and *subtract* a multiple of the output entropy,

$$L = L_{\mathrm{CE}} - \beta\, H(p).$$

The sign is easy to flip, so let me nail it: I minimize $L$, and the term $-\beta H$ means that to make $L$ small I want $H$ *large* — less peaked outputs — so I am penalizing low entropy, penalizing confidence. The scaffold's `compute_loss` returns this directly: cross-entropy minus a weight times the mean per-sample entropy, with $\beta$ controlling how hard I push.

Whether this is surgical or a sledgehammer is settled by the gradient that flows back. Using the softmax Jacobian $\partial p_j / \partial z_i = p_j(\delta_{ij} - p_i)$,

$$\frac{\partial H}{\partial z_i} = -\sum_j p_j(\delta_{ij} - p_i)(\log p_j + 1),$$

and splitting the $\delta_{ij}$ and $-p_i$ pieces, with $\sum_j p_j(\log p_j + 1) = -H + 1$, this collapses to

$$\frac{\partial H}{\partial z_i} = p_i(-\log p_i - H).$$

Sit with that. The quantity $-\log p_i$ is the surprisal of class $i$ and $H$ is the *mean* surprisal, so the entropy gradient on logit $i$ is the deviation of $i$'s surprisal from the mean, weighted by $p_i$. Since I optimize $-\beta H$, descent moves *opposite* the entropy gradient. For the class the model is confident about, $p_i$ is large and $-\log p_i$ is small (below the mean), so $\partial H/\partial z_i$ is negative, the loss gradient positive, and the update pulls that logit *down*. For a class the model has nearly killed, $p_i \approx 0$ multiplies the whole term to nothing and the push is negligible. So this does *not* yank every dead class toward uniform; it acts mostly on the dominant, over-confident class. That is exactly the adaptivity label smoothing lacked — smoothing forced every wrong class toward the same $\varepsilon/K$, whereas the confidence penalty weights its pressure by the model's own current probabilities and so concentrates precisely on the spiked, memorized predictions generating the membership signal.

I can make the contrast exact, which nails *why* this should beat smoothing on the attack rather than just shifting things. Smoothing, up to a constant, adds the *forward* KL $D_{\mathrm{KL}}(u\|p) = \sum_i u_i \log(u_i/p_i)$, whose per-class weight is the *constant* $u_i = 1/K$ — equal pressure on every class. The confidence penalty is $-H(p)$, and $D_{\mathrm{KL}}(p\|u) = \sum_i p_i \log(p_i/u_i) = -H(p) + \log K$, so up to a constant it is the *reverse* KL to uniform, whose per-class weight is the model's own $p_i$ — large precisely on the classes it is currently confident about. Same target distribution (uniform), opposite KL direction; the reversal is the formal reason one is adaptive and the other uniform. Against a threshold attack, adaptive is what I want — flatten the spikes that leak, not relocate every output.

An honesty check, because this is the rung where I learn whether *any* mean-control regularizer can break the attack or only buy stability. The confidence penalty is still, at bottom, a device that lifts the confidence floor by pushing spiked outputs toward uniform. It is more targeted than smoothing, so I expect it to dent the attack where smoothing could not, especially on the easy and moderate benchmarks. But it has no explicit lever on the *variance* of the member-confidence distribution: it compresses the high end of the member hump without deliberately *spreading* members out to overlap the non-members. So I expect a mean-region intervention that helps but, the harder the dataset, may not be enough.

There is a sharper, CIFAR-100-specific worry, from a tension the entropy bonus carries. In reinforcement learning the entropy bonus is wanted *throughout* training — keep exploring — but in supervised learning I want fast convergence on the easy examples and humility only near the end, once the model starts memorizing. A *constant* $-\beta H$ from epoch one is blunt: early it fights the very convergence I want, late it does its job. The richer version of this method anneals $\beta$ or replaces it with a hinge $+\beta\max(0, \Gamma - H)$ that only switches on once entropy drops below a threshold — weak early, strong near convergence. But the scaffold edit I land here is the *plain, fixed-$\beta$, unscheduled* form: $L_{\mathrm{CE}} - \beta H(p)$ with $\beta$ constant for all 300 epochs and no threshold. The harness exposes `epoch` so I *could* anneal, but this baseline does not, and I reason against exactly that. The risk is real: a constant entropy push on the highest-capacity model (VGG-16-BN, 100 classes), still flattening outputs while the schedule decays the learning rate, can prevent the model from ever committing to the correct class, and on a 100-class problem with low baseline accuracy that can tip into a degenerate solution near chance. A fixed $\beta$ has no relief valve, where smoothing's strictly-positive target never could destabilize — so I am trading guaranteed safety for adaptivity, and on the hardest benchmark that trade carries collapse risk.

The knob $\beta$ trades data-fitting against humility and is genuinely task-dependent, but the established default for this penalty as a general regularizer is $\beta = 0.1$, and I keep it fixed across all three benchmarks with no schedule. The implementation is one extra reduction over the logits I already have: a numerically stable `log_softmax`, recover $p$, the per-sample entropy $-\sum_i p_i \log p_i$, average over the batch, and return $L_{\mathrm{CE}} - \beta\bar{H}$ — no extra forward or backward pass, no auxiliary network, `epoch` unused.

The falsifiable expectations against the smoothing numbers: because the pressure is adaptive and concentrated on spiked outputs, I expect the attack AUC to *drop* where smoothing left it flat — on resnet20-cifar10 below $0.7678$ and ERM's $0.7275$ at $\sim 0.79$ accuracy, recovering privacy_score above smoothing's $0.5200$ and plausibly above ERM; on mobilenetv2-fmnist a clean small win over $0.8460$, back toward ERM's $0.8685$. CIFAR-100 is where I am genuinely unsure: either a modest AUC dent below $0.8686$ at preserved accuracy, *or* the failure mode I argued — a training collapse in which the unscheduled push prevents the 100-class model from committing, accuracy craters toward chance, and the composite becomes meaningless. If CIFAR-100 collapses while the two easier benchmarks improve, that is decisive: a fixed, unscheduled mean-control penalty is both insufficiently powerful *and* unsafe on the hardest case, and the next rung must do two new things at once — hold the loss *level* without crushing it to a degenerate point, and deliberately *spread* the member distribution to overlap the non-members instead of only shifting its mean.

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Cross-entropy minus a predictive-entropy bonus (Pereyra et al., 2017)."""

    def __init__(self):
        self.entropy_weight = 0.1

    def compute_loss(self, logits, labels, epoch):
        ce = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        return ce - self.entropy_weight * entropy
```
