PGD-AT showed exactly what its single CE term could and could not buy. On MNIST it is strong — clean $0.9856$, PGD-50 $0.8932$, a tiny nine-point gap — because that problem has so much slack one term does both jobs. But on the CIFAR rows the single-term cost is written everywhere: PreActResNet-18 on CIFAR-10 at clean $0.8501$ but PGD-50 $0.4585$, a 39-point gap; VGG-11-BN at $0.7796 / 0.4194$; and CIFAR-100 collapsing from clean $0.5828$ to PGD-50 $0.2214$, the lowest robustness on the board. The shape is consistent — robustness lifts far above the floor, clean accuracy is paid down, robust accuracy lands well below clean — and there is nowhere in $\mathbb{E}[\max_{x'} L(x', y)]$ to dial that balance. The failure to fix is not the attack; I solved the inner max, and PGD found genuinely high-loss points. It is the *objective*: one term, no knob. I want a formulation that writes the robustness/accuracy trade-off in as a parameter, and ideally one that explains *why* the trade-off exists so the CIFAR-100 collapse stops being a mystery.

I propose **TRADES** (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization), and the place it starts is not the min-max but the robust error itself. The honest object is the probability that *some* point in the ball is misclassified, $R_{\mathrm{adv}}(f) = \mathbb{E}\,\mathbb{1}\{\exists\, x' \in B(x,\epsilon): f(x') \text{ wrong}\}$, against the natural error $R_{\mathrm{nat}}(f) = \mathbb{E}\,\mathbb{1}\{f(x)\text{ wrong}\}$. A robustly-misclassified point arises in exactly two mutually exclusive ways: either $x$ itself is already misclassified — that is the natural error — or $x$ is classified correctly but sits so close to my own decision surface that a permitted nudge tips it over. Naming the boundary's $\epsilon$-neighborhood gives the boundary error $R_{\mathrm{bdy}}(f) = \mathbb{E}\,\mathbb{1}\{x \text{ near the boundary},\ f(x)\text{ correct}\}$, the mass of *correctly* classified points crowding the surface. These two buckets tile the robustly-wrong set exactly and disjointly, so this is an *equality*, not the loose upper bound PGD-AT had:

$$R_{\mathrm{adv}} = R_{\mathrm{nat}} + R_{\mathrm{bdy}}.$$

That separation is the whole gain: clean accuracy lives entirely in $R_{\mathrm{nat}}$, the extra fragility entirely in $R_{\mathrm{bdy}}$. Give each a differentiable handle and the weight between them *is* the knob the PGD-AT row was missing.

What makes the knob mandatory rather than merely convenient is that the two terms genuinely pull apart. Take $X$ uniform on $[0,1]$ with a label oscillating with period $2\epsilon$. The Bayes classifier tracks the oscillation and has clean error zero, but every point is within $\epsilon$ of a flip, so $R_{\mathrm{bdy}} = 1$ and $R_{\mathrm{adv}} = 1$. The constant all-ones classifier is wrong half the time, $R_{\mathrm{nat}} = 1/2$, but it has *no boundary*, so $R_{\mathrm{bdy}} = 0$ and $R_{\mathrm{adv}} = 1/2$ — strictly more robust than Bayes, while Bayes is strictly more accurate. The accuracy-optimal and robustness-optimal models are different models, far apart, regardless of capacity. So the tension is intrinsic; a single-objective method cannot be right in general, and PGD-AT's CIFAR-100 numbers suffer because it cannot choose where on that curve to sit.

Both terms are 0-1 losses, so I need surrogates. The natural term is the already-solved problem: by calibration theory (Bartlett, Jordan, McAuliffe), a calibrated surrogate margin loss $\phi$ squeezes excess 0-1 risk by excess surrogate risk, so cross-entropy on the clean logits handles $R_{\mathrm{nat}}$. The boundary term is the novelty and has no off-the-shelf surrogate, so I grind on it. First slacken: drop the "correctly classified" qualifier, $R_{\mathrm{bdy}} \le \Pr[x \text{ near boundary}]$, which loses a little but produces a quantity that *no longer references the label* — the boundary term is about the geometry of $f$ near $x$, not about correctness. Now, "$x$ within $\epsilon$ of the boundary" means there is an $x'$ in the ball where the prediction's sign differs, so $\Pr[x\text{ near boundary}] = \mathbb{E}\max_{x'\in B}\mathbb{1}\{f(x)f(x') \le 0\}$ — an indicator of *disagreement between the center and a perturbed point*. Upper-bound that indicator by a margin surrogate: for a nonnegative, non-increasing $\phi$ with $\phi(0)\ge 1$, the pointwise inequality $\phi\!\big(f(x)f(x')/\lambda\big) \ge \mathbb{1}\{f(x)f(x')/\lambda \le 0\}$ gives a differentiable upper bound $R_{\mathrm{bdy}} \le \mathbb{E}\max_{x'\in B}\phi\!\big(f(x)f(x')/\lambda\big)$. The difference from PGD-AT is the entire point: PGD-AT's inner loss is $\phi(f(x')\,y)$, the perturbed point against the *label*; this is $\phi(f(x)f(x'))$, the perturbed prediction against the *clean prediction*. Mine pushes the boundary away from the data by keeping the score's sign across the ball without demanding the perturbed point match the label — which is precisely how it protects clean accuracy where PGD-AT overspent it. Stacking the two pieces gives

$$R_{\mathrm{adv}} - R_{\mathrm{nat}}^* \ \le\ \psi^{-1}\!\big(R_\phi - R_\phi^*\big) \;+\; \mathbb{E}\max_{x'\in B}\phi\!\big(f(x)f(x')/\lambda\big),$$

reading as "be accurate on clean data" plus "do not let any nearby perturbation disagree with your clean prediction." The scale $\lambda$ is the knob — small $\lambda$ punishes disagreement severely (robustness-heavy), large $\lambda$ lets the natural term dominate — and on the toy it recovers Bayes as $\lambda\to\infty$ and the constant classifier as $\lambda\to 0$. The bound is tight (a two-atom construction saturates it), so this is the *right* regularizer, not merely *a* one.

Crossing to deep multi-class nets, three gaps close cleanly. Cross-entropy is the calibrated multi-class surrogate, so the accuracy term is $\mathrm{CE}(f(x), y)$ on clean logits. The boundary surrogate $\phi(f(x)f(x'))$ becomes a discrepancy between the two output distributions, and the natural multi-class form is $\mathrm{KL}(p(x)\,\|\,p(x'))$ — zero exactly when $p(x') = p(x)$, the "no disagreement" baseline, and with the same $x'$-gradient as the cross-entropy form because the entropy of $p(x)$ is constant in $x'$. I drop the explicit $\psi^{-1}$ transform as a heuristic, since its only practical effect is to rescale how hard the accuracy term competes with the boundary term and I already have a scalar for that. Writing $\beta = 1/\lambda$, the objective is

$$\min_f\ \mathbb{E}\Big[\mathrm{CE}(f(x), y) \;+\; \beta \max_{x'\in B}\mathrm{KL}\big(p(x)\,\|\,p(x')\big)\Big].$$

The inner max is the operational heart, and one subtlety is forced rather than chosen. At $x' = x$ the KL is exactly zero — its global minimum — so the gradient vanishes and an attack started at the clean point never moves. PGD-AT started from a uniform random point in the ball; here I must start from a tiny random nudge off the center, $x' = x + 0.001\cdot\mathrm{randn}$, to break the stationarity at the global min. I use the *forward* KL with the clean distribution as the anchor: the inner attack ascends the perturbed prediction's divergence from the clean anchor with signed-gradient PGD over the $L_\infty$ box, the outer minimization drives it back. In this edit surface the clean logits are computed once and *detached* as the KL target, so the attack ascends only $x_{\mathrm{adv}}$ against a frozen clean anchor, and the $\mathrm{batchmean}$ reduction already divides by the batch so no separate $1/\mathrm{batch\_size}$ is needed. I craft the attack in eval mode (frozen BN), then switch to train mode for one SGD step on $\mathrm{CE}(f(x),y) + \beta\,\mathrm{KL}(p(x)\,\|\,p(x'))$. The value $\beta = 6.0$ sits robustness-heavy, which is what the CIFAR rows demand — though that same uniform weight is the next weakness, because it pushes *every* example's boundary equally hard, including the ones the model already nails, with no way to reallocate effort toward the examples that actually move robustness.

```python
class AdversarialTrainer:
    """TRADES (Zhang et al., 2019)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0  # TRADES regularization weight

    def train_step(self, images, labels, optimizer):
        self.model.train()

        # Clean forward pass (the anchor for the KL target).
        logits_clean = self.model(images)
        loss_clean = F.cross_entropy(logits_clean, labels)

        # Inner max: generate adversarial examples by maximizing KL(clean || adv).
        self.model.eval()
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            logits_adv = self.model(adv_images)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean.detach(), dim=1),   # frozen clean anchor
                reduction='batchmean',
            )
            grad = torch.autograd.grad(loss_kl, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Outer min: clean CE + beta * KL(clean || adv).
        self.model.train()
        logits_adv = self.model(adv_images)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_clean.detach(), dim=1),
            reduction='batchmean',
        )
        loss = loss_clean + self.beta * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'loss': loss.item(),
            'loss_clean': loss_clean.item(),
            'loss_kl': loss_kl.item(),
        }
```
