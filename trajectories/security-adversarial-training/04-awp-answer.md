**Problem (from step 3).** MART recovered the CIFAR robustness TRADES lost (C10 PGD-50 0.4891, VGG
0.4746, C100 0.2713) but MNIST slipped (0.935→0.9055), and three loss redesigns hit the same CIFAR
ceiling. The unaddressed failure is robust *overfitting*: a ~40-point train/test robust-accuracy gap
that widens late in training, which no choice of per-example loss weight touches.

**Key idea.** Adversarial training flattens the *input* loss landscape but not the *weight* loss
landscape, and (measured correctly, with on-the-fly attacks + filter-normalized directions) flatter
weight landscapes track smaller robust gaps — every successful method only flattens it implicitly. So
flatten it directly: `min_w {rho(w) + (rho(w+v) - rho(w))}` collapses to `min_w rho(w+v)`, and the
worst-case `v` (PAC-Bayes upper bound on expected sharpness) is the efficient probe:
`min_w max_{||v_l|| <= gamma||w_l||} rho(w+v)`, on top of the TRADES base loss.

**Why it works.** A flat weight minimum lets training robustness translate into *test* robustness
instead of diverging late. The perturbation is sized per layer relative to `||w_l||` (ReLU scale
invariance), found by one normalized-gradient ascent step (L2 budget → normalized gradient, not sign),
and applied only as a temporary probe — descend at `w+v`, then restore the center.

**Scaffold edit.** A **proxy** `copy.deepcopy(model)` with its own SGD carries the throwaway BN-stat
damage. `_calc_awp` loads model weights into the proxy, computes the TRADES loss, *negates* it, and
takes one proxy SGD step (negated loss = ascent). `_diff_in_weights` renormalizes the raw step to
`||w_l||` per multi-dim `'weight'` tensor (`1e-20` floor); `perturb`/`restore` add `±gamma*diff`. The
attack maximizes `KL(clean||adv)` with `reduction='sum'` and the `0.001*randn` nudge start; the base
loss uses `batchmean` KL. Four steps per minibatch: craft `x'`, AWP `diff` on proxy, perturb, TRADES
loss + step, restore.

**Hyperparameters.** `gamma = 0.005` (band `[1e-3, 5e-3]`); `beta = 6.0` (TRADES base); proxy SGD
`lr = 0.1` (magnitude set by `gamma`+renorm, not proxy lr); `eps`, `alpha`, `attack_steps` per scenario.

**What to watch.** PGD-50 should rise above MART where the generalization gap is largest — CIFAR-10
ResNet above 0.4891, CIFAR-100 above 0.2713 — and MNIST should recover toward 0.935. VGG (MART's
board-best 0.4746) may only match, given BN/proxy interaction. This is the endpoint: it must beat
MART's row by flattening the weight landscape where three loss redesigns could not.

```python
class AdversarialTrainer:
    """AWP + TRADES (Wu et al., 2020 + Zhang et al., 2019)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        import copy
        from collections import OrderedDict
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0        # TRADES regularization weight
        self.gamma = 0.005     # AWP perturbation magnitude (paper default)
        self._EPS_AWP = 1e-20
        # Proxy model + optimizer so BN running stats of the main model are untouched.
        self.proxy = copy.deepcopy(model)
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=0.1)

    def _diff_in_weights(self):
        from collections import OrderedDict
        diff = OrderedDict()
        model_sd = self.model.state_dict()
        proxy_sd = self.proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_sd.items(), proxy_sd.items()):
            if old_w.dim() <= 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w - old_w
                diff[old_k] = old_w.norm() / (diff_w.norm() + self._EPS_AWP) * diff_w
        return diff

    def _add_into_weights(self, diff, coeff):
        with torch.no_grad():
            names = diff.keys()
            for name, param in self.model.named_parameters():
                if name in names:
                    param.add_(coeff * diff[name])

    def _calc_awp(self, adv_images, clean_images, labels):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss_natural = F.cross_entropy(self.proxy(clean_images), labels)
        loss_robust = F.kl_div(
            F.log_softmax(self.proxy(adv_images), dim=1),
            F.softmax(self.proxy(clean_images), dim=1),
            reduction='batchmean',
        )
        loss = -1.0 * (loss_natural + self.beta * loss_robust)  # negate => ascent
        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        return self._diff_in_weights()

    def train_step(self, images, labels, optimizer):
        # Step 1: TRADES-style attack (maximize KL(clean || adv), sum reduction).
        self.model.eval()
        adv_images = images.detach() + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            loss_kl = F.kl_div(
                F.log_softmax(self.model(adv_images), dim=1),
                F.softmax(self.model(images), dim=1),
                reduction='sum',
            )
            grad = torch.autograd.grad(loss_kl, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign().detach()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        self.model.train()

        # Step 2: AWP — worst-case weight perturbation via proxy, apply to model.
        diff = self._calc_awp(adv_images, images, labels)
        self._add_into_weights(diff, coeff=1.0 * self.gamma)

        # Step 3: TRADES loss under perturbed weights.
        optimizer.zero_grad()
        logits_adv = self.model(adv_images)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(self.model(images), dim=1),
            reduction='batchmean',
        )
        logits_clean = self.model(images)
        loss_clean = F.cross_entropy(logits_clean, labels)
        loss = loss_clean + self.beta * loss_robust

        # Step 4: backward, step, then RESTORE the center weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self._add_into_weights(diff, coeff=-1.0 * self.gamma)

        return {'loss': loss.item()}
```
