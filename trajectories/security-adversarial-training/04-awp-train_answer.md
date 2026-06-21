MART did what I predicted on the rows that mattered, and the way it succeeded reveals what is left undone. The per-example reallocation recovered exactly the CIFAR scenarios TRADES had lost: CIFAR-10 PGD-50 climbed to $0.4891$, past PGD-AT's $0.4585$ and well past TRADES's $0.4410$; VGG-11-BN reached $0.4746$, the board-best, up from $0.4194 / 0.4065$; CIFAR-100 hit $0.2713$, clearly above TRADES's flat $0.2268$; and the margin-boosted CE bought back clean accuracy, CIFAR-10 clean recovering to $0.8342$ from $0.7894$. But two things say the loss-design lever is nearly exhausted. MNIST PGD-50 actually *fell*, $0.9350 \to 0.9055$ — the boosted CE and per-example gate are tuned for the hard CIFAR distributions and slightly overfit the easy one, so I am now trading between scenarios rather than lifting all of them. And decisively: every method so far — PGD-AT, TRADES, MART — is a different choice of *what loss to put on which example*, yet headline robust accuracy is still stuck in the mid-$0.4$s on CIFAR-10 and high-$0.2$s on CIFAR-100. Three loss redesigns, and the ceiling barely moved. The remaining gap is not about the loss at all.

Look past the per-scenario headline at what every run shares. A PGD-AT model on CIFAR-10 classifies about 84% of the *training* set correctly under PGD but only about 43% of the *test* set under the same attack — a 41-point robust generalization gap where a normally-trained net keeps its train/test gap under ten. And it worsens over the curve: test robustness peaks early, around the first learning-rate decay, then *decays* while training robustness keeps climbing. That is overfitting of the *worst-case* loss — robust overfitting. People recover most of the apparent gains of fancier losses just by early-stopping at that peak, which is a confession, not a cure: it closes the gap only by quitting before the model has actually become robust. MART, TRADES, PGD-AT all sit inside this gap; none addresses it, because reweighting which example gets which loss does nothing about the late-training divergence between train and test robustness. That is the failure that survived all three rungs.

What does adversarial training do geometrically? It flattens the loss as a function of the *input* — around each training point the loss barely moves inside the ball, by construction. But flattening in input space says nothing about how the loss behaves in *weight* space, and the gap I am chasing is a property of the weights I land on, not of any single input. The ordinary flat-minima story applies: if the loss as a function of the *weights* is flat around the solution — you can jiggle the weights and the loss barely rises — that solution generalizes better than one in a sharp, narrow valley. Measured correctly the link survives. The naive measurement is wrong: probing the surface with a fixed adversarial set generated once on the unperturbed model makes every perturbed model look artificially robust (an example crafted for $f_w$ is a weak attack on $f_{w+v}$), so the landscape looks artificially flat. The honest measurement regenerates the PGD attack on-the-fly for each perturbed weight and uses filter-normalized directions (scale-fair under ReLU scale invariance). Done that way the evidence is precise: before the best epoch the weight landscape is flat and the robust gap small; after it, as test robustness falls, the landscape sharpens in lockstep with the gap — and across PGD-AT, the KL-regularized variant, the misclassification-aware variant, and early-stopped AT, the smaller-gap methods all have flatter weight landscapes. Every method on this ladder, MART included, has been flattening this surface only *implicitly*, by a different indirect route — which is why three loss redesigns hit the same ceiling.

I propose **AWP** (Adversarial Weight Perturbation), on a TRADES base — stop being coy and flatten the weight loss landscape *directly*. Write what "flat at $w$" means: how much the adversarial loss $\rho$ rises if I move the weights to $w+v$, i.e. $\rho(w+v) - \rho(w)$. I want the loss low *and* that bump small, so $\min_w\{\rho(w) + (\rho(w+v) - \rho(w))\}$. The $\rho(w)$ and $-\rho(w)$ cancel and the whole thing collapses to $\min_w \rho(w+v)$: minimizing "loss plus flatness penalty" is just minimizing the loss *at the perturbed point*, because the only way to keep the loss small at both $w$ and $w+v$ is for the surface between them to be flat — no separate regularizer needed. Which $v$? A random $v$ mostly points along directions the loss does not care about, so I use the *worst* direction, found cheaply at tiny magnitude:

$$\min_w\ \max_{\|v_l\| \le \gamma\|w_l\|}\ \rho(w+v).$$

A double perturbation — inputs perturbed adversarially (inner) and weights perturbed adversarially (outer over $v$) — and the PAC-Bayes flatness bound justifies the worst case as a conservative upper bound on the expected sharpness, since an expectation over $V$ cannot exceed the max on $V$, so driving down $\max_v \rho(w+v) - \rho(w)$ squeezes the expected-sharpness term that controls the gap. The bound also says how to *size* $v$: per layer relative to $\|w_l\|$, because ReLU scale invariance makes absolute weight magnitudes meaningless — the same filter-normalization lesson, with a single dimensionless $\gamma$ as the weight-space analogue of $\epsilon$. The outer $\max_v$ is solved by projected gradient *ascent*; the budget here is an $L_2$/Frobenius ball per layer, not an $L_\infty$ box, so the steepest-ascent direction is the *normalized* gradient, not the sign, scaled to the radius. I take the cheapest version that implements the math — one alternation, so $v = 0$ when I craft $x'$ (generated on the plain model), then one ascent step for $v$ on those $x'$.

What makes this real are two traps the edit surface pins. First, batch-norm: computing $\nabla_v$ needs forward passes through $f_{w+v}$, throwaway passes on a deliberately corrupted model that I do *not* want polluting the real model's BN running stats. The clean fix is a *proxy* — `copy.deepcopy` of the model with its own SGD — load current weights in, do the ascent there, read off the perturbation, apply it to the real model only for the actual training pass, restore afterward. The proxy carries the BN damage; the real model stays clean. The proxy lr matches the main lr, though the magnitude is governed by $\gamma$ and the renormalization, not the proxy lr. Second, ascent-as-negated-descent: an SGD optimizer minimizes, so to make the proxy *ascend* I feed it the negated loss, $\mathrm{loss} = -1.0\cdot(\mathrm{loss\_natural} + \beta\cdot\mathrm{loss\_robust})$, and take one step, after which $\Delta w = w_{\mathrm{proxy}} - w$ points along $+\nabla\rho$. The renormalization implements the per-layer relative-size constraint exactly: for each multi-dim weight tensor, normalize the raw $\Delta w$ to unit norm and multiply by $\|w_l\|$, so $\mathrm{diff}_l = \big(\|w_l\| / (\|\Delta w_l\| + 10^{-20})\big)\,\Delta w_l$, with the tiny floor against a zero step; the applied perturbation $v_l = \gamma\,\mathrm{diff}_l$ then sits right on the boundary of the $\gamma\|w_l\|$ ball, so the projection is satisfied by construction. Only tensors with $\dim > 1$ whose name contains `'weight'` — conv and linear weights, where "relative to $\|w_l\|$" is meaningful — are perturbed, skipping BN scales and biases. The outer descent has its own trap: the perturbation is a temporary probe, so I must not leave the weights at $w+v$. After computing $\mathrm{diff}$ I perturb (add $+\gamma\,\mathrm{diff}$), compute the base loss under the perturbed weights, backward, $\mathrm{optimizer.step()}$, then restore (add $-\gamma\,\mathrm{diff}$) to bring the center back. The net effect is that the SGD direction is the gradient *at the perturbed point* applied to the center weights — a flat-minimum-seeking step without drifting toward the perturbation.

The base loss is the strongest I have built, TRADES: $\mathrm{CE}(\mathrm{clean}) + \beta\,\mathrm{KL}(\mathrm{clean}\,\|\,\mathrm{adv})$ with $\beta = 6.0$, the adversarial example crafted by maximizing the KL between clean and perturbed predictions. I combine with TRADES rather than MART deliberately: AWP is generic over the inner loss and TRADES is the cleanest, strongest base to flatten, while MART's per-example reweighting is orthogonal to the axis I am now attacking — the weight landscape, not the example allocation. In this edit surface the attack uses `F.kl_div(F.log_softmax(model(adv)), F.softmax(model(clean)), reduction='sum')` with the $0.001\cdot\mathrm{randn}$ nudge start (the KL's global-minimum-at-$x$ problem still forces it), and both the proxy ascent and the final descent use the $\mathrm{batchmean}$ TRADES loss with the same argument order. The magnitude $\gamma = 0.005$ lives in the useful band $[10^{-3}, 5\times 10^{-3}]$ — too small flattens nothing, too large makes $w+v$ a poor training point. Because AWP attacks robust *overfitting* rather than the loss allocation, a flatter weight minimum lets training robustness translate into test robustness instead of diverging late, which is the lift the three loss redesigns could not produce.

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
