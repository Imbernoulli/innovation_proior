TRADES confirmed its knob helped exactly where there was slack and *hurt* where there was not. On MNIST it climbed PGD-50 from $0.8932$ to $0.9350$ with clean held at $0.9863$, because that problem has room for the boundary regularizer to flatten the score for free. But on CIFAR the robustness-heavy $\beta = 6.0$ did the opposite of help: CIFAR-10 PGD-50 *dropped* from PGD-AT's $0.4585$ to $0.4410$ with clean collapsing $0.8501 \to 0.7894$; VGG-11-BN fell $0.4194 \to 0.4065$ with clean $0.7796 \to 0.7421$; CIFAR-100 PGD-50 barely moved, $0.2214 \to 0.2268$, while clean dropped $0.5828 \to 0.5249$. So TRADES spent a lot of clean accuracy and on three of four scenarios did not even buy robustness back. That is the structural complaint I flagged, not a knob mis-set by a little: TRADES pushes *every* example's boundary equally hard — the ones the model nails and the ones it already blows — and on the harder CIFAR distributions that uniform pressure wastes itself on the easy examples while starving the hard ones. The leverage is mis-allocated.

Before designing anything I want to know *which* examples carry robustness, because the TRADES row says "treat them all the same" is wrong without saying what is right. The definition of an adversarial example is the clue: it is a perturbation of a *correctly classified* input that flips it to wrong. But at any epoch the current model already gets some natural examples wrong with no perturbation, and for those the "adversarial example" PGD crafts is perturbing an already-misclassified input — not even well-defined under the textbook notion. So split the training data by what the current model does on the *natural* image: $S^-$, the examples it misclassifies, and $S^+$, an equal-size chunk it gets right. Three measurements isolate the recipe. First, *not* perturbing $S^-$ during training drops final robustness drastically, while doing the same to $S^+$ barely moves it — the misclassified examples carry most of the robustness, exactly the ones TRADES under-serves. Second, replacing PGD with weak one-step FGSM on $S^-$ leaves robustness essentially unchanged, while weakening the attack on $S^+$ degrades it — so on the misclassified subset *how hard I attack* is nearly irrelevant, which kills any instinct to fix the CIFAR drop with a cleverer inner max. Third, adding a consistency regularizer (perturbed output should match clean output) on $S^-$ improves robustness substantially, while on $S^+$ it helps far less. The lesson is sharp: the leverage is on misclassified examples, the action there is in the *outer loss*, and whatever I build must single those examples out.

I propose **MART** (Misclassification-Aware adveRsarial Training). Write the risk that singles them out, against the 0-1 loss to stay honest. The standard adversarial risk is $R(h) = \frac{1}{n}\sum_i \max_{x'}\mathbb{1}\{h(x') \ne y_i\}$. On $S^+$ I keep it — those citizens the existing recipes handle fine. On $S^-$ my first instinct is to also demand $\mathbb{1}\{h(\hat x_i') \ne y_i\} = 0$, but that asks the perturbed version of an already-wrong example to be *correct*, a harder target than I meet on the easy case; it will thrash, and the diagnostic agrees it was *consistency*, not a harder classification demand, that helped. So I ask the gentler thing — not "be correct after perturbation" but "be *stable* under it" — adding $\mathbb{1}\{h(x_i) \ne h(\hat x_i')\}$, so $R^-(h, x_i) = \mathbb{1}\{h(\hat x_i') \ne y_i\} + \mathbb{1}\{h(x_i) \ne h(\hat x_i')\}$. Now check whether the split needs two formulas. On $S^+$, $h(x_i) = y_i$, so the stability indicator becomes $\mathbb{1}\{y_i \ne h(\hat x_i')\}$ — identical to the first term. Applying $R^-$ everywhere therefore only double-counts the standard risk on $S^+$ and adds the genuinely new stability term on $S^-$, so I need no hard branch — I write one risk where the stability term is *gated on* only for misclassified examples:

$$R_{\mathrm{misc}}(h) = \frac{1}{n}\sum_i \Big\{\mathbb{1}\{h(\hat x_i') \ne y_i\} + \mathbb{1}\{h(x_i) \ne h(\hat x_i')\}\cdot \mathbb{1}\{h(x_i) \ne y_i\}\Big\}.$$

The first term is the ordinary adversarial risk on everyone; the second is a misclassification-aware regularizer gated by $\mathbb{1}\{h(x_i)\ne y_i\}$ so it fires only on examples whose clean version is already wrong. The gate is the whole idea, and it falls out of the algebra rather than being imposed — the formal version of "single out the misclassified examples and regularize *them*," which is the reallocation the TRADES CIFAR rows demanded.

It is a sum of indicators, so each needs a physically meaningful surrogate. For $\mathbb{1}\{h(\hat x') \ne y\}$, classification on the perturbed input, the reflex is plain CE — but this term applies to every example and is the backbone of robustness, and since a robust boundary is more complicated than a clean one I want the loss to push harder for separation. CE maximizes the true-class probability but does not care how large the *best wrong-class* probability is once the true class is on top. So I *boost* CE with a margin term, $\mathrm{BCE}(p(\hat x'), y) = -\log p_y(\hat x') - \log\big(1 - \max_{k\ne y} p_k(\hat x')\big)$: the second piece is small when the runner-up probability is small and blows up as a competitor nears certainty, so minimizing it widens the margin to the nearest competitor — answering "robust classification needs a stronger classifier" with the loss, not just model size. For the stability indicator $\mathbb{1}\{h(x)\ne h(\hat x')\}$ I want a surrogate measuring how *different* two output distributions are: $\mathrm{KL}(p(x)\,\|\,p(\hat x'))$, zero when they match and growing as they diverge — the same local-flatness instinct as TRADES, here aimed only at the hard examples. The gate $\mathbb{1}\{h(x)\ne y\}$ is the delicate one: a hard 0/1 switch inside a differentiable loss freezes the decision and cannot co-evolve as an example's status flips over training. I want a *soft* gate — a continuous quantity large when the clean example is misclassified and small when it is confidently correct — and the model hands me one, $1 - p_y(x)$, the mass on everything except the true class. Confidently correct, $p_y(x)\approx 1$, gate $\approx 0$; misclassified or uncertain, $p_y(x)$ small, gate $\approx 1$. It is smooth, jointly learnable, and *graded* — it leans hardest on the most-wrong examples and tapers as one becomes confident. Assembling, with BCE on all examples and the KL weighted per example by the soft gate, the per-example loss is

$$\mathrm{BCE}(p(\hat x'), y) \;+\; \lambda\,\mathrm{KL}\big(p(x)\,\|\,p(\hat x')\big)\cdot\big(1 - p_y(x)\big),$$

averaged over the batch — one global scalar $\lambda$ balancing classification against the regularizer, since the *per-example* weighting is already the gate's job. The adversarial example $\hat x'$ I still craft with strong CE-PGD, the same attack as PGD-AT, because the diagnostic was unambiguous that attack strength barely matters on the misclassified examples; the differentiation belongs entirely in the outer loss. Against TRADES the contrast is exact: TRADES fits the *clean* label with $\mathrm{CE}(p(x), y)$ and pushes the boundary with a KL whose weight is the same for every example — which is why its CIFAR robustness stalled — whereas MART fits the *perturbed* input with a boosted CE and weights the KL per example by $(1 - p_y(x))$, reallocating the robustness push toward the examples that actually move robustness.

In this edit surface a few places need care, and the harness pins conventions I honor. The inner attack is standard: eval mode, a tiny random nudge start $\mathrm{images} + 0.001\cdot\mathrm{randn}$ (MART's official recipe seeds CE-PGD this way, not from PGD-AT's uniform-ball start), then $\mathrm{attack\_steps}$ of signed-gradient ascent on cross-entropy with per-step projection into the $\epsilon$-box and $[0,1]$. For the margin term I need the largest probability among classes *other than* $y$: argsort the probabilities, take the top two indices, and set $\mathrm{new\_y} = (\text{top-1} == y\ ?\ \text{top-2} : \text{top-1})$ so $p_{\mathrm{new\_y}} = \max_{k\ne y} p_k$ in both cases, writing the margin piece as $\mathrm{nll\_loss}$ against $\mathrm{new\_y}$ on $\log(1.0001 - \mathrm{adv\_probs} + 10^{-12})$, the slack keeping the log finite when a probability rounds to 1. For the KL I take the per-sample sum over classes weighted by $(1 - p_y(x))$, and here the harness pins a convention that differs from the bare write-up: the clean logits are taken *detached* — `logits_clean = self.model(images).detach()` — so the KL target $\mathrm{nat\_probs}$ and the soft-gate weight $\mathrm{true\_probs}$ both come from a *frozen* clean forward pass and the regularizer's gradient flows only through the adversarial branch. (The standalone derivation keeps clean logits attached so the gradient flows through $p(x)$ too; this task deliberately detaches them, stabilizing the KL target and the per-example weighting, and that is the implementation I reproduce.) Multiplying the per-sample KL by $(1.0000001 - \mathrm{true\_probs})$, summing, and dividing by batch size gives $\mathrm{loss\_robust}$; the total is $\mathrm{loss\_adv} + \beta\,\mathrm{loss\_robust}$ with $\beta = 6.0$, the value the edit hardcodes (overriding the headline $\lambda = 5.0$).

```python
class AdversarialTrainer:
    """MART (Wang et al., 2020)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0  # MART regularization weight

    def train_step(self, images, labels, optimizer):
        # Inner max: strong CE-PGD seeded with a small random nudge (official mart.py).
        self.model.eval()
        adv_images = images.detach() + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Outer min: misclassification-aware loss.
        self.model.train()
        optimizer.zero_grad()

        logits_clean = self.model(images).detach()  # detached: frozen KL target + soft-gate weight
        logits_adv = self.model(adv_images)
        adv_probs = F.softmax(logits_adv, dim=1)

        # Boosted CE: standard CE + margin penalty on the best wrong class.
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(
            tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1],
        )
        loss_adv = F.cross_entropy(logits_adv, labels) + F.nll_loss(
            torch.log(1.0001 - adv_probs + 1e-12), new_y,
        )

        # Misclassification-aware KL regularization weighted by (1 - p_y(x)).
        nat_probs = F.softmax(logits_clean, dim=1)
        true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        kl_per_sample = F.kl_div(
            torch.log(adv_probs + 1e-12), nat_probs, reduction='none',
        ).sum(dim=1)
        batch_size = images.size(0)
        loss_robust = (1.0 / batch_size) * torch.sum(
            kl_per_sample * (1.0000001 - true_probs)
        )

        loss = loss_adv + self.beta * loss_robust

        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
