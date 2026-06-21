I train image classifiers to be robust through the saddle-point recipe: for each example I find the worst perturbation inside the $L_\infty$ ball of radius $\epsilon$ with projected gradient descent, then minimize cross-entropy on that perturbed input. It is the one procedure that produces genuinely robust networks rather than ones that only look robust until someone attacks them properly. And yet on CIFAR-10 — a dataset I can drive to near-perfect clean accuracy without trying — the robust accuracy under a strong white-box PGD attack sits far below the clean accuracy, and the gap refuses to close. The question is therefore not whether adversarial training works but what the standard recipe leaves on the table, and rather than reach for more data or a bigger network I want to stare at the formulation itself. The thing that bothers me is definitional: an adversarial example, by its own textbook meaning, is a perturbation of a *correctly classified* example that flips it to wrong. But at any epoch during training the current model already gets some natural training examples wrong with no perturbation at all, and for those the object PGD crafts is perturbing an input that was already misclassified — not even well-defined under the textbook notion. The standard loss does not care: it perturbs every example and applies the same cross-entropy to every perturbed input, treating "the example the model nails" and "the example the model already blows" as identical citizens of the training set. Nobody had asked whether that distinction matters.

So before designing anything I probe it. Take a model trained the standard way to roughly $87\%$ clean training accuracy and partition the natural training set by the model's current clean prediction into $S^- = \{i : h(x_i)\neq y_i\}$, the examples it misclassifies, and an equal-size correctly-classified subset $S^+ = \{i : h(x_i)=y_i\}$, then manipulate one half at a time. Three measurements come back, and they are pointed. First, if I leave $S^-$ *unperturbed* during training while still perturbing everything else, final robustness drops drastically, whereas doing the same to $S^+$ barely moves it — the misclassified examples carry most of the robustness. Second, if I replace the strong PGD attack with a weak one-step FGSM on $S^-$ only, final robustness is essentially unchanged, while weakening the attack on $S^+$ degrades it — on the misclassified subset, *how hard I attack* is nearly irrelevant. Third, if I add a consistency regularizer that asks the perturbed output to match the clean output to the loss on $S^-$ only, robustness improves substantially, while the same change on $S^+$ helps far less. The lesson is that the leverage is on the misclassified examples, and on those the inner maximization is a near-no-op while the outer minimization is where the action is. That kills the instinct to invent a cleverer attack; whatever I build must live in the outer loss and must single out misclassified examples and treat them differently. The standard recipe's sin is precisely that it does not differentiate, and the prior options each repeat it: standard PGD adversarial training applies one cross-entropy to all perturbed inputs identically; ALP/CLP pull the perturbed output toward the clean output with an $L_2$ penalty under a single global weight applied uniformly; TRADES decomposes robust error into a natural plus a boundary term, $\mathrm{CE}(p(x),y) + \tfrac{1}{\lambda}\max_{x'} \mathrm{KL}(p(x)\,\|\,p(x'))$, but its KL regularizer carries the *same weight for every example*; and MMA does use the misclassification distinction but as a *hard*, hand-thresholded split that is never itself optimized and cannot co-evolve with the network. None of them reweights the outer loss toward the examples the diagnostic shows actually move robustness.

I propose MART — Misclassification Aware adveRsarial Training. I work first against the 0-1 loss to stay honest about what I want before surrogates blur it. The standard adversarial risk is $R(h) = \tfrac{1}{n}\sum_i \max_{x'\in B_\epsilon(x_i)} \mathbb{1}(h(x')\neq y_i)$, and I split it by the clean prediction. For the correctly-classified half I have no reason to change anything, so $S^+$ keeps the standard adversarial risk $\mathbb{1}(h(\hat x'_i)\neq y_i)$, where $\hat x'_i$ is the adversarial example. For the misclassified half my first instinct — also demand $\mathbb{1}(h(\hat x'_i)\neq y_i)=0$, i.e. that the perturbed version be classified correctly — is the wrong ask: the clean version is already wrong, so demanding its perturbed version be right is a harder target than I have met even on the easy examples, and pushing hard there will just thrash. The diagnostic agreed from the other side that it was *consistency*, not a harder classification demand, that helped on $S^-$. So on the misclassified half I ask for something gentler and more honest — not "be correct after perturbation" but "be *stable* under perturbation," that the perturbed prediction not differ from the clean prediction — written as an extra indicator $\mathbb{1}(h(x_i)\neq h(\hat x'_i))$. The misclassified per-example risk is then $\mathbb{1}(h(\hat x'_i)\neq y_i) + \mathbb{1}(h(x_i)\neq h(\hat x'_i))$. The load-bearing step is checking whether the case split is even necessary: on $S^+$, $h(x_i)=y_i$ by definition, so the stability indicator $\mathbb{1}(h(x_i)\neq h(\hat x'_i))$ becomes $\mathbb{1}(y_i\neq h(\hat x'_i))$, which is *identical* to the standard risk term. The stability term and the adversarial risk are the same event on correctly-classified examples, so applying the stability term everywhere merely double-counts harmlessly on $S^+$ and adds the genuinely new term on $S^-$. I therefore do not need two formulas with a hard branch; I write a single risk where the stability term is gated by the misclassification indicator,

$$R_{\mathrm{misc}}(h) = \frac{1}{n}\sum_i \Big\{\, \mathbb{1}(h(\hat x'_i)\neq y_i) \;+\; \mathbb{1}(h(x_i)\neq h(\hat x'_i))\cdot\mathbb{1}(h(x_i)\neq y_i)\,\Big\},\qquad \hat x'_i = \arg\max_{x'\in B_\epsilon(x_i)}\mathbb{1}(h(x')\neq y_i).$$

The first term is the ordinary adversarial risk applied to everyone; the second is a misclassification-aware regularizer whose gate $\mathbb{1}(h(x_i)\neq y_i)$ fires only when the clean version is already wrong. The gate is the whole idea, and crucially it falls out of the algebra rather than being imposed by hand — it is exactly what makes the two-subset definition collapse into one expression.

This target is a sum of indicators — discontinuous and intractable — so I replace each of the three indicators with a surrogate that is meaningful for what that indicator was doing, not merely smooth. For the classification term $\mathbb{1}(h(\hat x')\neq y)$, which applies to every example and is the backbone of robustness, the reflex is plain cross-entropy $-\log p_y(\hat x')$. But a robust decision boundary is genuinely more complex than a clean one, robust classification needs a *stronger, wider-margin* classifier, and cross-entropy maximizes the true-class probability without directly caring how large the *best wrong-class* probability is once the true class is on top. So I boost cross-entropy with a margin term that explicitly drives the runner-up down:

$$\mathrm{BCE}(p(\hat x'), y) = -\log p_y(\hat x') \;-\; \log\!\big(1 - \max_{k\neq y} p_k(\hat x')\big).$$

The first term fits the true class; the second is small when the best competing probability is small and blows up as that competitor approaches certainty, so minimizing it shrinks the runner-up and widens the margin — answering the "robust classification needs a stronger classifier" pressure with the loss rather than only with model size. For the stability term $\mathbb{1}(h(x)\neq h(\hat x'))$ I need a surrogate that measures how *different* the two output distributions are, so I take the KL divergence $\mathrm{KL}(p(x)\,\|\,p(\hat x')) = \sum_k p_k(x)\log\!\big(p_k(x)/p_k(\hat x')\big)$, which is zero exactly when the distributions match and grows as they diverge; minimizing it makes the network locally flat around the input, the smoothness I wanted for the hard examples. The delicate one is the gate $\mathbb{1}(h(x)\neq y)$, the switch that turns the regularizer on. A hard 0/1 switch is the worst possible thing to put inside a loss I want to differentiate and learn jointly: it freezes the in-or-out-of-$S^-$ decision, cannot be backpropagated through, and cannot co-evolve as an example's status flips during training — which is precisely MMA's failing. I want a *soft* gate, a continuous quantity that is large when the clean example is misclassified and small when it is confidently correct, and the model already hands me one: $1 - p_y(x)$, the probability mass on everything except the true class. When the example is confidently correct $p_y(x)\approx 1$ so the gate is near zero and the regularizer is almost off; when the example is misclassified or merely uncertain $p_y(x)$ is small and the gate is near one. It is differentiable and jointly learnable, and it does something nicer than a hard switch — it gives a *graded* weight that leans hardest on the most-wrong examples and tapers off as an example becomes confident, the continuous version of the diagnostic's "spend effort on the hard ones."

Assembling the surrogate objective: BCE applies to all examples; the KL term is multiplied by the soft gate $(1 - p_y(x))$ so the regularizer is up-weighted on misclassified examples and down-weighted on confident ones; and a single global scalar $\lambda$ balances classification against regularization — one knob for the natural-vs-robust trade-off, fixed across examples because the *per-example* weighting is already carried by $(1 - p_y(x))$. The per-example loss and objective are

$$\ell(x,y,\theta) = \mathrm{BCE}(p(\hat x'), y) \;+\; \lambda\,\mathrm{KL}(p(x)\,\|\,p(\hat x'))\cdot\big(1 - p_y(x)\big),\qquad L_{\mathrm{MART}}(\theta) = \frac{1}{n}\sum_i \ell(x_i,y_i,\theta),$$

with $\hat x'$ still generated by the same strong CE-PGD attack as standard training and $\lambda = 6$, a robustness weight in the same range TRADES uses. I deliberately do not get clever in the inner maximization, because the measurements were unambiguous that attack strength on the misclassified examples barely matters; all the new structure belongs in the outer loss. The contrast with TRADES makes the design explicit: TRADES fits the *clean* label with $\mathrm{CE}(p(x),y)$ and pushes the boundary with a KL term weighted *uniformly* across examples; MART fits the *perturbed* input with boosted CE and weights the KL *per example* by $(1 - p_y(x))$, so an already-misclassified example gets a much heavier robustness push than a confidently-correct one. The uniform-weight version pushes equally hard on the example it already nails and the one it already blows; the diagnostic says that is the wrong allocation, and the $(1 - p_y)$ factor is the direct implementation of the single empirical finding that motivated the whole exercise.

Two places need care where the clean math meets a softmax. For the margin term I need, per example, the largest probability among classes *other than* $y$. Argsort the probabilities and take the top two indices: the top-1 is the model's predicted class, so if the top-1 equals $y$ the competitor I want is the top-2, and if the top-1 is not $y$ then the top-1 itself is already the best wrong class. Defining $\texttt{new\_y} = (\text{top-1}==y\,?\,\text{top-2}:\text{top-1})$ gives $p_{\texttt{new\_y}} = \max_{k\neq y}p_k$ in both cases, and I write $-\log(1 - p_{\texttt{new\_y}})$ as an `nll_loss` against $\texttt{new\_y}$ on $\log(1 - \texttt{adv\_probs})$, with a hair of slack ($1.0001$ plus a tiny epsilon) so a probability rounding to $1$ does not send the log to $-\infty$. For the KL term, the framework's primitive expects the *log* of the second distribution and the first distribution as target and returns the elementwise summand, so I feed it $\log(\texttt{adv\_probs})$ and $\texttt{nat\_probs}$ and sum over the class dimension; the soft weight is one minus the natural probability gathered at the true label. The natural logits stay attached to the graph because $p(x)$ appears in both the KL target and the $(1 - p_y)$ weight, and I want the regularizer's gradient to flow through the clean forward pass too. The total loss is the boosted-CE term plus $\lambda$ times the weighted KL.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTrainer:
    """MART: Misclassification Aware adveRsarial Training."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes
        self.beta = 6.0                     # lambda: KL-regularizer weight
        self.kl = nn.KLDivLoss(reduction='none')

    def train_step(self, images, labels, optimizer):
        # inner maximization: strong L_inf CE-PGD with a small random nudge
        self.model.eval()
        adv = images.detach() + 0.001 * torch.randn_like(images)
        adv = torch.clamp(adv, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv.requires_grad_(True)
            loss_ce = F.cross_entropy(self.model(adv), labels)
            grad = torch.autograd.grad(loss_ce, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, 0.0, 1.0).detach()

        # outer minimization: MART loss
        self.model.train()
        optimizer.zero_grad()
        batch_size = images.size(0)

        logits = self.model(images)
        logits_adv = self.model(adv)
        adv_probs = F.softmax(logits_adv, dim=1)

        # boosted CE = CE(adv, y) - log(1 - max_{k!=y} p_k(adv))
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == labels, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, labels) \
            + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        # misclassification-aware KL regularizer, weighted by (1 - p_y(x))
        nat_probs = F.softmax(logits, dim=1)
        true_probs = nat_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        kl_per_sample = torch.sum(
            self.kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1)
        loss_robust = (1.0 / batch_size) * torch.sum(
            kl_per_sample * (1.0000001 - true_probs))

        loss = loss_adv + self.beta * loss_robust
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}
```
