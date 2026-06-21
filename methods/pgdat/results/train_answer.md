A convolutional classifier can reach ninety-nine-plus percent on MNIST and still be flipped from a seven to a three by nudging every pixel less than a sensor would ever record. These adversarial inputs are not rare corner cases: one sits next to essentially every image, and the same perturbed image fools other architectures too, which tells me the model never learned the concept in any robust sense. With classifiers moving into systems where this matters — driving, face locks — clean test accuracy is simply not the property I need. The defenses on offer all share one shape of flaw: distilling the network, squeezing the input bit-depth, or bolting on a detector each tunes against one specific attack and certifies nothing about attacks I have not tried. I can never say "no perturbation within this allowed set can flip the label." That, not the absence of a defense, is the real gap. The min-max papers that came before (Huang et al. 2015; Shaham et al. 2018) had even written robustness as a worst-case objective, but they judged the inner worst-case search intractable, fell back to one-step linearization, and then measured robustness only against FGSM — a yardstick weak enough that a reported 70% accuracy at $\varepsilon = 0.7$ is meaningless, since any adversary allowed to move each pixel by more than $0.5$ can gray out any image. What was missing was not the objective but evidence that its inner problem can actually be solved, and a trustworthy way to measure the result.

I propose PGD adversarial training. The starting point is to write robustness as a robust-optimization saddle point. Standard training minimizes $\min_\theta \mathbb{E}_{(x,y)}[L(\theta,x,y)]$, but the expectation is over clean $x$ while the adversary gets to move $x$ afterward, so I bake the adversary into the objective: before paying the loss on each example, let an adversary pick the worst perturbation $\delta$ inside the allowed set $S$, taken as the $\ell_\infty$ ball $S = \{\delta : \|\delta\|_\infty \le \varepsilon\}$ that matches the agreed threat model and the linear story of how max-norm-bounded shifts fool high-dimensional models. The quantity I care about is then

$$\rho(\theta) = \mathbb{E}_{(x,y)\sim D}\Big[\max_{\delta\in S} L(\theta, x+\delta, y)\Big], \qquad S = \{\delta : \|\delta\|_\infty \le \varepsilon\},$$

and I want $\min_\theta \rho(\theta)$. This does more than it looks. Driving $\rho(\theta)$ small means the loss is small for *every* $\delta \in S$ on the typical example, so by construction no admissible perturbation produces high loss and no adversarial example exists within the budget — the guarantee the bolt-on defenses could never give. And the same expression unifies attack and defense: the inner $\max_{\delta\in S} L$ is exactly the attacker's job, the outer $\min_\theta$ exactly the defender's, so they are the two halves of one problem rather than separate research programs. FGSM and FGSM adversarial training fall out as the cheapest possible stab at the inner max plus an outer step on top of it.

The reason nobody committed to this is the inner maximization: a wildly non-concave maximization over the ball, which the prior min-max work assumed was hopeless. So the first question is not "what algorithm" but "is the inner max actually intractable." The natural tool for $\ell_\infty$-constrained maximization is projected gradient ascent. For a step bounded in $\ell_\infty$, the steepest-ascent direction solves $\max_{\|v\|_\infty \le \alpha}(\nabla_\delta L)^\top v$, a linear objective over a box whose maximizer pins each coordinate to its corner, $v = \alpha\,\mathrm{sign}(\nabla_\delta L)$. So the step is $\delta \leftarrow \delta + \alpha\,\mathrm{sign}(\nabla_\delta L)$ — FGSM's move, but taken repeatedly with a small $\alpha$ instead of one jump of size $\varepsilon$. Taking the sign rather than the raw gradient keeps the step in pixel units regardless of gradient magnitude, so $\alpha$ never needs retuning as the gradient scale drifts. After each step the iterate can leave the ball or the valid pixel range, so I project: clip $\delta$ coordinatewise to $[-\varepsilon,\varepsilon]$ (the Euclidean projection onto the $\ell_\infty$ ball) and clip $x+\delta$ into $[0,1]$. In terms of the perturbed image $x_t = x+\delta_t$,

$$x_{t+1} = \mathrm{clip}_{[0,1]}\Big(\mathrm{clip}_{[x-\varepsilon,\,x+\varepsilon]}\big(x_t + \alpha\,\mathrm{sign}(\nabla_x L(\theta, x_t, y))\big)\Big).$$

I start not at $\delta = 0$ but at a random point $\delta_0 \sim \mathrm{Uniform}(-\varepsilon,\varepsilon)$ per coordinate (then clipped into $[0,1]$): starting at $x$ would bias me into the near-linear neighborhood that one-step methods already exploit and that does not represent the whole ball, whereas a random start escapes that neighborhood and, across restarts, samples different basins.

The worry that killed this for everyone else is that PGD climbs a non-concave surface and only finds *local* maxima — so training against "whatever PGD found" could be a moving, unrepresentative target. Rather than assert otherwise, I probe the landscape: run PGD from on the order of $10^5$ random starts and collect the final loss values. From a random start the adversarial loss climbs consistently and plateaus in a handful of steps, and the final values form a tight, well-concentrated distribution with no fat tail of monstrous outliers. The maximizers are genuinely distinct points — pairwise $\ell_2$ distances as large as random points in the ball, angles near ninety degrees, so distinct basins rather than the same hilltop re-found — yet they all reach about the same loss, and even along the segment between two of them the loss only dips by a constant factor. So the wall I assumed turns out not to bite: the local maxima are interchangeable in the only way that matters, their loss. If every first-order trajectory ends at roughly the same height, then "what PGD finds" is a stable, representative estimate. I also notice that some perturbations PGD lands on have *negative* inner product with the gradient at $x$, and the correlation with the clean gradient decays as the perturbation grows — the surface is not the simple linear ramp the one-step picture assumes, which is exactly why one step underperforms and why running PGD to its plateau matters. Because any adversary using only input gradients is pulled into this same concentrated band, and because only first-order attacks scale on deep nets, I treat random-start PGD run to plateau as the universal first-order adversary: beating it should mean beating the whole class of gradient-based attacks.

That settles the inner problem; the outer problem is to descend $\rho(\theta) = \mathbb{E}[\varphi(\theta)]$ with $\varphi(\theta) = \max_{\delta\in S} L(\theta, x+\delta, y)$ by SGD, which needs $\nabla_\theta \varphi$. But $\varphi$ is a max over $\delta$ whose maximizer $\delta^\star$ itself depends on $\theta$, so naively I would differentiate through the entire inner optimization — exactly the implicit-function-theorem, second-derivative machinery I want to avoid. The escape is Danskin's theorem, and the method's validity rests on it. With $g(\theta,\delta)=L(\theta,x+\delta,y)$ and $\varphi(\theta)=\max_{\delta\in S} g(\theta,\delta)$, $S$ compact and $\nabla_\theta g$ continuous, the directional derivative is

$$\varphi'(\theta,h) = \sup_{\delta\in\delta^\star(\theta)} h^\top \nabla_\theta g(\theta,\delta), \qquad \text{and if } \delta^\star(\theta)=\{\delta^\star\}: \ \nabla\varphi(\theta) = \nabla_\theta g(\theta,\delta^\star).$$

The envelope intuition is that at the maximizer the inner gradient with respect to $\delta$ is zero or pinned by the constraint, so shifting $\delta^\star$ is a zeroth-order effect on the value of the max; the $d\delta^\star/d\theta$ term I feared genuinely drops out. To pin the sign, take $\bar\delta$ a maximizer and $h = \nabla_\theta L(\theta, x+\bar\delta, y)$: plugging $+h$ gives $\varphi'(\theta,h) \ge h^\top h = \|\nabla_\theta L(\theta,x+\bar\delta,y)\|^2 \ge 0$, identifying $+h$ as an ascent direction; in the unique-maximizer case Danskin collapses to $\nabla\varphi(\theta)=h$, so $\varphi'(\theta,-h) = -\|h\|^2 < 0$, the descent I need. So freezing the PGD-found perturbation, building $x+\delta^\star$, and backpropagating the ordinary loss $L(\theta, x+\delta^\star, y)$ through $\theta$ as if the input were fixed *is* SGD on the adversarial objective whenever the found point is the active maximizer. Two honest caveats: ReLU and max-pool kinks break differentiability, but on a measure-zero set I never sit on exactly, so the conclusion holds almost everywhere; and PGD returns an approximate local maximizer, not a certified global one, so I apply Danskin on a subregion $S' \subseteq S$ where PGD's point is the active maximum — if PGD found a genuinely high-loss example, stepping $\theta$ to reduce the loss there makes progress against that local adversary.

The remaining knobs follow from reasoning about their purpose. The inner step $\alpha$ must be smaller than the budget $\varepsilon$ with enough steps $k$ that $k\cdot\alpha$ comfortably exceeds $\varepsilon$: that way several small sign steps *traverse* the ball, the projection keeps each one legal, and the boundary is reachable from any interior random start — giving MNIST $\varepsilon=0.3,\ \alpha=0.01,\ k=40$ and CIFAR-10 $\varepsilon=8/255,\ \alpha=2/255,\ k=7$. The random start, beyond escaping the linear neighborhood, prevents the label-leaking pathology of one-step training: a deterministic start yields a narrow, predictable family of perturbed images the network can memorize, whereas random starts deny the outer training any fixed target; and since training runs many epochs with a fresh start each revisit, the restarts come for free across epochs, so one start per example per epoch suffices and no per-batch restarts are needed. Multi-step is worth its cost precisely because the inner max is what prior work got wrong — FGSM linearizes a loss that is not linear out to the $\varepsilon$ of interest, so a non-linear adversary walks right past an FGSM-trained defense. Finally the objective forces capacity to matter: a robust boundary must separate $\ell_\infty$ balls rather than points, a strictly more contorted surface, so a network sized for clean accuracy cannot represent the robust classifier — too small a net trained against PGD collapses toward a constant prediction, and growing capacity lowers the achievable saddle-point value. The method is therefore strongest-adversary *and* enough capacity together. Concretely each step puts the model in eval mode while crafting the attack so batch-norm and dropout do not shift under the perturbation, initializes the adversarial batch at a uniform-random point in the ball clipped to valid pixels, runs $k$ steps of input-gradient sign ascent with projection and clamping, detaching between steps so the graph does not accumulate, then switches to train mode and takes a single cross-entropy SGD step on the adversarial batch.

```python
import torch
import torch.nn.functional as F


class AdversarialTrainer:
    """PGD adversarial training: solve min_theta E[ max_{||delta||_inf <= eps} L ]."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps                    # L_inf perturbation budget
        self.alpha = alpha                # PGD inner step size (alpha < eps)
        self.attack_steps = attack_steps  # number of PGD steps k
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Inner max: same random-start L_inf PGD structure as torchattacks.PGD.forward.
        self.model.eval()
        adv_images = images.clone().detach()

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.eps, self.eps
        )
        adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            cost = F.cross_entropy(outputs, labels)  # maximize true-label CE
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # Outer min: freeze the adversarial batch and take one ordinary SGD step.
        self.model.train()
        outputs = self.model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
