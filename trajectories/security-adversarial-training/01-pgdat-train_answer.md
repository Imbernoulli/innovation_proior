The floor I start from is not a defense at all — it is the scaffold default, plain cross-entropy on clean images. It aces the clean test set and then collapses the instant anyone perturbs the input, because nothing in standard training, $\min_\theta \mathbb{E}[L(\theta, x, y)]$, ever sees a moved $x$. The lineage I inherited has already tried and failed to patch this. Defensive distillation, bit-depth squeezing, and detector front-ends each harden the model against one attack by hiding the gradient, but the gradient is still there and a transfer attack walks past them. FGSM training augments with one-step sign perturbations and does raise FGSM robustness, but it linearizes the loss inside the ball and overfits to its own weak adversary, so an iterative attack bypasses it. The pre-PGD min-max work framed the defense correctly as a saddle point but declared the inner maximization intractable and fell back to the same one-step linearization. So the task at this first rung is not to build yet another defense — defenses are everywhere — it is to state a goal precise enough that hitting it *means* something, and then actually solve it instead of assuming the hard part away.

I propose **PGD adversarial training**: bake the adversary directly into the objective and solve it. For each example, before I pay the loss, let an adversary pick the worst perturbation $\delta$ inside the allowed set $S$ — the $L_\infty$ ball of radius $\epsilon$ (the fixed threat model: $\epsilon = 0.3$ on MNIST, $8/255$ on CIFAR). The quantity I care about is the *adversarial risk*

$$\rho(\theta) = \mathbb{E}\big[\max_{\delta \in S} L(\theta, x+\delta, y)\big], \qquad \min_\theta \rho(\theta).$$

This does more than it looks. If I ever drive $\rho$ small, then the loss is small for *every* $\delta$ in $S$ on the typical example — there is no admissible perturbation left that produces high loss, so by construction there is no adversarial example within the budget. That is the guarantee the bolt-on defenses could never give. And the same expression unifies the two camps: the inner $\max$ is the attacker's job, the outer $\min$ the defender's, and FGSM training is just one outer step on top of a one-step stab at the inner max. Attack and defense are the two halves of one saddle point.

What makes this work where everyone else stopped is that the inner maximization is *not* actually intractable — it was only assumed to be. The right tool for large-scale constrained maximization is projected gradient ascent. Under an $L_\infty$ budget the steepest-ascent direction solves $\max_{\|v\|_\infty \le \alpha} (\nabla_\delta L)^\top v$, a linear objective over a box, so each coordinate pins to its corner: $v = \alpha \cdot \mathrm{sign}(\nabla_\delta L)$. That is FGSM's move taken *repeatedly* with a small step $\alpha$ instead of one jump of size $\epsilon$. The sign also makes $\alpha$ a pixel-unit quantity independent of the gradient's magnitude, so I never retune it as the gradient scale drifts. After each step I project: clip $\delta$ coordinatewise to $[-\epsilon, \epsilon]$ (the Euclidean projection onto an $L_\infty$ ball is per-coordinate), then clip $x+\delta$ into $[0,1]$. That is PGD as the inner adversary.

Where the perturbation *starts* is a load-bearing choice. FGSM starts at $\delta = 0$, right at the clean point, but there the loss is nearly linear in the ball — exactly the regime one-step methods already exploit, and not representative of the whole ball. So I start from a random point, $\delta_0 \sim \mathrm{Uniform}(-\epsilon, \epsilon)$ per coordinate, clipped into $[0,1]$. The random start breaks me out of the linear neighborhood, and across many epochs — a fresh start every time I revisit an example — it effectively samples the ball, which doubles as protection against the label-leaking pathology where the model memorizes one fixed deterministic adversary. The worry this raises is the one that killed the idea for everyone else: PGD climbs a non-concave surface and finds only local maxima, so training against "whatever PGD found" might mean training against a moving, unrepresentative target. But probed with many random restarts the landscape does not behave that way — the trajectories climb and plateau quickly and the final loss values concentrate tightly across distinct basins, so what PGD finds is a stable, representative estimate of the inner max. Random-start PGD is a reliable stand-in for the strongest first-order adversary, which is the only class that scales on these nets.

That settles the inner problem; the outer problem is where Danskin's theorem earns its place. To take an SGD step on $\rho(\theta) = \mathbb{E}[\phi(\theta)]$ with $\phi(\theta) = \max_\delta L(\theta, x+\delta, y)$, I need $\nabla_\theta \phi$, but $\phi$ is a max over $\delta$ and the maximizer $\delta^*$ itself depends on $\theta$. Naively I would differentiate through the whole inner optimization. Danskin says I do not have to: at the active maximizer, $\nabla \phi(\theta) = \nabla_\theta L(\theta, x+\delta^*, y)$ — the $d\delta^*/d\theta$ term drops out by the envelope argument, because at the maximum the gradient with respect to $\delta$ is zero or pinned by the constraint, so moving $\delta^*$ is a zeroth-order effect on the value of the max. So I freeze the perturbation PGD found, build $x+\delta^*$, and backprop the ordinary cross-entropy as if $x+\delta^*$ were a fixed input. Two honest caveats: the loss has ReLU/max-pool kinks, but those are measure-zero and I never sit exactly on one; and PGD returns a local maximizer, so the conclusion is local — but if PGD found a genuinely high-loss point, the step that reduces the loss there makes real progress against the adversary I am actually defending against.

The knobs follow from what each is for. The inner step $\alpha$ must be smaller than the budget $\epsilon$ with enough steps that $k\,\alpha$ comfortably exceeds $\epsilon$: if each step were $\epsilon$ I would leave the ball immediately and bounce on its surface; I want several smaller steps that traverse the ball with the projection keeping me legal and the boundary reachable from any interior start. The harness passes $\alpha$ and $\mathrm{attack\_steps}$ per scenario, so I do not hardcode them. I craft the attack in eval mode so batch-norm statistics and dropout do not shift under me as I perturb — I want the gradient of the deployed function — then switch to train mode for one forward on the adversarial batch, cross-entropy against the *true* labels (the untargeted attack maximizes true-label loss, which is what drives a misclassification), and a single SGD step. This is the genuine floor of adversarial methods: it should jump far above the scaffold's near-zero robust accuracy on every scenario. But its one structural weakness already points at the next rung — the objective is a *single* term, the loss at the worst point against the true label, with no place to dial the balance between clean accuracy and robustness, so clean accuracy just falls out as a byproduct, worst on the many-class CIFAR-100 boundary where one CE term has the least room.

```python
class AdversarialTrainer:
    """PGD Adversarial Training (Madry et al., 2018)."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Inner max: random-start L_inf PGD (eval mode so BN/dropout are frozen).
        self.model.eval()
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.attack_steps):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)               # untargeted: maximize true-label CE
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)  # project into eps-box
            adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()            # project into [0,1]

        # Outer min: one ordinary CE step on the adversarial batch (valid by Danskin).
        self.model.train()
        outputs = self.model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
```
