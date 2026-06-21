# Context: defending classifiers against membership inference (circa 2021)

## Research question

A classifier `f(.;theta)` trained on a private dataset leaks information about *which*
samples it was trained on. A membership inference attacker, given a query sample
`z = (x, y)` and access to the model, must decide whether `z` was in the training set.
The threat is pervasive — it works on images, medical records, and transaction histories,
and persists even under black-box access (query in, posterior out) or partial outputs
(top-k labels). The precise goal: design a *training-time* procedure that drives a strong
attacker's success rate down toward chance (an attack AUC of 0.5) without sacrificing the
model's test accuracy.

## Background

By this time the connection between privacy leakage and overfitting is well established, and
two results are load-bearing for everything that follows.

**Membership advantage equals the generalization gap (Yeom, Giacomelli, Fredrikson & Jha,
CSF 2018).** They formalize the membership experiment: sample a training set `S ~ D^n`, flip
a coin `b`, draw the challenge `z` from `S` if `b=0` and from `D` if `b=1`, and ask the
adversary `A` to guess `b`. The *membership advantage* is
`Adv = Pr[A=0 | b=0] - Pr[A=0 | b=1]` (true-positive minus false-positive rate). They then
exhibit a concrete attack — the **bounded-loss adversary** — that assumes only `ell(.) <= B`
and, on query `z`, outputs "non-member" with probability `ell(A_S, z) / B`. Its advantage is
exactly
```
Adv = R_gen(A) / B,   R_gen(A) = E_{z~D}[ell(A_S, z)] - E_{z~S}[ell(A_S, z)],
```
the average generalization error — expected loss on fresh data minus expected loss on the
training set. With the 0-1 loss (`B=1`) the membership advantage *is* the generalization
gap. They also note that when the loss is approximately Gaussian, an attacker who knows the
error distribution can simply threshold the loss, and its advantage grows as the ratio of
non-member to member loss standard deviations grows.

**The Bayes-optimal attack depends only on the sample loss (Sablayrolles, Douze, Ollivier,
Schmid & Jegou, ICML 2019).** Under the modeling assumption that the trained parameters
follow a posterior
```
P(theta | z_1, ..., z_n) ∝ exp( -(1/T) sum_i m_i * ell(theta, z_i) ),
```
(with `m_i` the membership bits and `T` a temperature controlling stochasticity — `T=1` is
the Bayesian posterior, `T→0` is MAP, small `T` models averaged SGD), they derive the
optimal membership inference in closed form. Singling out `z_1`, the optimal posterior
probability of membership reduces to a sigmoid of a *score* that, after the algebra,
contains `theta` only through the scalar `ell(theta, z_1)`:
```
s(z_1, theta, p) = (1/T) ( tau_p(z_1) - ell(theta, z_1) ),
tau_p(z_1) = -T log ∫ exp( -(1/T) ell(t, z_1) ) p_T(t) dt,
```
where `tau_p(z_1)` is the typical loss on `z_1` over models that did *not* train on it. The
attacker calls a point a member when `-ell(theta, z_1)` exceeds a threshold, equivalently
when its loss falls below a threshold. The white-box internals of `theta` add nothing beyond
this loss value; the optimal attack is a loss threshold (a global constant threshold, or a
per-sample one). So the single quantity an optimal attacker reads off the model is the
*per-sample loss*, and the only thing that separates members from non-members is how the
loss is *distributed* across the two groups.

**The distributional picture.** Put the two together. Let `P` be the distribution of the
per-sample loss over members and `Q` over non-members. Standard training pushes member
losses toward 0, so `P` piles up near zero with small spread, while `Q` sits at a higher
mean with larger spread — two well-separated humps. An optimal attacker thresholds the loss
to tell them apart, and how well it can do so is governed by the statistical distance
between `P` and `Q`. Empirically, on CIFAR-10 with a 20-layer ResNet, vanilla training
yields a loss-thresholding attack AUC around 0.84: members near zero loss, non-members far
above. The non-member loss distribution is observed to have *larger* variance than the
member one.

**The prevailing regularization toolkit** attacks overconfidence by smoothing the output
distribution. The relevant primitive is the softmax-cross-entropy objective
`ell_CE = -sum_c y^c log p^c` with `p = softmax(logits)`, whose gradient with respect to the
logits is `p - y`; this vanishes exactly as `p → y`, i.e. as the loss goes to zero. That
is why standard training drives member losses to ~0 and creates the gap in the first place.

## Baselines

These are the prior defenses in the natural comparison set.

**Early stopping.** Halt training before the model overfits, trading a smaller
generalization gap for lower accuracy by reading off checkpoints at increasing epochs.

**Dropout (Srivastava et al. 2014).** Randomly mask units during training to prevent
co-adaptation and reduce overfitting.

**Label smoothing (Szegedy et al. 2016; Mueller, Kornblith & Hinton 2019).** Replace the
one-hot target with a mixture of the one-hot and the uniform distribution, equivalently
adding a KL term to the uniform:
`L = alpha * KL(U || p_theta(y|x)) + (1-alpha) * L_CE`. This caps how confident the model
becomes, lifting member losses off zero.

**Confidence penalty (Pereyra, Tucker, Chorowski, Kaiser & Hinton 2017).** Add the negative
predictive entropy to the loss to penalize peaked outputs:
`L = -sum log p_theta(y|x) - beta * H(p_theta(y|x))`, with
`H = -sum_c p^c log p^c`. The entropy gradient w.r.t. logit `i` is
`p_i ( -log p_i - H )`, a weighted deviation from the mean — it continuously pushes outputs
toward higher entropy.

**(Self-)distillation (Hinton et al. 2015; Zhang et al. 2019).** Train the model to match a
softened teacher of the same architecture, transferring inter-class structure and softening
predictions.

**Adversarial regularization (Nasr, Shokri & Houmansadr 2018) and MemGuard (Jia et al.
2019).** Train (or, for MemGuard, perturb outputs at test time against) a *surrogate* attack
network, so the target model is optimized to maximize the surrogate attacker's error.
MemGuard is a test-time output perturbation rather than a training-time objective.

**DP-SGD (Abadi et al. 2016).** Clip per-sample gradients to `L2`-norm `C` and add Gaussian
noise before each update, giving a worst-case differential-privacy guarantee that provably
bounds membership advantage.

## Evaluation settings

- **Datasets / models:** CIFAR-10 and CIFAR-100 (color images) with a 20-layer ResNet and an
  11-layer VGG; CH-MNIST (colorectal-cancer tissue images) with a 20-layer ResNet; Texas100
  (medical records, 6169 binary features, 100 classes) and Purchase100 (shopping records,
  600 binary features, 100 classes) with fully-connected nets. Each dataset is evenly split
  into folds; one fold trains the target model, others train shadow and attack models.
- **Attacks (the yardstick):** black-box metric attacks — loss thresholding, prediction-
  entropy and modified-entropy thresholding, and a learned neural-network attack on the full
  logits; white-box gradient-norm attacks on the input gradient and the parameter gradient
  (`l1` and `l2`). Thresholds for metric attacks are selected on shadow models/data.
- **Metrics:** *utility* = target test accuracy; *privacy* = attack accuracy on a balanced
  query set (50% is chance) and attack AUC (0.5 is a perfect defense). A defense is read off
  a privacy-utility curve as the hyperparameter is swept — better defenses approach the
  top-left corner (high accuracy, low AUC).
- **Training protocol:** SGD with momentum 0.9 and weight decay 1e-4, initial learning rate
  0.1 with step decays, fixed seed and identical training setup across defenses for fair
  comparison.

For this benchmark specifically, the harness trains on a 50/50 train/non-train split of the
full dataset and then runs a confidence-based membership inference attack on train versus
held-out examples; the primary score is `privacy_score = test_acc - max(mia_auc - 0.5, 0)`.
The optimizer (SGD + cosine annealing), architecture, data pipeline, and attack are fixed;
only the per-batch training objective is editable.

## Code framework

A fixed training loop can delegate its per-batch objective to a small component. The loop
draws a minibatch, runs the fixed model, hands over the `logits`, the integer `labels`, and
the current `epoch`, and backpropagates whatever scalar is returned through the fixed
optimizer. Everything except the body of `compute_loss` already exists.

```python
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Supplies the per-batch training objective for the fixed loop.

    The loop owns the model, optimizer (SGD + cosine annealing), data pipeline,
    and the attack. This class owns only the scalar loss that is backpropagated.
    """

    def __init__(self):
        # any fixed scalars the objective needs
        pass

    def compute_loss(self, logits, labels, epoch):
        # logits: (B, C) model outputs for the minibatch
        # labels: (B,)   ground-truth class indices
        # epoch:  current training epoch (0-indexed)
        #
        # The per-sample cross-entropy and its batch mean are the materials
        # already on hand:
        #   loss_ce_full = F.cross_entropy(logits, labels, reduction='none')
        #   loss_ce      = loss_ce_full.mean()
        #
        # TODO: return the scalar loss tensor for this minibatch.
        pass


def train(model, defense, data_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = defense.compute_loss(logits, targets, epoch)
            loss.backward()
            optimizer.step()
```
