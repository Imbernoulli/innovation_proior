# Context: white-box sparse adversarial perturbations before mask-based gradient attacks

## Research Question

Given a classifier `f: [0,1]^{C x H x W} -> R^K`, a correctly classified image `x`, and its label `y`, craft an untargeted adversarial image `x'` that changes at most `k` spatial pixels and stays inside the image box. A spatial pixel counts once even if all color channels at that location are changed. The attack succeeds when `argmax_r f_r(x') != y`.

The useful regime is white-box evaluation: the attacker has forward and backward access to the model, and the output is used to estimate the robustness of adversarially trained models. A good method therefore has to satisfy four constraints at once. It must return a candidate that is exactly feasible under the spatial `L0` budget and the `[0,1]` box. It must exploit gradients rather than spend a separate query on every coordinate. It must search the combinatorial support well enough that robust accuracy is not overestimated. And it must be efficient enough to run with thousands of iterations, or with far fewer iterations inside adversarial training.

The hard part is not changing the value of a chosen pixel. Once a pixel is selected, its channels can move anywhere in `[0,1]` at no additional `L0` cost. The hard part is making a gradient method decide which `k` pixels should be selected.

## Background

Dense attacks such as FGSM and projected gradient descent became standard because `L_inf` and `L2` constraints have simple continuous projections: take a step that increases the attack loss, then clip or rescale back into the feasible set. Madry-style PGD can be restarted from random points and is treated as a strong first-order adversary in those smooth norm balls.

The `L0` feasible set has a different geometry. It is a union of coordinate subspaces, one for each support of size at most `k`. The Euclidean projection onto that set is a discrete top-`k` operation: keep the coordinates with largest magnitudes and zero the rest. A small dense gradient step followed by this projection can abruptly replace the support, and the resulting iteration is not a smooth projected descent in the way `L_inf` PGD is. Box constraints add another bookkeeping requirement: for every channel, the perturbation must stay in `[-x, 1-x]` so that `x'` is a valid image.

Sparse image perturbations also have two qualitatively different decisions. The support says where the image changes; the values say how those selected pixels move. Prior methods either make the support decision greedily, approximate a local decision boundary, project a dense update onto a top-`k` set, or randomly search feasible supports. None of those choices fully answers the white-box question: how should a differentiable attack use the gradient both to improve selected pixel values and to revise the selected support?

## Baselines

**JSMA (Jacobian Saliency Map Attack; Papernot et al. 2016).** A white-box targeted attack that builds saliency maps from the forward derivative and greedily changes high-saliency input features. It makes support decisions from local saliency, but it commits greedily and does not run a joint iterative optimization over support and values.

**SparseFool (Modas et al. 2019).** A white-box geometry-based sparse attack that repeatedly linearizes the classifier boundary in a DeepFool-like way, then converts the local step into a sparse perturbation. It is useful as a sparse baseline, but its local-linear boundary model can be brittle on adversarially trained models and it is not an exact fixed-budget support optimizer.

**PGD0 / sparse PGD projection (Croce and Hein 2019).** A white-box extension of PGD that takes a gradient step and projects the result to an `L0` or `L0 + L_inf` feasible set. It is fast and natural, but the top-`k` projection is exactly the discrete operation that can make the support unstable.

**Sparse-RS (Croce et al. 2022).** A black-box random-search attack that edits feasible sparse candidates with a decaying schedule and is strong for query-based fixed-budget sparse evaluation. Its gap in the white-box setting is that it does not use the model gradient even when that gradient is available.

**SAIF (Imtiaz et al. 2022).** A sparse adversarial framework that separates the perturbation values from support-like variables and uses Frank-Wolfe-style updates. It shows that treating sparsity structure explicitly is promising, but it leaves open whether a PGD-like white-box method can make the support update simple, direct, and efficient.

## Evaluation Settings

The standard experiments use image classifiers on CIFAR-10, CIFAR-100, GTSRB, and ImageNet-style subsets, especially models trained for `L_inf`, `L2`, `L1`, or `L0` robustness. Undefended models are usually too easy, so the informative question is how much robust accuracy remains on already robust targets.

Budgets are fixed spatial `L0` budgets such as `k=10`, `15`, `20`, or larger values on higher-resolution images. The magnitude budget for selected pixels is the image range itself: a selected pixel may move anywhere allowed by the `[0,1]` box. The attack is run only on initially correctly classified samples when measuring robust accuracy, and a candidate is valid only if its changed spatial support has size at most `k` and every pixel remains in range.

The metrics are robust accuracy and attack success rate under a fixed iteration or query budget. A robust evaluation should report the best adversarial candidate found during the run, not merely the last iterate, because sparse support search can move between qualitatively different supports.

## Code Framework

The surrounding attack harness is ordinary: evaluate the model, compute an attack loss, backpropagate, update internal attack state, clamp candidates into the image box, and keep the strongest candidate found. The unsolved slot is the sparse step: it has to update both a support decision and selected pixel values while guaranteeing a feasible image.

```python
import torch
import torch.nn.functional as F


def attack_loss(model, z, y):
    """A white-box untargeted loss. The sparse step decides how this gradient is used."""
    return F.cross_entropy(model(z), y, reduction="none")


def sparse_attack_step(model, x, y, state, k):
    """Open slot.

    Given the current internal attack state, use a white-box gradient to revise
    which spatial pixels are changed and how their channel values move. Return a
    valid candidate x_adv with at most k changed spatial pixels and 0 <= x_adv <= 1.
    """
    raise NotImplementedError


def run_sparse_attack(model, x, y, k, steps):
    model.eval()
    state = None
    best = x.clone()
    best_loss = torch.full((x.shape[0],), -float("inf"), device=x.device)
    for _ in range(steps):
        x_adv, state = sparse_attack_step(model, x, y, state, k)
        loss = attack_loss(model, x_adv, y)
        improve = loss > best_loss
        best_loss[improve] = loss[improve]
        best[improve] = x_adv[improve].detach()
    return best
```

The method has to fill `sparse_attack_step`: how to represent the sparse candidate, how to make the support decision interact with gradients, which sign convention the attack loss uses, and how to keep the result exactly feasible at every iteration.
