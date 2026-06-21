# Context: score-based black-box adversarial attacks under an Lp budget (circa 2018-2019)

## Research question

A classifier `f : [0,1]^d -> R^K` maps an image to per-class scores; it predicts
`argmax_k f_k(x)`. An attacker wants to find, for a correctly classified `(x, y)`, a perturbed
image `x_hat` that the model gets wrong while staying visually close to `x`:

```
argmax_k f_k(x_hat) != y,   ||x_hat - x||_p <= eps,   x_hat in [0,1]^d.
```

Equivalently, minimize a loss `L(f(x_hat), y)` subject to the `Lp`-ball and image-box
constraints. The hard part is the **threat model**: this is a *score-based black-box* setting.
The attacker can query the model on inputs of its choosing and read back the output scores
(logits or softmax), but has **no access to weights, no access to gradients**, and no ability
to backpropagate. Every query costs something — in a real deployment (an ML API) queries are
rate-limited, billed, or monitored — so the attacker is given a hard **per-sample query
budget** `N`. The two quantities that matter are the **success rate** (fraction of inputs
successfully flipped within budget) and the **average number of queries** used.

Two structural facts about the problem shape everything below. First, for the `Linf` threat
model the perturbation is constrained component-wise, `|x_hat_i - x_i| <= eps`, so the feasible
set is a high-dimensional box and its *corners* — where every component is pushed to `x_i +/-
eps` (clipped to `[0,1]`) — are the extreme points. Second, the images are processed by
**convolutional** networks, whose first layer correlates small local `s x s` patches of the
input against learned filters — so the *spatial structure* of a perturbation, not just its
per-pixel budget, governs how strongly it moves the network's internal activations.

## Background

**The score-based black-box regime and its query cost.** By this time adversarial examples
are a central concern for safety-critical ML (Szegedy et al. 2014; Goodfellow et al. 2015).
White-box attacks like PGD (Madry et al. 2018) are strong but assume gradient access. Realistic
attackers often only see scores. The dominant black-box recipe is **gradient estimation by
finite differences / sampling**: probe the model at `x +/- sigma u` for random directions `u`
and assemble an estimate of `grad_x L`. NES-style estimators (Ilyas et al. 2018) and SPSA
(Uesato et al. 2018) do exactly this, then run PGD on the estimated gradient. The estimate's
variance scales with the input dimension `d`, which for ImageNet is ~150k, so a single usable
gradient costs many queries; whole attacks run into the tens of thousands of queries.

**Gradient masking / obfuscated gradients.** Many proposed defenses do not actually remove
adversarial examples; they merely make the *gradient signal* useless — shattered, stochastic,
or vanishing gradients (Athalye et al. 2018; Papernot et al. 2017). Athalye et al. 2018
documented this as the reason a slate of ICLR defenses fell to adaptive attacks that follow
local gradient information.

**The corner observation for `Linf`.** A line of work observed empirically that successful
`Linf` perturbations almost always take values `+/- eps` in every component — they sit at a
corner of the `Linf`-ball (Moon et al. 2019; Al-Dujaili & O'Reilly 2019). This converts the
continuous search into a **discrete** one over `{-eps, +eps}^d`, opening the door to
combinatorial optimization and large per-step moves rather than tiny gradient nudges.

**Random search as a black-box optimizer.** Random search (Rastrigin 1963; Schumer & Steiglitz
1968) is one of the oldest derivative-free methods: at each iteration propose a random
perturbation of the current iterate, evaluate the objective, and keep the proposal only if it
improves. It is a greedy hill-climber that uses *only function values*, never gradients, and is
known to perform surprisingly well across many black-box problems (Zabinsky 2010). The classic
version samples updates uniformly on a sphere of fixed radius; the freedom — and the leverage —
is entirely in *which distribution the proposed update is drawn from* at each step.

**Convolutional structure and frequency sensitivity.** Two empirical facts about CNNs are
load-bearing here. (i) The first convolutional layer convolves the input with small `s x s`
filters; the change a perturbation induces in a first-layer activation `z_{u,v} = (delta * w)_{u,v}`
depends on how the nonzero entries of `delta` overlap the `s x s` receptive window — so the
*spatial layout* of where you spend perturbation budget matters, not only how much you spend.
(ii) CNNs are disproportionately sensitive to certain high-frequency input patterns; structured
high-frequency perturbations move predictions more than unstructured noise of the same norm
(Yin et al. 2019). These observations describe how existing networks behave.

**The margin / hinge objective.** For untargeted attacks a natural surrogate for "is it
misclassified" is the margin `L(f(x_hat), y) = f_y(x_hat) - max_{k != y} f_k(x_hat)`: it is
positive while `y` is still on top and crosses zero exactly when some other class overtakes
`y`. This is the standard CW-style objective (Carlini & Wagner 2017) and is the quantity a
success test must check anyway.

## Baselines

These are the prior methods a new black-box attack would be measured against and reacts to.

**NES / SPSA gradient-estimation attacks (Ilyas et al. 2018; Uesato et al. 2018; Bhagoji et
al. 2018).** Estimate the gradient of the loss from score queries — sample directions `u_i`,
form `grad ~ (1/sigma) E[L(x + sigma u_i) u_i]` (NES) or a simultaneous-perturbation finite
difference (SPSA) — then take a PGD step on the estimate, repeating. Dimensionality-reduction
tricks (PCA of the data, autoencoder latent search, low-dimensional noise priors) reduce the
per-step cost.

**SimBA — simple black-box attack via orthonormal search (Guo et al. 2019).** A random-search
variant for `L2`: maintain a perturbation; at each step pick a direction from a fixed
orthonormal basis (pixel basis or DCT basis), try adding `+alpha` or `-alpha` times that
direction, and keep whichever lowers the loss.

**Discrete corner-search attacks (Moon et al. 2019; Al-Dujaili & O'Reilly 2019).** Exploit the
corner observation: restrict every component to `+/- eps` and search the discrete cube. Both
divide the image by a **coarse, a-priori-fixed grid**, allow component-wise flips of `-eps`/`+eps`
within that lower-dimensional space, run a local / combinatorial search, then refine the grid
and repeat; Al-Dujaili & O'Reilly motivate it as estimating gradient *signs*.

**Evolutionary / tiling attacks (Meunier et al. 2019, building on Ilyas et al. 2019).** Reduce
the search dimension with the "tiling trick" — divide the perturbation into a set of tiles and
evolve the tile values with discrete/continuous evolutionary algorithms — reaching near-SOTA
`Linf` query efficiency. The size and position of tiles are fixed at the start; the method
searches tile *values* over a fixed partition of the image.

## Evaluation settings

The yardsticks already standard in the black-box literature at this time:

- **Datasets / models.** Untargeted attacks on ImageNet classifiers (Inception-v3, ResNet-50,
  VGG-16-BN), plus smaller-scale studies on CIFAR-10 and MNIST classifiers, including
  adversarially trained / robust models, to test whether the attack also serves as a
  robustness probe.
- **Threat models.** `Linf` with a fixed radius (e.g. `eps = 0.05` on ImageNet) and `L2` with a
  fixed radius (e.g. `eps = 5`), each enforced by projection (clipping for `Linf`) and the
  image-box constraint `[0,1]^d`.
- **Protocol.** A hard per-sample **query budget** `N` (e.g. 10,000), one or more random
  restarts, a fixed random seed; the attack queries the model only through its score output;
  one forward pass on a candidate counts as one query (a batch of `B` candidates counts as
  `B`); an attack stops querying a sample as soon as it is flipped.
- **Metrics.** **Success rate** (primary) = fraction of inputs flipped within budget; **average
  / median number of queries** over successful samples (lower is better) as the efficiency
  tie-break; clean accuracy as a sanity check.

## Code framework

The attack plugs into a fixed evaluation harness that wraps the model as a query-counting,
score-only oracle and enforces the `Linf` and `[0,1]` constraints per sample. The substrate
below is generic black-box random-search machinery: initialize a candidate, propose a new
candidate with a score-only rule, project it back to the feasible set, query the oracle, and
keep the accepted candidate.

```python
import torch
import torch.nn.functional as F


def margin_and_loss(model, x, y):
    """Untargeted margin: f_y - max_{k!=y} f_k. <= 0 iff already misclassified."""
    logits = model(x)
    xent = F.cross_entropy(logits, y, reduction="none")
    u = torch.arange(logits.shape[0], device=x.device)
    y_corr = logits[u, y].clone()
    logits = logits.clone()
    logits[u, y] = -float("inf")
    y_other = logits.max(dim=-1)[0]
    margin = y_corr - y_other
    return margin, margin  # another score loss, such as -xent, can fill this slot


def project_linf_box(x_candidate, x, eps):
    x_candidate = torch.min(torch.max(x_candidate, x - eps), x + eps)
    return torch.clamp(x_candidate, 0.0, 1.0)


def make_candidate(x, x_best, labels, it, n_queries, eps):
    # TODO: design the proposal used by the score-only search.
    return x_best.clone()


@torch.no_grad()
def run_attack(model, images, labels, eps, n_queries):
    """Score-based black-box Linf attack via random search.

    model:     query-only oracle returning logits (no gradients).
    images:    (N, C, H, W) in [0, 1];  labels: (N,)
    Constraint: ||x_adv - x||_inf <= eps,  x_adv in [0, 1].
    Budget:    n_queries forward queries per sample; maximize success rate.
    """
    x = images.clone()
    x_best = x.clone()
    margin_min, loss_min = margin_and_loss(model, x_best, labels)  # 1 query

    for it in range(n_queries):
        active = (margin_min > 0).nonzero().flatten()  # not-yet-fooled samples
        if len(active) == 0:
            break

        x_new = make_candidate(x[active], x_best[active], labels[active],
                               it, n_queries, eps)
        x_new = project_linf_box(x_new, x[active], eps)

        margin, loss = margin_and_loss(model, x_new, labels[active])  # 1 query each

        loss_improved = loss < loss_min[active]
        loss_min[active] = torch.where(loss_improved, loss, loss_min[active])
        accepted = loss_improved | (margin <= 0)

        sel = accepted.float().view(-1, *([1] * (x.ndim - 1)))
        x_best[active] = sel * x_new + (1 - sel) * x_best[active]
        margin_min[active] = torch.where(accepted, margin, margin_min[active])

    return x_best
```

The outer loop, the oracle, the margin objective, the projection, and the accept rule are in
place. The neutral `make_candidate` slot is the part the attack still has to design.
