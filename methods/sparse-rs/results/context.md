# Context: query-efficient black-box sparse (L0) adversarial attacks

## Research question

Given a classifier `f: [0,1]^{h x w x c} -> R^K` that exposes only output scores, craft an
untargeted adversarial example for a correctly classified image `x` of class `y` by changing
at most `k` spatial pixels. A pixel counts as changed if any color channel differs from the
original, and once a pixel is chosen its channels may take any value in `[0,1]`. The attack
succeeds when `argmax_r f_r(x') != y`. The query budget is the main resource, because each
score evaluation costs a full model forward pass.

A useful solution must satisfy five constraints at once: use scores only, never gradients or a
surrogate model; keep every candidate inside the exact `L0` budget; spend far fewer than the
input dimension `d = h*w*c` queries; scale from small `32x32` images up to `224x224` ImageNet
(`d = 150,528`); and be strong enough to serve as an *accurate robustness evaluation*, i.e.
not under-report a model's vulnerability the way a weak attack would.

## Background

**The vulnerability and the `L_p` orthodoxy.** Neural networks change their prediction under
small input perturbations (Biggio et al. 2013; Szegedy et al. 2014). The dominant attack
literature constrains the perturbation in an `L_inf`, `L_2`, or `L_1` norm — a *dense* change
spread thinly over all pixels. **Sparse** attacks pursue the opposite extreme: perturb only a
handful of pixels, but allow each to change a lot. The natural sparse budget is the `L0`
"norm" `||delta||_0 = #{i : delta_i != 0}`; such perturbations are visible but localized, do
not alter semantic content, and can be realized physically (stickers, tags, occlusions).

**Why the `L0` ball breaks continuous optimization.** Dense `L_inf`/`L_2`/`L_1` attacks
operate on continuous norm balls: convex bodies with smooth boundaries, where projected
gradient descent glides — take a step, clip back, repeat — and the PGD-based robustness
estimate is known to be reliable (Madry et al. 2018). The `L0` feasible set is different: it
is the union of all `k`-dimensional coordinate subspaces, indexed by the chosen support.
Projecting onto it means *selecting* which coordinates stay nonzero — a discrete top-`k`
operation, not a continuous clip. A gradient step spreads mass over many coordinates while the
projection abruptly discards all but `k`, so the support lurches discontinuously and the
iteration settles on poor supports. Empirically, PGD-style `L0` attacks give *inaccurate*
(too optimistic) robustness estimates — the discrete structure of the `L0` ball is simply not
amenable to continuous optimization. In the black-box setting it is worse: estimating a useful
gradient by finite differences burns many score queries before the attack commits to anything.

**The two-part structure of a sparse perturbation.** A sparse perturbation has two separable
objects: a **support** `M` of perturbed pixels, and the **values** placed on that support. The
budget constrains `M`, not the values. Because the input box is `[0,1]`, a selected pixel can
be pushed to any color in the cube `[0,1]^c` without paying extra `L0` cost.

**The linearized query model (a pre-method fact about the problem).** For a binary input
`x in {0,1}^d` and a linear score, the exact white-box sparse attack is simple: fold the label
and current value into an effective weight vector `w_hat_x = y*w_x ⊙ (1 - 2x)`, so flipping
coordinate `i` contributes one coefficient, and the optimal `k`-sparse attack chooses the `k`
smallest entries of `w_hat_x`. The black-box obstacle is that `w_hat_x` is hidden: a naive
strategy estimates the weights coordinate by coordinate at `O(d)` query cost, which on
ImageNet (`d = 150,528`) is already outside a practical budget. But the exact `k` smallest are
more than an attack usually needs — a relaxed target is to find `k` coordinates among the `m`
smallest for some `m > k`. This reframes query efficiency as a hitting-time question over
subsets, and since modern ReLU networks are piecewise-linear (Arora et al. 2018), a linear
analysis holds approximately in a small neighborhood of `x`. (These are facts about the
problem structure, knowable before any specific attack.)

## Baselines

The prior sparse attacks a new method would be measured against, with the gap each leaves.

**JSMA — Jacobian Saliency Map Attack (Papernot et al., EuroS&P 2016; arXiv:1511.07528).**
White-box, targeted. Build a *saliency map* from the Jacobian of the outputs w.r.t. the input,
ranking pixels by how much perturbing them raises the target logit while lowering the others,
then greedily perturb the most salient pixel(s). Core idea: support selection by first-order
sensitivity. **Gap:** needs white-box gradients (a Jacobian per class); a score-only version
requires expensive gradient estimation at `O(d)`/`O(m)` cost; greedy and not budget-bounded.

**SparseFool (Modas et al., CVPR 2019; arXiv:1811.02248).** White-box, geometry-inspired.
Linearizes the decision boundary in the DeepFool style to get a minimal crossing direction,
then sparsifies it. Core idea: exploit local linear geometry of the boundary. **Gap:**
white-box; built for `L1`; a single linearization is a poor model of the discrete `L0` set.

**CornerSearch (Croce & Hein, ICCV 2019; arXiv:1909.05040).** Black-box, score-based, sparse.
Probes color-cube corners `{0,1}^c` per pixel to build an ordering of pixels by adversarial
impact, then searches over small subsets in that ordering. Core idea: a black-box "saliency"
ordering from corner probes. **Gap:** the initial scoring phase alone costs about `8*h*w`
queries (`8*224*224 ~ 401k` on ImageNet) — orders of magnitude over any practical budget — and
it *minimizes* the number of changed pixels rather than attacking at a fixed budget.

**ADMM `L0` attack (Zhao et al. 2019).** Black-box alternating-direction optimization that
first finds a successful perturbation and then shrinks its `L0`-norm. **Gap:** minimizes `L0`
rather than attacking at a budget, and is query-heavy.

**Pointwise Attack (Schott et al. 2019).** Greedily resets perturbed pixels and resamples,
used to probe robust MNIST models. **Gap:** designed to minimize changed pixels on small
images; not query-efficient or scalable.

**PGD0 — projected gradient descent on the `L0` ball (Croce & Hein 2019).** White-box: take a
gradient step, then project onto the `L0` ball by keeping the top-`k` coordinates by
magnitude. **Gap:** white-box; the discrete top-`k` projection makes the iteration unstable
and the resulting robustness estimate inaccurate.

**Finite-difference gradient estimation (Ilyas et al. 2018).** Estimate `∇_z L` from pairs of
score queries and feed it to a white-box attack. **Gap:** reading the full gradient is `O(d)`
queries — the cost the linearized analysis above flags as prohibitive.

**Random search for dense black-box attacks (Andriushchenko, Croce, Flammarion, Hein, ECCV
2020; arXiv:1912.00049).** For the *dense* `L_inf`/`L_2` threat model, a random-search attack
reaches state-of-the-art query efficiency with a *carefully designed sampling distribution*:
candidates constructed to sit on the *boundary* of the `L_p` ball (full budget used each
step); *localized* square-shaped updates constant across channels (matched to conv receptive
fields); a specific initialization (vertical stripes / a grid); a *piecewise-constant decaying
schedule* halving the touched fraction `p^(i)` at fixed iteration thresholds
`{10,50,200,...,8000}` for `N=10000` (rescaled for other `N`), in analogy to step-size decay;
and the margin objective `f_y - max_{r!=y} f_r`, all controlled by a single knob `p_init`.
**Gap:** this geometry — "on the boundary of the ball", "a square of side `sqrt(p*w^2)`" — is
defined for a *continuous* `L_p` ball. For the `L0` set the budget is *which coordinates* are
nonzero, not a radius; "boundary of the ball" and "square of a given area" have no meaning. The
transferable part is only the derivative-free accept-if-improve skeleton, the margin objective,
the single-knob decaying schedule, and "use the budget maximally"; the sparse setting still
needs its own way to sample feasible supports and values.

## Evaluation settings

- **Datasets / models.** CIFAR-10 (`32x32x3`, 10 classes) and ImageNet (`224x224x3`, 1000
  classes, `d = 150,528`); MNIST for robust generative models. On standard undefended models a
  strong `L0` attack saturates trivially, so the informative setting is *adversarially robust*
  targets — e.g. `L2`-adversarially-trained PreActResNet-18 on CIFAR-10 (Rebuffi et al. 2021;
  Augustin et al. 2020; Engstrom et al. 2019, from the RobustBench model zoo), and
  normally-trained VGG-16-BN / ResNet-50 on ImageNet.
- **Budgets.** `L0` budget `k` (pixel space) — e.g. `k in {50,150}` on ImageNet, `k=24` on
  CIFAR-10, `k=12` on MNIST; query budget `N` (e.g. `N=10000`).
- **Metrics.** Success rate (fraction of initially-correct points pushed to misclassification)
  versus number of queries; for robustness evaluation, *robust error* = classification error
  on the crafted adversarial examples; query statistics (mean / median queries on successful
  points). A candidate is valid only if it respects the `L0` budget and stays in `[0,1]`.
- **Protocol.** Take the points initially classified correctly; run the attack under the fixed
  query budget; report success rate / robust error. Random-search methods are repeated over a
  few seeds to report variance.

## Code framework

A score-based black-box attack plugs into a generic random-search harness. What already
exists: a model returning logits for a batch (the only oracle), the margin objective whose
sign is the misclassification certificate, and the outer loop that proposes a candidate,
queries it, and keeps it iff it improves. What is *not* settled — and is the whole problem —
is how a candidate that respects the `L0` budget should be proposed and how the proposal should
change over the run. That is the single empty slot below.

```python
import torch
import torch.nn.functional as F


def margin_loss(model, z, y):
    """Untargeted margin: f_y(z) - max_{r != y} f_r(z). Negative <=> misclassified.
    One forward pass = one query. The only signal the black-box attack gets."""
    with torch.no_grad():
        logits = model(z)
    u = torch.arange(z.shape[0], device=z.device)
    correct = logits[u, y].clone()
    logits[u, y] = -float("inf")
    best_other = logits.max(dim=-1)[0]
    return correct - best_other


def propose_candidate(x_orig, z_best, state, it, n_queries, pixels):
    """The sampling distribution + how a feasible L0 candidate is drawn and how the
    proposal shrinks over iterations. This is exactly what has to be designed.

    Must return a candidate image in [0,1] differing from x_orig in at most `pixels`
    spatial positions, plus whatever bookkeeping the search needs.
    """
    # TODO: the proposal/update rule we will design (feasible L0 candidate from the
    #       current iterate), and how it evolves across iterations.
    pass


def run_attack(model, images, labels, pixels, device, n_classes):
    """Generic accept-if-improve random-search loop. The model is queried only through
    margin_loss; candidates must stay within the L0 budget and [0,1]."""
    model.eval()
    x = images.to(device)
    y = labels.to(device)

    state = None                      # bookkeeping the proposal needs (TODO: define)
    z_best = x.clone()                # current best adversarial candidate
    # TODO: initial feasible candidate (which pixels, what colors)
    loss_best = margin_loss(model, z_best, y)

    for it in range(1, n_queries):
        z_new, state_new = propose_candidate(x, z_best, state, it, n_queries, pixels)
        loss_new = margin_loss(model, z_new, y)
        improved = loss_new < loss_best
        # TODO: accept the move where it improves; update z_best, loss_best, state
        pass

    return z_best.detach()
```

The outer loop is fixed; `propose_candidate` (and the initialization and the accept update it
implies) is the one slot the method fills in.
