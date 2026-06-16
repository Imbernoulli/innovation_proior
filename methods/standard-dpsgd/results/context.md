# Context: training deep networks on sensitive data with formal privacy (circa 2014-2016)

## Research question

Deep neural networks have started to win at image classification, speech, language, and game
play, and the wins are powered by large, often crowdsourced, datasets that contain sensitive
personal records. The trained model is then shipped — increasingly onto phones and tablets,
where its parameters are exposed to inspection. The problem is that a high-capacity network
can memorize fine details of individual training examples, and a determined adversary can pull
them back out: Fredrikson et al. (2015) reconstruct recognizable face images from a facial-
recognition model with only black-box access, and Shokri & Shmatikov (2015) study deep learning
under a much stronger access model where participants disclose sanitized updates. So the threat
model that matters is strong: an adversary who knows the entire training procedure, can read the
final model parameters, and may even control every training record except the one whose privacy
we are trying to protect.

The precise goal is to train such a model with a rigorous, worst-case privacy guarantee —
specifically `(ε, δ)`-differential privacy — at a *small*, "single-digit" `ε`, while the model
is non-convex, has several layers, and tens of thousands to millions of parameters, and while
keeping the loss in accuracy, training speed, and software complexity modest. What makes this
hard is the combination: prior work could get strong privacy on *convex* models with few
parameters, or could handle a real neural network but only at a privacy loss so large the
guarantee was vacuous. Closing that gap — deep, non-convex networks *and* a meaningful `ε` —
is the problem.

## Background

The field state rests on a small number of load-bearing definitions and tools.

**Differential privacy** (Dwork, McSherry, Nissim & Smith 2006; the approximate variant with
the additive `δ`, Dwork et al. 2006 "Our Data, Ourselves"). In the add/remove convention used
by the subsampling analysis, two databases are *adjacent* if one is obtained from the other by
adding or removing a single record; under replacement adjacency the same mechanisms pay only a
constant-factor change in sensitivity. A randomized mechanism `M : D → R` is `(ε, δ)`-
differentially private if for every pair of adjacent inputs `d, d'` and every set of outputs `S`,

```
Pr[M(d) ∈ S] ≤ e^ε · Pr[M(d') ∈ S] + δ.
```

Pure `ε`-DP is the `δ = 0` case; the `δ` term (kept `≪ 1/|d|`) allows the `e^ε` bound to fail
with tiny probability. Three properties make it usable: it *composes* (a sequence of private
mechanisms is private), it degrades gracefully under *group privacy*, and it is robust to any
side information the adversary holds.

**Sensitivity and the Gaussian mechanism.** The standard recipe for privatizing a deterministic
vector-valued query `f` is additive noise scaled to its *sensitivity* `S_f`, the largest change
`‖f(d) − f(d')‖` over adjacent `d, d'`. The Gaussian mechanism releases

```
M(d) = f(d) + N(0, S_f² σ² I),
```

and a single application is `(ε, δ)`-DP whenever `δ ≥ (4/5) exp(−(σε)²/2)` with `ε < 1`
(Dwork & Roth, Thm 3.22); equivalently, for unit-sensitivity `f`, taking `σ = √(2 ln(1.25/δ)) / ε`
suffices. The noise is calibrated in the `L2` norm, which is what makes Gaussian noise the
natural match. Critically, this analysis is *post hoc*: many `(ε, δ)` pairs satisfy the same
inequality, so one can fix the noise and read off the privacy afterward.

**Composition and the privacy accountant.** A private computation that touches the data many
times spends privacy each time, and the spent budget must be tracked — the *privacy accountant*
of McSherry (PINQ). The cheapest accounting is *basic composition*: `k` applications of an
`(ε, δ)` mechanism give `(kε, kδ)`. The better tool is the *advanced (strong) composition
theorem* (Dwork, Rothblum & Vadhan 2010, "Boosting and Differential Privacy"): the `k`-fold
adaptive composition of `(ε, δ)`-DP mechanisms is `(ε̃, kδ + δ')`-DP with

```
ε̃ = ε √(2k ln(1/δ')) + k ε (e^ε − 1)/(e^ε + 1) ≈ ε √(2k ln(1/δ')),
```

so that for many small steps the privacy parameter grows only with the *square root* of the
number of steps rather than linearly. The proof works by viewing the **privacy loss as a random
variable** — for output `y` and inputs `d, d'`, `c(y) = ln( Pr[M(d)=y] / Pr[M(d')=y] )` — and
noting that under adaptive composition the total loss is the *sum* of per-step losses, each
roughly in `[−ε, ε]` but with average behaviour around `ε²`. A useful concrete fact in that
frame: for a Gaussian mechanism whose neighboring outputs differ by `Δ`, if the loss is
`log p_d/p_{d'}` and the output is drawn from `M(d)`, then `c(Y) ∼ N(Δ²/2σ², Δ²/σ²)`; drawing
from the other neighbor flips the mean sign and keeps the same variance. The full distribution,
not just one tail point, is known in closed form.

**Privacy amplification by subsampling** (Kasiviswanathan et al. 2011; Beimel et al. 2014).
Running an `(ε, δ)`-DP mechanism not on the full dataset but on a uniformly random `q`-fraction
subsample tightens the guarantee to roughly `(O(qε), qδ)`: an example that is unlikely to even
be looked at is, in expectation, better protected.

**Deep learning by SGD.** A network defines a parameterized map as a composition of affine
layers and nonlinearities (ReLU, sigmoid). Training minimizes an empirical loss
`L(θ) = (1/N) Σ_i L(θ, x_i)`, which for deep nets is non-convex. The workhorse is mini-batch
stochastic gradient descent: at each step form a batch `B`, compute
`g_B = (1/|B|) Σ_{x∈B} ∇_θ L(θ, x)` as an unbiased estimate of `∇L(θ)`, and step
`θ ← θ − η g_B`. Two diagnostic facts about plain SGD matter here. First, there is no a-priori
bound on the norm of an individual `∇_θ L(θ, x)` — it can be arbitrarily large for an outlier
example. Second, *gradient clipping* — rescaling a gradient down when its norm exceeds a
threshold — is already a routine ingredient of deep-network SGD for stability reasons, though
in that non-private setting it is applied to the *averaged* batch gradient, after summation.
Per-example gradients, which plain batched autodiff does not expose, can nonetheless be computed
efficiently (Goodfellow 2015).

## Baselines

These are the prior approaches a new private-training method would be measured against and react
to.

**Output and objective perturbation for private convex ERM (Chaudhuri, Monteleoni & Sarwate
2011; Bassily, Smith & Thakurta 2014).** For empirical risk minimization with an
`L`-Lipschitz convex loss over a bounded set, privacy can be obtained either by adding noise to
the final minimizer (output perturbation) or by adding a random linear term to the objective
before minimizing (objective perturbation). Bassily et al. give matching upper and lower bounds
on the excess empirical risk for both `(ε, 0)`- and `(ε, δ)`-DP, e.g. `Õ(√p / ε)` excess risk
for the general convex case, with a noisy gradient-descent algorithm achieving it. The analysis
leans on convexity twice over: the Lipschitz constant gives a clean per-step gradient-sensitivity
bound, and convexity is what lets the sensitivity of the *minimizer* be characterized at all.
**Gap:** for a non-convex, many-layer network there is no tight handle on how the final
parameters depend on any one record, so output/objective perturbation has nothing to calibrate
its noise against except a worst case, which is so pessimistic it would wipe out the model's
utility; and the risk bounds, being convex, do not transfer.

**Noisy stochastic gradient updates (Song, Chaudhuri & Sarwate 2013; the stochastic variant in
Bassily, Smith & Thakurta 2014).** Rather than perturb the endpoint, perturb the *process*:
inject noise into each SGD gradient step, bounding the per-step contribution of an example and
using the randomness of which example is drawn. Empirically, raising the batch size reduces the
relative impact of the noise. Bassily et al. run such an algorithm for `T = n²` steps and reach
the optimal convex risk, observing that to run a first-order method for enough steps one
*appears* to need the strong composition of `(ε, δ)`-privacy. **Gap:** these treatments are
convex, and — more sharply — the privacy of the many-step process is accounted by feeding each
step's `(ε, δ)` into the generic strong composition theorem, which is wasteful because it is
blind to the specific noise distribution being composed.

**Strong composition as the accounting tool (Dwork, Rothblum & Vadhan 2010).** As the privacy
accountant for an iterative private algorithm, advanced composition is the best generic option:
`ε` grows like `√(k log(1/δ'))`. **Gap:** it is a bound for *arbitrary* `(ε, δ)` mechanisms. It
does not inspect the shape of the noise distribution being composed, so for a repeated noisy
gradient computation it can pay for privacy as if only one tail point of each step were known.
It also accumulates a `kδ` term in the `δ` part; with the number of steps `T` far exceeding
`1/q` (each example revisited many times) and `δ` taken very small, that slack can make the
reported budget too large to be useful.

## Evaluation settings

The natural yardsticks already in use, all on public datasets with a long benchmark history:

- **MNIST** — 60,000 training / 10,000 test grayscale `28×28` handwritten-digit images,
  10 classes. The standard small-scale image-classification benchmark.
- **CIFAR-10** — 50,000 training / 10,000 test color `32×32` natural images, 10 classes; a
  harder benchmark where private training is known to be far more demanding.
- Protocol: train a convolutional or fully-connected classifier by mini-batch SGD; report test
  **accuracy** as the utility metric, alongside the **privacy budget** `(ε, δ)` consumed (with
  `δ` fixed small, e.g. `10⁻⁵`). Privacy and accuracy are read off against each other — the
  same model at several `ε` values, or several models at a fixed `ε` budget. Relevant knobs that
  exist before any method is chosen: the lot/batch size, the number of epochs (hence training
  steps `T`), the subsampling rate `q`, the learning-rate schedule, and the per-coordinate noise
  scale.

## Code framework

The private mechanism plugs into a fixed mini-batch SGD training harness. The model, the data
pipeline, the optimizer (`SGD` with momentum), the learning-rate schedule, the per-example
gradient computation, and the training/evaluation loops are all given. There is also a privacy-
accounting utility that can be called on a declared per-step randomized release and then composed
over the planned number of training steps. What is not settled is how the per-example gradients
of a lot get turned into the single aggregate gradient that the optimizer steps on, while keeping
the full run inside a fixed `(ε, δ)` budget. That is the one empty slot.

```python
import torch


# ---- given: privacy accounting for a declared per-step private release ----
def compute_epsilon(steps, release_description, q, delta):
    """Return accumulated epsilon for `steps` planned releases at target `delta`."""
    ...

def calibrate_release_to_epsilon(target_epsilon, steps, q, delta):
    """Choose the release parameters that spend at most target_epsilon."""
    ...


class PrivateGradientMechanism:
    """Turns a batch of per-example gradients into one aggregate gradient for the optimizer
    step, subject to the fixed (target_epsilon, target_delta) budget. The per-example
    gradients arrive as a list of tensors, each [B, *param_shape]; the method returns a list
    of aggregate tensors, each [*param_shape]."""

    def __init__(self, release_parameters, expected_lot_size,
                 dataset_size, epochs, target_epsilon, target_delta):
        self.release_parameters = release_parameters
        self.expected_lot_size = expected_lot_size
        # ... remaining bookkeeping (dataset_size, epochs, targets) ...

    def aggregate(self, per_sample_grads, step, epoch):
        # TODO: the data-to-aggregate transformation we will design.
        pass


# existing private training loop the mechanism plugs into
def train(model, loss_fn, data_loader, optimizer, dp_mechanism):
    for inputs, targets in data_loader:                       # a random lot of expected size qN
        per_sample_grads = per_example_gradients(model, loss_fn, inputs, targets)
        agg = dp_mechanism.aggregate(per_sample_grads, step, epoch)  # the slot to design
        optimizer.zero_grad()
        set_param_grads(model, agg)                           # write agg into p.grad
        optimizer.step()                                      # ordinary SGD step on the aggregate
```

The harness supplies the per-example gradients of a lot; `aggregate` is where the
data-to-release transformation will live, and the accounting utility checks the declared
release against the budget before training starts. If several physical batches are accumulated
into one lot, the mechanism reports an aggregate scaled by the expected lot size.
