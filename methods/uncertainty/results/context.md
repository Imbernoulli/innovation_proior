## Research question

A single deep network is increasingly asked to do several things at once from one shared
representation — predict per-pixel depth *and* segment the scene semantically *and* localise object
instances, or predict camera rotation *and* translation, or classify *and* detect. The appeal is
concrete: a shared encoder is cheaper at inference (one forward pass for all outputs, which matters
for real-time robotics), and the auxiliary tasks act as an inductive bias that can regularise the
representation and improve generalisation on the task one actually cares about. The standard way to
train such a model is to add up the per-task losses into one scalar and backpropagate it:
`L_total = Σ_i w_i L_i`.

The weights `w_i` must be chosen somehow. The per-task losses are not commensurable: a depth
regression loss is measured in (squared) meters or millimeters, a classification loss is a
dimensionless cross-entropy in nats, an instance-vector loss is in pixels. Their raw magnitudes can
differ by orders of magnitude, and the appropriate relative weighting depends on those measurement
scales and on how noisy each task's labels are. Empirically, final model quality is sensitive to the
choice of `w_i`: there is a narrow band of weightings in which the joint model beats every
single-task model. The question is how to combine `K` heterogeneous task losses into a single
training objective whose *relative task weighting* is set automatically and adapts to the data,
rather than fixed by a manual search.

## Background

**Multi-task learning as inductive transfer.** The foundational idea (Caruana, *Multitask Learning*,
1998; Baxter 2000; Thrun 1996) is that learning several related tasks from a *shared* hidden
representation lets what is learned for one task help the others: the auxiliary tasks bias the
shared features toward structure that generalises, acting as a data-dependent regulariser. This is
why a coarse auxiliary objective (e.g. a superclass label, or surface normals alongside depth) can
improve a finer primary objective rather than merely competing with it — provided the auxiliary
signal is given an appropriate strength. It is also why the *weighting* matters: too little weight
and the auxiliary task contributes no useful bias; too much and it dominates the shared features and
hurts the primary task.

**Maximum-likelihood training and observation noise.** Most supervised losses are, up to constants,
negative log-likelihoods of a probabilistic observation model. For a regression output the usual
model is a Gaussian with the network output as its mean, `p(y | f^W(x)) = N(f^W(x), σ²)`, whose
negative log-likelihood is `(1/2σ²)||y - f^W(x)||² + (1/2)log σ² + const`. For a classification
output the usual model squashes the logits through a softmax and the loss is the cross-entropy,
`-log Softmax(f^W(x))_c`. The scalar `σ` here is the *observation noise* — how much the labels
scatter around the model's prediction. In ordinary single-task training `σ` is treated as a fixed
constant and folded into the learning rate / weight decay, so it never appears explicitly; the loss
is just the squared error.

**Heteroscedastic / mean-variance regression.** A line of work makes that noise an explicit, learned
quantity. Nix & Weigend (*Estimating the mean and variance of the target probability distribution*,
ICNN 1994) train a network with two outputs — a predicted mean `μ(x)` and a predicted variance
`σ²(x)` — by maximising the Gaussian likelihood, i.e. minimising
`Σ_i [ ||y_i - μ(x_i)||²/σ²(x_i) + log σ²(x_i) ]`. Le et al. (2005) develop the same idea for
heteroscedastic Gaussian-process regression. The point is that the noise variance need not be a
labelled target: it can be learned implicitly from the residuals, because the likelihood couples it
to the squared error. When the noise is allowed to depend on the input it is called *heteroscedastic*;
when it is a single constant for all inputs it is *homoscedastic*.

**Two kinds of predictive uncertainty.** In Bayesian modelling one distinguishes *epistemic*
uncertainty — uncertainty in the model parameters, reducible with more data, captured by a
distribution over weights — from *aleatoric* uncertainty — irreducible noise in the observations,
captured by a distribution over the output. Aleatoric uncertainty in turn splits into
heteroscedastic noise, where the scale depends on the input and may be predicted as a model output,
and constant observation-noise scales, which single-task training usually treats as fixed nuisance
constants and absorbs into the overall loss scale.

## Baselines

**Naive uniform / hand-tuned weighted sum.** The dominant approach in essentially all prior
multi-task vision systems: pick the `w_i` by hand (often all equal) and minimise `Σ_i w_i L_i`.
Used by OverFeat (Sermanet et al. 2014, classification + localisation + detection), Eigen & Fergus
(2015, depth + surface normals + semantic labels under a shared multi-scale architecture), UberNet
(Kokkinos 2016, many low/mid/high-level vision tasks in one network), MultiNet (Teichmann et al.
2016, detection + classification + segmentation), and Uhrig et al. (2016, semantic + instance under
a classification framing).

**PoseNet-style fixed scalar between heterogeneous units (Kendall, Grimes & Cipolla, ICCV 2015).**
For camera relocalisation the loss combines a rotation error (in quaternion units) and a translation
error (in meters) as `L_pos + β L_rot` with a single hand-tuned scalar `β` balancing the two
incommensurable quantities.

**Learning the weights directly as free parameters.** An obvious idea is to make the `w_i` in
`Σ_i w_i L_i` themselves trainable and let gradient descent set them.

**Heteroscedastic aleatoric regression (Nix & Weigend 1994; Le et al. 2005).** Learn an
input-dependent noise `σ(x)` jointly with the mean by maximising the Gaussian likelihood, giving the
loss `||y - μ(x)||²/(2σ(x)²) + (1/2)log σ(x)²`. This shows that a noise scale can be recovered from
residuals alone, with the logarithmic term keeping it finite, and that noisy observations can be
attenuated inside a single regression problem.

## Evaluation settings

The natural yardstick is dense multi-task scene understanding on a road-scene dataset. CityScapes
(Cordts et al. 2016) provides stereo road-scene imagery with per-pixel semantic labels over 20
classes, per-instance segmentation masks, and disparity/depth maps (from SGM stereo, treated as
pseudo-ground-truth; sky pixels assigned zero inverse depth), with 2,975 training and 500 validation
images at 2048×1024, plus a withheld test set scored on an online server. Experiments are run both
at full resolution and on a down-sampled "Tiny" version (128×256) to cut compute. The tasks are
semantic segmentation (pixel cross-entropy), instance segmentation (regressing each pixel's vector
to its instance centroid under an L1 loss, then clustering the votes), and per-pixel inverse-depth
regression (L1). Metrics are per task and on different scales: mean intersection-over-union for
segmentation, mean instance error in pixels, and mean / RMS inverse-depth error. The same model is
trained as single-task baselines (one task on, others off), as naive weighted sums (uniform and
grid-searched weights), and as the joint model, and the per-task metrics are compared. The shared
backbone is a deep convolutional encoder–decoder (ResNet-style encoder with an atrous spatial
pyramid pooling context module, separate small decoders per task) trained with SGD with momentum and
a polynomial-decayed learning rate. A parallel, smaller-scale setting is a synthetic regression
problem with two Gaussian outputs whose true noise levels are known, used to check whether an
automatic combination rule responds to known task-scale differences.

## Code framework

What already exists is the standard shared-backbone multi-task training harness. A two-head (more
generally `K`-head) model produces one output per task from a shared encoder; each head has its own
loss `L_i` computed per minibatch (cross-entropy for the classification head, an L-norm for the
regression heads), reduced to a scalar. A combination module is handed those per-task scalar losses
and must return a single scalar for `backward()`. The optimiser owns the model parameters *and* any
parameters the combination module registers — so if the combination rule keeps trainable state, that
state is trained jointly with the network by the same SGD step; this hook exists already. The outer
loop draws a minibatch, runs the shared model, computes each head's loss, calls the combination
module to get the total, backpropagates, and steps.

The only thing not settled is the combination rule itself — how the per-task losses are merged into
one scalar, and what trainable state (if any) that rule maintains. That is the single empty slot.

```python
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Combines several per-task scalar losses into a single training scalar.

    Receives the already-computed per-task losses (and the training-progress
    counters) each step and returns one scalar to call .backward() on. Any
    nn.Parameter registered here is added to the optimizer and trained jointly
    with the network, exactly like a model weight."""

    def __init__(self, num_tasks=2):
        super().__init__()
        # TODO: any trainable state the combination rule we design will need.
        pass

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        # fine_loss, coarse_loss : scalar tensors, one per task head
        # epoch, total_epochs    : training-progress counters (available if useful)
        # TODO: the combination rule we will design — merge the per-task
        #       losses into one scalar and return it.
        pass


# existing shared-backbone multi-task training loop the module plugs into
def train(model, mtl_loss, data_loader, optimizer, total_epochs):
    for epoch in range(total_epochs):
        for inputs, fine_targets, coarse_targets in data_loader:
            optimizer.zero_grad()
            fine_logits, coarse_logits = model(inputs)         # shared encoder, two heads
            fine_loss = nn.functional.cross_entropy(fine_logits, fine_targets)
            coarse_loss = nn.functional.cross_entropy(coarse_logits, coarse_targets)
            total = mtl_loss(fine_loss, coarse_loss, epoch, total_epochs)
            total.backward()                                   # trains model AND mtl_loss params
            optimizer.step()
```

The per-task losses arrive already reduced; `MultiTaskLoss.forward` is where the combination rule
will live, and `__init__` is where any trainable state it needs is registered so the optimiser picks
it up.
