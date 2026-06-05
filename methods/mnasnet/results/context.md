# Context: efficient CNNs for mobile and architecture search (circa 2018)

## Research question

Deploying convolutional networks on phones is a tug-of-war: a model must be **accurate**,
yet also **small and fast enough** to run on a battery-powered, memory-constrained device
with tight latency budgets. Hand-designing such a model means balancing many architectural
knobs — kernel sizes, channel widths, which cheap operation to use where, where to add skip
connections — across a huge design space, by hand, by experts. Neural architecture search
can automate the design, but as it stands it has two problems for the mobile setting.
First, it optimizes a *single* objective (accuracy) and treats efficiency only indirectly:
when efficiency is considered at all, it is through a *proxy* like FLOPS/multiply-adds,
which is not what actually determines whether a model is fast on a given phone. Second, to
keep the search tractable, it searches for one or a few *cells* and then stacks identical
copies of them throughout the network — which forbids the network from using *different*
structure at different depths, exactly the layer diversity that matters for the
accuracy-vs-latency trade-off.

The precise goal: an automated search that (1) optimizes accuracy and **real, measured**
inference latency on the actual target device *jointly*, returning models on or near the
accuracy-latency Pareto front; and (2) searches over a space rich enough to let different
parts of the network differ, while staying small enough to be searchable.

## Background

- **The FLOPS-is-a-bad-proxy observation (motivating diagnostic).** Multiply-add count is
  the usual stand-in for "speed," but it is a poor predictor of real latency: two models
  with nearly equal FLOPS can have very different wall-clock latency on the same phone
  (e.g. ~575M vs ~564M multiply-adds, yet 113ms vs 183ms on the same device), because real
  latency depends on memory-access patterns, kernel implementations, parallelism, and other
  hardware/software idiosyncrasies that FLOPS ignores. This is a pre-method fact about the
  world and it motivates measuring latency directly on-device rather than approximating it.

- **The cost structure of a layer.** For a depthwise-separable convolution with kernel
  `(K, K, M, N)` mapping an `(H, W, M)` input to `(H, W, N)`, the multiply-adds are
  `H·W·M·(K·K + N)`. So at fixed compute there is a real trade between kernel size `K` and
  output width `N`, and — because `H·W` shrinks with depth — early (high-resolution) layers
  cost far more per unit of `K·K + N` than late layers. Different layers therefore *want*
  different choices, which is the argument for layer diversity.

- **Mobile building blocks.** Depthwise-separable convolution (a per-channel spatial filter
  + a 1×1 channel-mixing convolution) drastically cuts compute versus a full convolution.
  The **inverted residual with linear bottleneck** (MobileNetV2) expands a thin bottleneck
  to a wide intermediate representation, applies a depthwise convolution there, projects
  back down linearly, and adds a residual between the thin bottlenecks — the dominant
  efficient mobile block. **Squeeze-and-excitation** adds a cheap channel-attention
  recalibration (global-pool → small MLP → per-channel gate). Pooling and plain/grouped
  convolutions round out the menu.

- **RL-based architecture search.** A recurrent (RNN/LSTM) controller with parameters θ
  autoregressively emits a sequence of tokens `a_{1:T}` describing an architecture; each
  sampled model is built, trained, and scored, and θ is updated to raise the expected
  reward `J = E_{P(a_{1:T};θ)}[R(m)]`. Updating θ from sampled, non-differentiable rewards
  is a policy-gradient problem; **Proximal Policy Optimization** (PPO) is a stable
  policy-gradient method that constrains each update via a clipped surrogate so the new
  policy stays close to the old one. The prevailing search spaces are cell-based: search a
  cell, stack identical copies.

- **Multi-objective optimization and Pareto optimality.** With two objectives (accuracy,
  latency) there is generally no single best model but a *Pareto front* — models where you
  cannot improve one objective without worsening the other. Standard scalarizations combine
  objectives into one scalar: a *weighted sum* or a *weighted product*. The choice of
  scalarization shapes which Pareto-optimal points the search is pushed toward.

## Baselines

**Hand-crafted mobile CNNs.** SqueezeNet (1×1 convs, fewer filters); MobileNetV1 (depthwise
separable convolutions throughout); ShuffleNet (grouped convs + channel shuffle);
CondenseNet (learned grouped-conv connectivity); **MobileNetV2** (inverted residuals with
linear bottlenecks, state-of-the-art among mobile-size models). **Gap:** each is the
product of large manual effort exploring a vast space by hand; they don't *learn* novel
operation compositions, and tuning them per latency target is more manual work.

**Cell-based RL/evolutionary NAS (NASNet, AmoebaNet, PNASNet).** Search a convolutional
cell (+ reduction cell) on a small proxy task (CIFAR-10), stack identical copies, transfer
to ImageNet. Strong accuracy. **Gap (two):** (1) the objective is accuracy only — latency
is not in the loop, so the resulting mobile-size models can be slow (e.g. NASNet-A at 183ms
on a phone); and (2) stacking *identical* cells forbids layer diversity, removing exactly
the freedom that the per-layer cost analysis says is valuable for efficiency. Differentiable
search (DARTS) shares both gaps.

**Multi-objective NAS on proxy tasks (MONAS, DPP-Net, Pareto-NASH, RNAS).** These do put
multiple objectives (e.g. model size + accuracy) into the search. **Gap:** they optimize
*proxy* efficiency metrics (params/FLOPS, not measured device latency) and search on small
tasks like CIFAR, not directly under real mobile-latency constraints on a large task.

**Compression/pruning (quantization, filter pruning, NetAdapt).** Shrink a *given* baseline
model post hoc, sometimes using platform-aware metrics. **Gap:** tied to a fixed baseline
architecture; they don't discover new compositions of operations.

## Evaluation settings

- **ImageNet** classification (ILSVRC 2012): ~1.28M training images, 1000 classes; metrics
  top-1 / top-5 accuracy on the validation set. The natural large-scale target task.
- **Real on-device latency**: inference time measured by *running the model on an actual
  phone* (single large CPU core, batch size 1) — the direct efficiency metric, replacing
  the FLOPS proxy. A target latency `T` (e.g. ~75ms, matching an existing mobile model)
  defines the operating point.
- **COCO** object detection: the searched backbone plugged into the SSD/SSDLite detector;
  metric mean Average Precision (mAP). Input size 320×320 in the mobile detection setting.
- Standard ImageNet training recipe of the time: RMSProp (decay 0.9, momentum 0.9),
  batch-norm after every conv (momentum 0.99), weight decay 1e-5, dropout 0.2 on the final
  layer, learning-rate warmup then exponential decay, Inception-style preprocessing at
  224×224. Model scaling is studied via depth (width) multiplier and input resolution, as
  is standard for mobile models.

## Code framework

The search machinery already exists as a sample-eval-update loop: an RNN controller emits a
token sequence describing a model, a trainer turns the model into an accuracy, and the
controller's parameters are nudged toward higher reward by a policy-gradient update. Two
things are unspecified and are the slots to fill: **what reward the loop optimizes** (the
existing reward is accuracy alone, evaluated however the trainer measures it) and **what
space the controller samples over** (the existing space stacks identical cells). Everything
method-specific is left empty.

```python
import torch
import torch.nn as nn


class Controller(nn.Module):
    """RNN that autoregressively emits a token sequence a_{1:T} describing a model,
    and supports a policy-gradient update on its parameters theta."""

    def __init__(self, search_space, hidden=...):
        super().__init__()
        # LSTM + per-decision softmax heads over the choices in `search_space`
        # TODO: the structure of the search space the controller samples over
        pass

    def sample(self):
        # TODO: emit tokens a_{1:T}, return the decoded model spec + log-probs
        pass


def build_model(spec):
    """Realize a sampled token sequence as a runnable CNN."""
    # TODO: how a spec maps to a concrete network (the search space)
    pass


def train_and_eval_accuracy(model, train_loader, valid_loader):
    """Train the candidate and return its accuracy ACC(m)."""
    # TODO
    pass


def reward(model):
    """The scalar the controller maximizes for each sampled model.

    In the existing harness this is just accuracy. What it SHOULD be when we
    also care about how fast the model runs on a real device is the open slot."""
    acc = train_and_eval_accuracy(model, ...)
    # TODO: combine acc with whatever else we care about
    return acc


def search(controller, opt_theta):
    """Sample-eval-update loop: sample models, score them, raise expected reward
    by a stable policy-gradient step on theta."""
    while not done:
        spec, log_prob = controller.sample()
        model = build_model(spec)
        r = reward(model)
        # TODO: policy-gradient update maximizing E[r] (constrain the step for stability)
        pass
```
