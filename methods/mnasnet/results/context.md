# Context: efficient CNNs for mobile and architecture search (circa 2018)

## Research question

Deploying convolutional networks on phones requires balancing accuracy against the
constraints of a battery-powered, memory-constrained device with tight latency budgets.
Hand-designing such a model means tuning many architectural knobs — kernel sizes, channel
widths, which cheap operation to use where, where to add skip connections — across a huge
design space. Neural architecture search can automate the design, but applying it to the
mobile setting raises questions about what objective to optimize and how to define the
search space. The broad question is: how to use automated search to find convolutional
networks that are both accurate and fast on real mobile hardware?

## Background

- **FLOPS and real latency.** Multiply-add count is the usual stand-in for "speed." Real
  latency depends on memory-access patterns, kernel implementations, parallelism, and other
  hardware/software idiosyncrasies, so two models with nearly equal FLOPS can have very
  different wall-clock latency on the same phone (e.g. ~575M vs ~564M multiply-adds, yet
  113ms vs 183ms on the same device).

- **The cost structure of a layer.** For a depthwise-separable convolution with kernel
  `(K, K, M, N)` mapping an `(H, W, M)` input to `(H, W, N)`, the multiply-adds are
  `H·W·M·(K·K + N)`. At fixed compute there is a trade between kernel size `K` and output
  width `N`, and — because `H·W` shrinks with depth — early (high-resolution) layers cost
  far more per unit of `K·K + N` than late layers.

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
linear bottlenecks, state-of-the-art among mobile-size models).

**Cell-based RL/evolutionary NAS (NASNet, AmoebaNet, PNASNet).** Search a convolutional
cell (+ reduction cell) on a small proxy task (CIFAR-10), stack identical copies, transfer
to ImageNet. Differentiable search (DARTS) follows the same cell-stacking paradigm.

**Multi-objective NAS on proxy tasks (MONAS, DPP-Net, Pareto-NASH, RNAS).** These put
multiple objectives (e.g. model size + accuracy) into the search, optimizing proxy
efficiency metrics such as parameter count or FLOPS on small tasks like CIFAR.

**Compression/pruning (quantization, filter pruning, NetAdapt).** Shrink a given baseline
model post hoc, sometimes using platform-aware metrics.

## Evaluation settings

- **ImageNet** classification (ILSVRC 2012): ~1.28M training images, 1000 classes; metrics
  top-1 / top-5 accuracy on the validation set. The natural large-scale target task.
- **Real on-device latency**: inference time measured by running the model on an actual
  phone (single large CPU core, batch size 1). A target latency `T` (e.g. ~75ms, matching
  an existing mobile model) defines the operating point.
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
existing reward is accuracy alone) and **what space the controller samples over** (the
existing space stacks identical cells). Everything method-specific is left empty.

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

    In the existing harness this is just accuracy."""
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
