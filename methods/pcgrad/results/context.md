## Research question

We want one network, with shared parameters `theta`, to learn several tasks at once. The hope
is that structure shared across tasks makes joint learning more efficient than training each
task from scratch. The standard way to do this is to put the per-task losses together into a
single objective and descend it: `L(theta) = sum_i L_i(theta)`, with the combined gradient
`g = sum_i g_i` where `g_i = grad L_i(theta)`. Yet in practice this often *fails to deliver* —
joint training frequently reaches *worse* final performance and data efficiency than training
the tasks separately, to the point that several multi-task pipelines deliberately train tasks
independently first and only then distill the independent models into one network, paying away
the efficiency the shared model was supposed to buy. The precise problem is to understand *why*
naive joint optimization underperforms, and to find a way to combine the per-task gradient
signals so that the auxiliary tasks help the primary objective instead of degrading it —
ideally a fix that is cheap, makes no assumption about the network architecture, and drops in
front of whatever base optimizer (SGD with momentum, Adam) is already in use.

## Background

Multi-task learning has a long history: the premise (Caruana, *Multitask Learning*, 1997) is
that an inductive bias from related tasks, sharing a representation, improves generalization
and sample efficiency. The dominant modern instantiation is **hard parameter sharing** — a
shared trunk with small task-specific heads — trained by gradient descent on the summed loss.
The promise is real but so is a stubborn optimization difficulty: jointly learning many tasks
is empirically harder than learning any one of them, and the reasons are not well understood.

Several diagnostic explanations have been floated in prior work. Tasks are observed to learn at
**different speeds**, so a faster task can dominate training before a slower one gets going;
the loss landscape is observed to have **plateaus** where joint progress stalls; and much
attention has gone into **architecture** (how much to share, where to branch). These are
observations about *existing* joint-training systems.

A useful lens for thinking about what the combined gradient is doing: the per-task gradients
`g_i` are vectors in the same parameter space, and the optimizer steps along their sum. Two
geometric quantities then matter for how that sum behaves. The first is the **angle** between
two task gradients, summarized by their cosine similarity `cos phi_ij = (g_i . g_j) /
(||g_i|| ||g_j||)`; a negative value means the two gradients point partly *against* each other,
so their sum is shorter than either and the overlap region of the descent is cancelled. The
second is the **relative magnitude**: if one `||g_i||` is far larger than another, the sum
`g = sum_i g_i` is essentially the large gradient, and the small task contributes almost
nothing to the step. A natural scalar for this is the gradient magnitude similarity
`Phi(g_i, g_j) = 2 ||g_i|| ||g_j|| / (||g_i||^2 + ||g_j||^2)`, which equals 1 when the two
magnitudes match and falls toward 0 as they diverge.

Curvature is the third piece of the picture. A first-order step trusts a local linear model of
the loss; in a region of **high positive curvature** that linear model is wrong in a specific,
asymmetric way. Along a direction where the loss curves up sharply, a gradient step
*overestimates* the improvement it will actually realize on the task whose gradient dominates
the step, and a step taken mostly along one task's direction *underestimates* the damage it
does to a task whose gradient it is moving against. A compact way to measure curvature in the
direction the optimizer is actually moving is the path-averaged quantity
`H(L; theta, theta') = integral_0^1 grad L(theta)^T grad^2 L(theta + a(theta'-theta)) grad L(theta) da`,
the curvature of `L` between the current and next iterate along the multi-task gradient. That
deep neural-network loss valleys are narrow and highly curved is an observed property of these
landscapes (Goodfellow et al. 2014, on qualitatively characterizing neural-net optimization).

A simple two-dimensional picture sharpens the intuition and is fully knowable before any fix.
Take two task objectives that are each a deep, curved valley — e.g. `L_1` and `L_2` of the form
`c log(max(|a theta_1 +/- tanh(theta_2) + b|, eps))` — whose summed objective has its optima
where the two valleys meet. Run a strong adaptive optimizer (Adam) on the sum from a fixed
start. It descends into one valley and then **stalls**, unable to traverse to where the valleys
meet, precisely at a point where the two task gradients conflict (negative cosine), differ
greatly in magnitude (one dominates), and sit in high curvature. This is a diagnostic
observation about the *plain* optimizer's behavior — a property of the landscape and the
averaged-gradient update, present before any new method exists.

## Baselines

These are the prior approaches a new method for combining task gradients would be compared to
and would react to.

**Plain summed-loss joint training (the default; Caruana 1997 for the MTL premise).** Minimize
`L(theta) = sum_i L_i(theta)` by stepping along `g = sum_i g_i` with the usual optimizer. Core
idea: let the shared representation absorb common structure. *Observed limitation:* on many
task sets it reaches worse final accuracy and efficiency than separate training; where task
gradients oppose each other the summed gradient cancels in the overlap, and where one task's
gradient is much larger it swamps the others, so the shared trunk is pulled almost entirely by
the dominant task.

**Uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018).** Replace the equal-weight sum
with a learned weighted sum derived from each task's homoscedastic (data-independent)
uncertainty: minimize `sum_i (1/(2 sigma_i^2)) L_i + log sigma_i`, learning a per-task variance
`sigma_i^2` jointly with the network (a log-variance `s_i = log sigma_i^2` is the stable
parameterization, with the `log sigma_i` term preventing the trivial `sigma_i -> infinity`
solution). Core idea: noisier / harder tasks should contribute less to the loss; the weights
are learned rather than grid-searched. *Observed limitation:* it only rescales the *magnitude*
of each task's contribution to a scalar sum. Every per-task gradient still enters the step
pointing in its original direction; a weighting cannot cancel a direction that opposes another
task. When two gradients conflict, reweighting changes only how much each conflicting vector
counts, not the fact that they conflict.

**GradNorm (Chen et al., ICML 2018).** Adapt the loss weights `w_i` during training so that the
*weighted* per-task gradient norms grow at similar rates, balancing how fast each task trains.
Core idea: equalize training dynamics across tasks by normalizing gradient magnitudes.
*Observed limitation:* like uncertainty weighting, it acts on magnitudes only — it tunes
`w_i ||g_i||` but never touches the relative *directions* of the `g_i`, so directional conflict
survives untouched.

**Multi-task as multi-objective optimization / MGDA (Sener & Koltun, NeurIPS 2018; Désidéri
2012).** Treat the tasks as a vector objective and seek a common descent direction by finding
the minimum-norm point in the convex hull of the task gradients: solve
`min_{alpha} || sum_t alpha_t g_t ||^2` subject to `sum_t alpha_t = 1, alpha_t >= 0`, then step
along `sum_t alpha_t g_t`. Désidéri showed the solution is either zero (a Pareto-stationary
point) or a direction that decreases every task. For two tasks there is a one-dimensional
analytic `alpha`; for more tasks it is a constrained quadratic program, solved at scale via
Frank-Wolfe. Core idea: a single update that does not increase any task's loss. *Observed
limitation:* it must solve a constrained QP over the simplex each step; its update is a *convex
combination* of the task gradients (a reweighting inside the simplex), so it down-weights
conflicting tasks rather than excising the conflicting component; and at a Pareto-stationary
point the min-norm direction can collapse to near zero, stalling progress while individual
tasks could still improve.

**Gradient projection for continual learning — GEM (Lopez-Paz & Ranzato, NeurIPS 2017) and
A-GEM (Chaudhry et al., 2019).** In *sequential* learning, protect already-learned tasks: take
the proposed gradient `g` for the current task and project it to the nearest vector `g~` that
does not increase any past task's loss, `min ||g - g~||^2` subject to `<g~, g_k> >= 0` for all
past `k` — a QP solved in its dual. A-GEM cheapens this to a single inner product by projecting
`g` onto only one *reference* (averaged past) gradient when the angle is obtuse. Core idea: use
the *sign of the inner product* between gradients to detect when an update would hurt another
task, and project to remove the harmful part. *Observed limitation:* these target the sequential
(continual) setting — they protect *past* tasks from the *current* one, so they project only the
single current-task gradient and treat the others asymmetrically; GEM still pays a per-step QP.
They are not built for learning all tasks *simultaneously*, where there is no privileged current
task and the task losses interact within the same update.

**Cosine regularization (CosReg; Suteu & Guo 2019).** Add a penalty that drives the cosine
similarity between different tasks' gradients toward 0. Core idea: discourage interference by
encouraging orthogonality. *Observed limitation:* it pushes gradients toward orthogonality
*unconditionally*, so it suppresses cooperative (positively-aligned) gradients too, throwing
away the very positive transfer that motivated sharing in the first place. (A related line,
Du et al. 2018, uses the cosine sign merely to decide whether to *keep or drop* an auxiliary
task wholesale — a binary choice that discards the task's useful component along with its
harmful one.)

## Evaluation settings

The natural yardsticks already in use for multi-task learning, as pre-existing datasets,
metrics, and protocols:

- **Multi-task CIFAR-100**: the 100 fine classes grouped into superclasses give a natural
  multi-task / multi-head classification setup on a shared convolutional backbone; per-task
  classification accuracy is the metric. Backbones in use include small convolutional networks
  (e.g. a few `3x3`-conv layers plus fully-connected heads) and standard residual networks.
- **Multi-task scene understanding** on **NYUv2** and **CityScapes**: a shared encoder-decoder
  (SegNet-style) jointly predicting semantic segmentation, depth, and (for NYUv2) surface
  normals; metrics are mean IoU and pixel accuracy for segmentation, absolute/relative error
  for depth, and angular error for normals. Architectures compared include split networks,
  cross-stitch networks (Misra et al. 2016), and attention-based MTAN (Liu et al. 2018).
- **MultiMNIST**: two overlaid digits classified by a left head and a right head — a clean
  two-task setup following Sener & Koltun's protocol.
- **Multi-task reinforcement learning** on the **Meta-World** MT10 / MT50 benchmarks (Yu et al.
  2019): 10 or 50 simulated Sawyer manipulation tasks under a shared policy/critic trained with
  soft actor-critic (Haarnoja et al. 2018); average success rate across tasks is the metric.
  Goal-conditioned pushing is the continuous-task-distribution variant.
- Protocol: a shared trunk with per-task heads or a task-conditioned model `f_theta(y|x, z_i)`
  with a one-hot task code `z_i`; identical backbone across the combination strategies being
  compared, so differences are attributable to how the per-task gradients are combined.

## Code framework

A standard multi-task training step already has a shared-trunk model with per-task heads,
per-task losses, a base optimizer (SGD-momentum or Adam) that consumes one gradient per
parameter, and a loop that, per step, computes each task's loss, backpropagates once per task,
records a flat gradient vector and a mask for which parameters that task touched, hands one
merged gradient back to the optimizer, and steps. The only open slot is the generic rule that
takes the per-task gradient vectors plus their masks and produces the merged gradient.

```python
import torch


def flatten_tensors(tensors):
    return torch.cat([x.flatten() for x in tensors])


def get_parameter_grads(optimizer):
    grads, shapes, has_grads = [], [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            shapes.append(p.shape)
            if p.grad is None:
                grads.append(torch.zeros_like(p))
                has_grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.clone())
                has_grads.append(torch.ones_like(p))
    return flatten_tensors(grads), shapes, flatten_tensors(has_grads)


def compute_task_gradients(losses, optimizer):
    """Return one flat gradient vector and one touched-parameter mask per task."""
    grads, masks, shapes = [], [], None
    for L_i in losses:
        optimizer.zero_grad(set_to_none=True)
        L_i.backward(retain_graph=True)
        flat, shapes, mask = get_parameter_grads(optimizer)
        grads.append(flat)
        masks.append(mask)
    return grads, masks, shapes


def combine_task_gradients(grads, masks):
    """Combine the per-task gradient vectors into the single gradient to apply.

    grads: list of flat per-task gradients [g_1, ..., g_n], all the same length.
    masks: list of flat 0/1 vectors marking which parameters each task touched.
    returns: one flat gradient vector of that length.
    """
    # TODO: define the combination rule.
    pass


def unflatten_like(flat_grad, shapes):
    pieces, offset = [], 0
    for shape in shapes:
        length = int(torch.tensor(shape).prod().item())
        pieces.append(flat_grad[offset:offset + length].view(shape).clone())
        offset += length
    return pieces


def set_parameter_grads(optimizer, grads):
    idx = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = grads[idx]
            idx += 1


def train_step(model, losses_fn, batch, optimizer):
    losses = losses_fn(model, batch)          # a per-task loss, e.g. [L_fine, L_coarse]
    grads, masks, shapes = compute_task_gradients(losses, optimizer)
    combined = combine_task_gradients(grads, masks)
    set_parameter_grads(optimizer, unflatten_like(combined, shapes))
    optimizer.step()                          # SGD-momentum / Adam, unchanged
```

The base optimizer, the per-task losses, and the gradient bookkeeping are all in place;
`combine_task_gradients` is the only empty function.
