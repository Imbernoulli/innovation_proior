## Research question

A network is trained not on one dataset but on a *sequence* of tasks `μ = 1, 2, 3, …`,
arriving one at a time. While training on task `μ` the learner sees only that task's loss
`L_μ`; the data and loss functions of all earlier tasks `ν < μ` are gone. The goal is the
total loss over everything ever seen, `L = Σ_μ L_μ`, but we can only ever descend one summand
at a time. The structural problem is that ordinary gradient descent on `L_μ` moves the shared
weights wherever lowering `L_μ` demands, and those same weights encoded the earlier tasks, so
the earlier losses `L_ν` inadvertently shoot back up. After a handful of tasks the network
solves only the most recent one and has reverted to chance on the first — the network forgets
catastrophically.

A solution has to keep the network *plastic* enough to learn each new task well, yet *stable*
enough that learning a new task does not destroy the old ones — the stability-plasticity
dilemma. Concretely it must answer two coupled questions. First, *which* parameters were
important for the tasks already learned, so that those in particular should be held in place?
Second, *how* should that importance be turned into a penalty added to the per-step loss, so
that future training flows preferentially through the parameters that did not matter before?
The practical bar set by the deployment setting is steep: whatever measures importance must be
cheap (no extra backward passes per datapoint), must use only quantities available *during*
ordinary training, must cost constant memory as the number of tasks grows, and must be
computable locally, per parameter, from the optimization itself.

## Background

Catastrophic forgetting (or catastrophic interference) was documented for connectionist
networks by McCloskey & Cohen (1989) and Ratcliff (1990): a backprop network trained on a
second set of associations after a first abruptly loses the first, even though the two could
in principle coexist in the same weights. French (1999) traced the mechanism to the *shared,
distributed* representation — when many tasks are encoded in overlapping weights, any weight
change that helps a new task tends to corrupt the old one, and the more new learning, the more
the old is overwritten. The standard cure, retraining on all data jointly (the multitask
regime), is exactly what the continual setting forbids: the old data is not available, and
replaying it would cost memory proportional to the number of tasks (Goodfellow et al. 2013;
Choy et al. 2006).

Several framings of the loss geometry are available and load-bearing for any importance-based
fix. A deep network is heavily over-parameterized, so a task does not pin down a single
solution but a whole low-loss manifold (Sussmann 1992): there are typically many weight
configurations achieving the same performance on task A. That over-parameterization is the
source of hope — after learning A there may well be a nearby configuration that *also* solves
B, if only one can be steered toward it instead of toward an arbitrary B-minimum. Near a
converged point `θ*` of a task, a first-order Taylor expansion of the loss in a small
parameter move is `L(θ + δ) − L(θ) ≈ Σ_k g_k δ_k` with `g_k = ∂L/∂θ_k`; and a second-order
expansion `L(θ) ≈ L(θ*) + ½ (θ−θ*)ᵀ H (θ−θ*)` describes the local bowl by its Hessian `H`,
whose curvature along a direction says how much loss is paid for moving the weights that way.
Stiff (high-curvature) directions are exactly the ones a past task cared about; flat
(low-curvature) directions are free to move. Empirically the relevant Hessian in these
problems is *low rank* — most directions in weight space are flat for any given task — which is
precisely what leaves room for later tasks to be solved without disturbing earlier ones. The
first-order expansion also assigns an immediate coordinate-wise contribution to a small update,
which gives a local language for asking which parameters did work during optimization.

There is also a probabilistic framing. Optimizing a task is finding the most probable
parameters given its data, and after task A all that A taught is, in principle, summarized in
the posterior `p(θ | D_A)`. A Gaussian (Laplace) approximation of that posterior centered at
`θ*_A` has a precision (inverse covariance) given by the curvature there; the diagonal of the
Fisher information matrix is a tractable, positive-semidefinite stand-in for that curvature,
computable from first-order gradients and equal to the second derivative of the loss near a
minimum (MacKay 1992; Pascanu & Bengio 2013). This connects "how important is each parameter"
to "how sharply does the loss rise if I move it," i.e. to a curvature/precision per parameter.

A separate strand notes that biological synapses are not scalar: they carry internal state
across multiple timescales, and a body of neuroscience on cascade memories, synaptic tagging,
and memory consolidation (Fusi et al. 2005; Redondo & Morris 2011; Benna & Fusi 2016) argues
such complexity underlies the brain's ability to consolidate memories — to render the synapses
important for an old skill less plastic while leaving the rest free. This is the suggestive
analogy: give each artificial parameter a small amount of internal state that tracks its own
importance.

## Baselines

**Elastic weight consolidation (Kirkpatrick et al. 2017, PNAS).** The probabilistic framing
turned into an algorithm. After task A converges at `θ*_A`, approximate the task-A posterior as
a Gaussian with mean `θ*_A` and diagonal precision equal to the diagonal Fisher information
`F`, and protect A by adding a quadratic spring that pulls each weight back toward `θ*_{A,i}`
with stiffness `F_i`:

```
L(θ) = L_B(θ) + Σ_i (λ/2) · F_i · (θ_i − θ*_{A,i})^2 .
```

The spring is stiff exactly on the weights A cared about (large `F_i`) and slack on the rest,
so B is learned through the unimportant directions. `F_i = E[(∂ log p(y|x) / ∂θ_i)^2]` is the
diagonal Fisher, positive-semidefinite and equal near a minimum to the curvature. **Gaps:**
the Fisher is a *point* estimate evaluated *at the converged endpoint* `θ*_A`, computed in a
*separate phase after* the task finishes — an extra sweep that is not part of training and that
sees only the final point, not how the network got there. Computing the diagonal Fisher
exactly requires summing over all possible output labels, so its cost grows with the number of
outputs, restricting the method to low-dimensional output spaces. And for a third task one must
either keep a separate Fisher per past task (memory growing with the number of tasks) or fold
them together.

**Functional / distillation regularizers (Li & Hoiem 2016, "Learning without Forgetting"; Jung
et al. 2016).** Instead of constraining the weights, constrain the network's *function*: when
training on new data, keep the new network's outputs (or final hidden activations) close to the
old network's via a distillation penalty (Hinton et al. 2014). **Gap:** every new datapoint
requires a forward pass through a stored copy of the *old* network to produce the target to
match, which is computationally expensive, and it preserves the input-output map only on the
distribution of the *new* data.

**Architectural methods (freezing layers, Razavian et al. 2014; reduced learning rates for
shared layers, Donahue et al. 2014, Yosinski et al. 2014; Progressive Networks, Rusu et al.
2016).** Prevent interference by changing the network rather than the objective — freeze the
weights of solved tasks, or copy the whole network per task and add fresh capacity. **Gap:** the
architecture (and hence memory and compute) grows with the number of tasks, and frozen weights
forfeit any positive transfer to later tasks.

**A single quadratic anchor (uniform-stiffness L2).** The crude regularizer
`Σ_i (θ_i − θ*_{A,i})^2` pulls *every* weight back equally. **Gap:** one global stiffness cannot
be right — large enough to hold A in place is too rigid to learn B; small enough to learn B
fails to hold A. The whole point is that the stiffness must be *per parameter*, set by how much
each weight mattered.

## Evaluation settings

The natural yardsticks for continual classification are the datasets, metrics, and training
protocols already in use:

- **Split MNIST.** The MNIST digits (LeCun et al. 1998) split into 5 binary tasks (0/1, 2/3,
  4/5, 6/7, 8/9), presented in sequence. A small MLP (two hidden layers of 256 ReLU units),
  categorical cross-entropy, minibatch 64, a small number of epochs per task, optimized with
  Adam (`η = 1e-3, β₁ = 0.9, β₂ = 0.999`). A multi-head readout computes the loss only over the
  classes present in the current task, to avoid label-distribution crosstalk at the output.
- **Permuted MNIST.** Each task applies a different fixed random permutation to all input
  pixels (Goodfellow et al. 2013), so the tasks share digit semantics but have uncorrelated
  inputs. A larger MLP (two hidden layers of 2000 ReLU units), softmax, Adam, minibatch 256,
  more epochs per task.
- **Split CIFAR-10 / CIFAR-100.** A CNN (4 convolutional layers then 2 dense layers with
  dropout) trained first on CIFAR-10, then on successive 10-class subsets of CIFAR-100 (10
  classes per task) (Krizhevsky & Hinton 2009), multi-head, Adam, minibatch 256.
- **Protocol and metric.** Train the tasks strictly in sequence; after each task evaluate on all
  tasks seen so far; the summary number is the *average classification accuracy over all
  learned tasks* as the number of tasks grows. Natural controls are a network with no
  consolidation (plain fine-tuning) and a network trained jointly on all data at once.

## Code framework

The regularizer plugs into an otherwise-ordinary minibatch training loop that sweeps the tasks
in sequence and, on each step, adds a regularization penalty to the task loss before
backpropagation. What is *not* settled — and is exactly what must be designed — is the per-
parameter importance and the penalty built from it, so the substrate is only the generic
sequential-training machinery that already exists: a loop over tasks, a per-step update of the
parameters by an existing optimizer, a snapshot of the parameters taken when a task ends to
serve as the reference point the penalty pulls toward, and two empty slots. The first slot is
filled once per task, after it finishes, and returns a per-parameter update to the carried
importance; the second is called every step and returns the scalar penalty added to the loss.

```python
import torch


def train_continual(model, optimizer, tasks, reg_strength, device):
    importance = {}                  # per-parameter importance accumulated over past tasks
    prev_params = {}                 # parameter snapshot from the end of the previous task

    for task in tasks:               # tasks arrive one at a time; only this task's loss is visible
        for inputs, targets in task.loader():
            prev = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer.zero_grad()
            task_loss = model.loss(inputs, targets)
            penalty = compute_regularization_loss(model, importance, prev_params)
            (task_loss + reg_strength * penalty).backward()
            optimizer.step()
            # the loop may record any per-step quantities the importance estimator needs
            record_step(model, prev)

        new_imp = estimate_importance(model, task.dataset, prev_params, device)
        accumulate(importance, new_imp)                       # carry importance forward
        prev_params = {n: p.detach().clone() for n, p in model.named_parameters()}  # new anchor


def estimate_importance(model, dataset, prev_params, device):
    """Called once after a task finishes. Returns {param_name: importance_update_tensor}.
    May use stored per-step quantities or run forward/backward passes over `dataset`."""
    # TODO: the per-parameter importance measure we will design.
    pass


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Called every training step; must be cheap. Returns the scalar penalty added to the loss."""
    # TODO: the penalty built from the importance and the reference parameters.
    pass
```

The outer loop supplies the task stream, the reference snapshot, and a per-step hook; the two
stubs are where the importance measure and its penalty will live.
