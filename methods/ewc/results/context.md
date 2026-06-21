## Research question

A neural network with a fixed set of weights `theta` is shown a *sequence* of tasks — task A,
then task B, then C, ... — and after it has moved on from a task it gets no further access to
that task's data. The goal is a single network that keeps performing the earlier tasks while
it learns the later ones. When standard stochastic gradient descent is trained on task B after
task A, the gradient on B moves the weights that encoded A and performance on A drops — the
phenomenon known as *catastrophic forgetting*.

The setting fixes the operating regime. The network has *fixed capacity*: no new parameters are
added per task. The old data is not retained and interleaved during later training. And the
training cost stays linear in the number of parameters and in the number of training examples,
so the method runs on the same large deep networks used elsewhere. The question is how to carry
a compact record of each finished task forward into ordinary training on the next one.

## Background

Catastrophic forgetting (also called catastrophic interference) is an old observation about
connectionist models: McCloskey & Cohen (1989) and Ratcliff (1990) showed that training a
network on a second task in sequence rapidly destroys what it learned on the first, and French
(1999) framed this as the *stability-plasticity dilemma* — a network plastic enough to learn B
is, for that very reason, unstable enough to forget A.

A diagnostic empirical study by Goodfellow, Mirza, Xiao, Courville & Bengio (2014) measured
this directly across activation functions and training algorithms. They trained on a first
task, then a second, and tracked accuracy on *both*, under three regimes of task relationship:
tasks that are functionally identical but presented in a different input format, tasks that are
similar, and tasks that are dissimilar. The "identical task, different input format" regime is
constructed by taking a dataset and applying a *fixed random permutation to the input pixels* —
each permutation is a new task of the same intrinsic difficulty but requiring different weights.
Two of their findings: the amount and character of forgetting depends strongly on task
similarity, so a sequential learner is tested across that axis rather than on a single pair; and
among the methods of the day, dropout was the most consistently effective at resisting
forgetting, with the best dropout networks being *larger*.

There is also a biological parallel. In mammalian neocortex, when a skill is learned a fraction
of excitatory synapses are strengthened (enlarged dendritic spines), and those particular spines
*persist* through later learning, accounting for retained performance months later; selectively
erasing them erases the skill. The reading is that the brain protects old knowledge not by
walling off whole regions but by rendering *specific* synapses less plastic — task-specific
synaptic consolidation.

Two further pieces of standard machinery are on the table. The **Laplace approximation**
(MacKay 1992, in the Bayesian-backprop framework): training a network to a (local) optimum can
be read as finding the mode of a posterior `p(theta | D)`; around that mode the negative log
posterior is, to second order, a quadratic, so the posterior is locally a Gaussian with mean at
the optimum and precision equal to the Hessian of the negative log posterior there. This turns
"the distribution over good weights for a task" into a tractable Gaussian summary. And the
**Fisher information matrix** `F` (Pascanu & Bengio 2013, in the natural-gradient setting), with
three properties that are repeatedly useful: (a) under the model distribution and the usual
regularity conditions, it equals the expected Hessian of the negative log-likelihood; for
`p = p_theta(y|x)`, the expectation of the `grad_theta^2 p` term becomes
`grad_theta^2 sum_y p_theta(y|x) = grad_theta^2 1 = 0`, leaving the identity
`E[-grad_theta^2 log p] = E[(grad_theta log p)(grad_theta log p)^T]`; (b) because of that outer-product form it can be
computed from *first-order* derivatives alone, so it is feasible even for very large models; and
(c) it is positive semi-definite by construction, unlike the raw Hessian of the empirical loss,
which may be indefinite.

## Baselines

The prior approaches a sequential-learning method is measured against:

**Plain SGD (sequential fine-tuning).** Train on A to `theta*_A`, then continue training on B
with the ordinary task-B loss. The B-gradient has no term referring to task A; every weight is
equally free to move.

**Multitask / joint training, and replay-based system-level consolidation (McClelland et al.
1995).** If all tasks' data are available at once, interleaving them jointly optimizes the
weights for every task. When tasks arrive sequentially this uses an episodic memory that *stores
and replays* old examples during later training, so the stored-and-replayed data grows with the
number of tasks.

**Uniform quadratic anchoring (plain L2 to the old weights).** Add `sum_i (lambda/2)(theta_i -
theta*_{A,i})^2` to the task-B loss — pull every weight back toward its task-A value with the
*same* stiffness. The stiffness is parameter-independent, set by the single global constant
`lambda`.

**Dropout regularization (Goodfellow et al. 2014; Hinton et al. 2012).** The strongest of the
standard regularizers at resisting forgetting in the diagnostic study, by training-with-dropout
plus early stopping and cross-validated hyperparameters.

**Quadratic energy-surface penalties on small models (French & Chater 2002; Eaton & Ruvolo
2013, ELLA).** These approximate the old-task loss surface by a quadratic penalty so that old
data can be discarded. French & Chater used random inputs to build a quadratic approximation to
the error surface, recomputing the curvature at each sample; ELLA maintains a per-task model with
a shared latent basis, computing and inverting matrices whose dimensionality is the number of
parameters, applied in practice to linear and logistic regression.

## Evaluation settings

The natural yardsticks for a sequential-learning method, all of which predate it:

- **Permuted MNIST.** Take the MNIST handwritten-digit dataset (LeCun et al. 1998; 28x28
  images, 10 classes) and generate a sequence of tasks, each defined by a fixed random
  permutation of the input pixels (Srivastava et al. 2013; Goodfellow et al. 2014). Every task
  is of equal difficulty to plain MNIST but needs a different weight configuration. Train on
  each task for a fixed number of epochs in the usual shuffled-minibatch way, then move on with
  no further access to that task's data. Protocol detail used as a strong-baseline control: for
  the dropout baseline, dropout probability 0.2 on the input and 0.5 on hidden layers, with
  early stopping on a validation set pooled over all permutations seen so far.
- Network for the MNIST tasks: a fully connected multilayer perceptron of rectified-linear
  units (e.g. two hidden layers of 400 units; or deeper/wider for diagnostic ablations).
  Learning rate `1e-3`, ~20 epochs per task.
- **Class-split image classification.** Split a labelled image dataset into disjoint groups of
  classes, each group a task presented in sequence (e.g. MNIST digits into binary tasks, or a
  100-class dataset into ten 10-way tasks) — task-incremental evaluation.
- **Sequential reinforcement learning on Atari 2600** (Bellemare et al. 2013, the Arcade
  Learning Environment), with a Deep Q-Network agent (Mnih et al. 2015; double Q-learning, van
  Hasselt et al. 2016): a set of ten games played in sequence, returning to earlier ones,
  scored by human-normalized return summed across the ten games.
- Primary metric throughout: **average performance across all tasks after the full sequence is
  trained** (accuracy for classification, summed human-normalized score for Atari) — higher is
  better. Hyperparameters chosen by cross-validated random search.

## Code framework

A fixed continual-learning training harness exposes two hooks. The outer loop trains the
network on each context with the ordinary shuffled-minibatch routine. One hook is called once
after a context finishes; it may run forward/backward passes over that context's data and must
return a compact stored record. The other hook is called every training step and returns a scalar
added to the current task loss. The substrate is only the generic machinery already available: a
model exposing its named parameters, a per-context data loader, a place to store compact records
between contexts, and a training loop that adds the returned scalar to the per-step loss.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def summarize_finished_context(model, dataset, device):
    """Called ONCE after training on a context finishes.

    May run forward/backward passes over `dataset`. Returns a compact
    stored record for the just-finished context.
    """
    # TODO: the compact record we will define.
    pass


def extra_training_loss(model, summary_state, strength=1.0):
    """Called at EVERY training step; must be cheap.

    Returns a scalar tensor added to the task loss.
    """
    # TODO: the scalar term this hook will return.
    pass


# existing continual-learning training loop the two hooks plug into
def train_on_context(model, loss_fn, data_loader, optimizer,
                     summary_state, device, regularizer_strength=1.0):
    model.train()
    for inputs, targets in data_loader:           # shuffled minibatches, as usual
        optimizer.zero_grad()
        outputs = model(inputs)                    # forward through the existing model
        task_loss = loss_fn(outputs, targets)      # ordinary task loss for this context
        extra_loss = extra_training_loss(model, summary_state, regularizer_strength)
        (task_loss + extra_loss).backward()
        optimizer.step()
    # after the context: summarize it for storage before the next context
    new_summary = summarize_finished_context(model, data_loader.dataset, device)
    return new_summary
```
