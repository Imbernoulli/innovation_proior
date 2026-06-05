# Context: automating neural architecture design (circa 2017-2018)

## Research question

Designing the architecture of a neural network — how many layers, which operations
at each layer, how layers connect, where to place skip connections, which nonlinearity
in a recurrent cell — is still done by hand, by experts, through slow trial and error.
The obvious dream is to *learn* the architecture: pose a search space of candidate
networks and have an algorithm find a good one. The recent reinforcement-learning
formulation makes this concrete — a controller proposes a candidate architecture (a
"child model"), the child is trained and its validation performance is fed back as a
reward, and the controller is nudged toward more promising architectures. It works, and
it produces architectures competitive with the best hand-designed ones on image
classification and language modeling.

But it is brutally expensive. The cost is dominated by one step: to score a single
candidate architecture, you train that child network *from scratch to convergence*, read
off its validation accuracy, and then throw the trained weights away. Every candidate
starts from random initialization and is trained independently. A single search can burn
hundreds of GPUs for days — on the order of tens of thousands of GPU-hours — because the
inner loop is "train a full network to convergence," repeated for thousands of candidates.
Cutting the budget (fewer epochs per child, fewer candidates) reliably produces weaker
architectures. The precise problem: drive the cost of architecture search down by orders
of magnitude — to something a single GPU can run in under a day — without giving up the
quality of the discovered architectures. The bottleneck to attack is the wasteful
"train-each-child-from-scratch-then-discard-its-weights" inner loop.

## Background

The field state rests on a few load-bearing ideas:

- **Policy-gradient / REINFORCE (Williams 1992).** To optimize an expected reward
  `E_{a~π(·;θ)}[R(a)]` where the action `a` (here, a discretely sampled architecture)
  is non-differentiable, the score-function estimator gives
  `∇_θ E[R] = E[ R · ∇_θ log π(a;θ) ]`. The gradient needs only the ability to sample
  actions and evaluate their (black-box) reward; the reward need not be differentiable.
  Its estimator has high variance, standardly reduced by subtracting a baseline `b`:
  `∇_θ E[R] ≈ (R − b) · ∇_θ log π(a;θ)`, with `b` an exponential moving average of recent
  rewards. The baseline does not bias the estimator because `E[ b · ∇_θ log π ] = 0`.

- **Autoregressive controllers.** A candidate architecture is a sequence of discrete
  decisions (which op, which input, which nonlinearity). An RNN/LSTM controller emits this
  sequence token-by-token, feeding the previous decision back as the next input, with a
  softmax over choices at each step. This is the mechanism the RL-based search uses to
  define `π(architecture; θ)`.

- **Cells, skip connections, and the structure of good architectures.** Two structural
  motifs are known to matter. Residual/skip connections (He et al. 2016) let information
  and gradients bypass layers, enabling much deeper nets. And rather than designing a whole
  network, one can search for a small *cell* (a convolutional cell and a downsampling
  "reduction" cell) and stack copies of it — this shrinks the search space and gives
  architectures that transfer across depths and datasets. Recurrent cells (LSTM, and the
  more recent Recurrent Highway Network, which adds depth inside the cell via highway gates
  `h = c⊙f(·) + (1−c)⊙h_prev`) are the recurrent analog.

- **Depthwise-separable convolutions (Chollet 2017) and pooling** are the cheap building
  blocks a vision search space would draw on: separable convolutions factor a standard
  convolution into a per-channel spatial filter plus a 1×1 mixing convolution, sharply
  cutting compute.

- **The motivating efficiency observation.** Across the RL-search and evolutionary-search
  literature, the reported budgets are enormous and almost entirely spent on training
  child models to convergence only to measure one scalar and discard the weights. This is
  the diagnostic fact that frames the problem: the expensive part is not the controller,
  it is re-learning weights from scratch for every candidate.

- **Transfer and multitask learning** establish that weights learned for one model/task
  are reusable for related models/tasks with little modification — evidence that a set of
  weights need not be tied to one fixed architecture.

- **Weight inheritance in neuro-evolution** mutates an architecture and carries the
  parent's weights into the child instead of reinitializing — a precedent for not throwing
  weights away between candidates.

## Baselines

**Reinforcement-learning NAS (Zoph & Le 2017; Zoph et al. 2018, NASNet).** An LSTM
controller `π(·;θ)` autoregressively samples an architecture; the child is trained to
convergence; its validation accuracy is the reward; `θ` is updated by REINFORCE with a
moving-average baseline. NASNet refines this to search a convolutional cell + reduction
cell and stack them. The discovered architectures are excellent. **Gap:** every candidate
is trained from scratch and discarded — 450 GPUs for 3-4 days (tens of thousands of
GPU-hours) in the NASNet case. The expense is structural, not incidental: it lives in the
from-scratch inner loop.

**Evolutionary NAS with weight inheritance (Real et al. 2017).** Maintain a population of
architectures; mutate the best ones; when a child is a small mutation of a parent, copy
the parent's weights rather than reinitializing, so the child needs only a short
fine-tune. **Gap:** weight inheritance is local (child must be near the parent) and tied
to the evolutionary outer loop; there is no single shared pool of weights serving the
*entire* search space simultaneously.

**SMASH / HyperNetworks (Brock et al. 2017).** Avoid training each child by having a
hypernetwork *generate* a child's weights from an encoding of its architecture, so a child
can be scored using generated weights. **Gap:** the hypernetwork produces weights via
tensor products, which forces the generated weight matrices into a low-rank subspace
(`rank(A·B) ≤ min(rank A, rank B)`). The search then favors architectures that happen to
work well under low-rank weights rather than under normal, unconstrained training — a
systematic bias.

**Performance prediction / progressive / hierarchical search (concurrent lines).** Predict
a child's final accuracy from a partial training curve, or grow architecture complexity
progressively, or search over a hierarchy of motifs — each reduces how much training each
candidate needs. **Gap:** these still train (partial) children and do not share one set of
weights across the whole space; the per-candidate cost is reduced but not collapsed.

## Evaluation settings

The natural yardsticks for an architecture-search method, all pre-existing:

- **Penn Treebank** language modeling: the standard pre-processed PTB corpus, word-level,
  measured by test perplexity. Recurrent-cell search is run here. Standard regularization
  for RNN language models — variational dropout, tying input embedding and output softmax
  weights, an `ℓ2` weight penalty — and a parameter budget (~24M) so comparisons are at
  matched capacity. Reward during search is a function of validation perplexity (e.g.
  `c / valid_ppl`). No post-training tricks (neural cache, dynamic evaluation).
- **CIFAR-10** image classification: 50,000 train / 10,000 test 32×32 images; standard
  preprocessing (per-channel mean/std normalization, pad-to-40 then random 32×32 crop,
  random horizontal flip), optional Cutout. Convolutional-architecture search is run here.
  Reward during search is validation accuracy on a held-out split (45,000 train / 5,000
  validation). Test error (%) is the metric; the cost of the search itself (GPU-days) is
  reported alongside, since efficiency is the whole point.
- Protocol: separate the data into a training split (for the child's weights) and a
  validation split (for the controller's reward, to select for generalization rather than
  training-set overfitting). After search, re-train the single best discovered architecture
  from scratch with standard settings and report its test metric.

## Code framework

The substrate is the existing RL-based-search harness: an LSTM controller that
autoregressively samples discrete architecture decisions, a child-model builder that
realizes a sampled architecture as a trainable network, an SGD trainer for the child, and
a REINFORCE update for the controller. What is *not* settled is how candidates relate to
one another — in the existing harness each sampled architecture is a fresh network with
its own freshly initialized weights, trained from scratch and discarded. That relationship
is exactly the slot to redesign. Everything method-specific is left as an empty stub.

```python
import torch
import torch.nn as nn


class Controller(nn.Module):
    """Autoregressive LSTM that samples a discrete architecture token-by-token,
    feeding each decision back as the next input. Emits the sampled architecture,
    the sum of log-probabilities of its decisions, and the sampled entropy."""

    def __init__(self, hidden_size=100, ...):
        super().__init__()
        # LSTM cell, per-decision embedding tables, per-decision softmax heads
        # TODO: define the exact set of decisions the controller samples
        pass

    def sample(self):
        # TODO: run the LSTM, sample each decision from its softmax, accumulate
        #       log-probs and entropy, feed the decision back as the next input
        pass


def build_child(architecture, weights):
    """Realize a sampled architecture as a runnable network.

    How weights relate to the architecture is the open design question:
    in the existing harness each child gets its OWN freshly initialized
    weights and is trained from scratch. That is the slot to redesign."""
    # TODO
    pass


def train_child(child, train_loader, optimizer):
    """One SGD pass training the child's weights on the training split."""
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(child(inputs), targets)
        loss.backward()
        optimizer.step()


def controller_reward(child, valid_loader):
    """Reward the controller maximizes: child performance on the VALIDATION
    split (accuracy, or a function of perplexity), to select for generalization."""
    # TODO: evaluate child on a validation minibatch, return scalar reward
    pass


def controller_update(controller, reward, log_prob, baseline, entropy,
                       opt_ctrl, entropy_weight, bl_dec):
    """REINFORCE with a moving-average baseline (the open slot is what reward
    we feed it and over what weights the child was evaluated)."""
    baseline = baseline - (1 - bl_dec) * (baseline - reward)
    loss = -log_prob * (reward - baseline) - entropy_weight * entropy
    opt_ctrl.zero_grad()
    loss.backward()
    opt_ctrl.step()
    return baseline


def search(controller, train_loader, valid_loader):
    # The outer search loop: sample an architecture, get weights for it somehow,
    # measure its validation reward, update the controller. The "get weights for
    # it somehow" step is where the cost lives and what must be redesigned.
    # TODO
    pass
```
