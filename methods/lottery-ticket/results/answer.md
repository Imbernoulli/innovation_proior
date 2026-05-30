# The Lottery Ticket Hypothesis

## Problem

Pruning shows a trained network's function can be represented by a subnetwork with <10% of the
weights — yet training that sparse architecture from a fresh random initialization learns slower and
reaches lower accuracy the sparser it is. A good sparse solution provably exists (pruning produced
one) but SGD cannot reach it from a random start. The question: is a sparse architecture inherently
hard to train, or is there a missing ingredient that makes it trainable from the start?

## Key idea

The active ingredient is the **initialization**, not the architecture. A randomly-initialized dense
network contains a sparse subnetwork — a *winning ticket* — whose original initial weights, together
with its connectivity, make it trainable in isolation to the full network's accuracy.

## The Lottery Ticket Hypothesis

A randomly-initialized, dense network f(x; θ₀) contains a subnetwork (mask m ∈ {0,1}^{|θ|}) such that
training f(x; m ⊙ θ₀) in isolation matches the dense network's test accuracy in at most the same
number of iterations. Formally, if the dense net reaches min validation loss at iteration j with test
accuracy a, there exists m with: j′ ≤ j (commensurate training time), a′ ≥ a (commensurate accuracy),
and ‖m‖₀ ≪ |θ| (fewer parameters).

## The central experiment

1. Randomly initialize f(x; θ₀), θ₀ ~ D_θ (Gaussian Glorot).
2. Train for j iterations → θ_j.
3. Prune p% of θ_j by smallest magnitude → mask m.
4. **Reset** each surviving weight to its value in θ₀ (the original init, *before* training) → the
   winning ticket f(x; m ⊙ θ₀).

The distinguishing step is (4): rewind survivors to their *pre-training* random values, not the
trained values. This isolates the initialization as the cause.

**Iterative magnitude pruning.** Rather than one large cut (one-shot), repeat over n rounds: train the
masked network, prune a small fraction (e.g. 20%) of the *surviving* weights by magnitude, reset
survivors to θ₀, repeat. Keeping 80% per round compounds geometrically (fraction remaining = 0.8^k
after k rounds). Iterative pruning finds winning tickets at smaller sizes than one-shot because the
magnitude ranking is refreshed at every sparsity level.

**Pruning heuristic.** Layer-wise magnitude pruning (remove a fixed fraction of the lowest-|magnitude|
weights within each layer); connections to the output layer pruned at half the rate; unstructured
(per-weight). P_m = ‖m‖₀/|θ| denotes the fraction of weights remaining.

**Random-reinitialization control.** Keep the mask m but draw a fresh θ′₀ ~ D_θ and train f(x; m ⊙ θ′₀).
Winning tickets perform far worse this way — structure alone does not explain their success; the
specific initialization does.

**Deeper networks.** On Resnet-18 and VGG-19, iterative pruning finds winning tickets only with
learning-rate warmup (the large early steps at standard learning rates otherwise destroy the
reset-to-init ticket).

## Why the initialization matters

Winning-ticket weights move *farther* during training than other weights, so they are not "already
trained"; the benefit is that m ⊙ θ₀ lands in a region of the loss landscape amenable to the
optimizer for this dataset/model, and the structure (found via training data) encodes a task-specific
inductive bias. Test accuracy rises then falls with pruning (an Occam's hill): the dense net has
excess capacity, extreme pruning too little. The **Lottery Ticket Conjecture**: SGD seeks out and
trains a well-initialized subnetwork; dense networks are easier to train because they contain more
candidate subnetworks, so one is more likely to have won the initialization lottery.

## Implementation

Grounded in the canonical magnitude-pruning routine and iterative experiment loop.

```python
import numpy as np
import copy

def prune_by_percent_once(percent, mask, final_weight):
    sorted_weights = np.sort(np.abs(final_weight[mask == 1]))     # rank surviving weights by |value|
    cutoff_index = int(np.round(percent * sorted_weights.size))
    cutoff = sorted_weights[cutoff_index]
    return np.where(np.abs(final_weight) <= cutoff, np.zeros(mask.shape), mask)  # zero below cutoff

def prune(percents, masks, final_weights):
    # `percents` sets the output layer to half the rate of the rest
    return {k: prune_by_percent_once(percents[k], masks[k], final_weights[k]) for k in masks}

def find_winning_ticket(model_init, train_fn, percents, num_rounds):
    theta_0 = model_init()                                       # random init θ₀, drawn once
    masks   = {k: np.ones_like(v) for k, v in theta_0.items()}   # start dense
    weights = copy.deepcopy(theta_0)
    for _ in range(num_rounds):
        trained, metrics = train_fn(weights, masks)              # train masked subnetwork (mask fixed)
        masks = prune(percents, masks, trained)                  # magnitude-prune survivors
        weights = {k: masks[k] * theta_0[k] for k in masks}      # RESET survivors to θ₀  -> m ⊙ θ₀
    return masks, weights

def random_reinit_control(model_init, masks):                    # control: structure, not init
    theta_prime = model_init()
    return {k: masks[k] * theta_prime[k] for k in masks}         # m ⊙ θ′₀ — trains worse

def masked_forward(x, weights, masks, layer_order, layer_op):
    h = x
    for k in layer_order:
        h = layer_op(h, masks[k] * weights[k])                   # pruned weights held at 0, no gradient
    return h
```

Lenet-300-100 on MNIST: Adam (1.2e-3), fc pruned 20%/round. Conv-2/4/6 on CIFAR-10: Adam, conv
10–15% / fc 20%. Resnet-18, VGG-19: SGD momentum 0.9, stepped LR 0.1→0.01→0.001, with warmup. All
initialized Gaussian Glorot.
