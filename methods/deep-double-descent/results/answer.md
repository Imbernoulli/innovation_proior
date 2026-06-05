# Deep Double Descent

## Problem

Classical statistics says test error is U-shaped in model complexity ("larger models are worse" past
a threshold; early stopping helps); modern deep learning says "larger models are better" and training
to zero training error helps. Both are empirically true. The goal is a single picture of test error
vs. complexity that unifies them — and a notion of "complexity" general enough to cover model size,
training time, regularization, and dataset size at once.

## Key idea

The two wisdoms are halves of one curve split at the **interpolation threshold** (where training
error hits ≈0): test error follows the classical U up to it, then *descends again* in the
over-parameterized regime — *double descent*. The right complexity axis is not parameter count but
**Effective Model Complexity (EMC)**: the largest training-set size a procedure can fit to ≈0 train
error. Test error peaks where EMC ≈ n (number of training samples).

## Effective Model Complexity

For a training procedure T (architecture + optimizer + #steps + augmentation + regularization —
anything mapping a labeled set S to a classifier T(S)), distribution D, and small ε:

  EMC_{D,ε}(T) := max { n : E_{S~D^n}[ Error_S(T(S)) ] ≤ ε }   (ε ≈ 0.1 heuristically).

EMC rises with width, but also with *training time*, with augmentation off, with less regularization.
Unlike Rademacher complexity / VC dimension, EMC depends on (1) the *true labels* of the distribution
and (2) the *training procedure* — the two ingredients needed to locate a peak that moves with label
noise and with epochs.

## Generalized Double Descent Hypothesis

For natural D, neural-net procedure T, small ε, predicting n samples:
- **Under-parameterized** (EMC ≪ n): increasing effective complexity *decreases* test error.
- **Over-parameterized** (EMC ≫ n): increasing effective complexity *decreases* test error.
- **Critically parameterized** (EMC ≈ n): increasing effective complexity *may increase or decrease*
  test error.

Test error peaks at the interpolation threshold EMC ≈ n, with a "critical interval" around it (whose
width depends on D and T) where more complexity can hurt.

## Three corollaries (three ways to cross EMC ≈ n)

- **Model-wise double descent:** fix a large #steps, vary model width. Test error: classical U, peak
  at the size that first interpolates the train set, then a second descent. The hypothesis predicts
  regimes where *bigger models are worse*. Label noise, data augmentation, and more samples raise the
  interpolation threshold and shift the peak toward larger models.
- **Epoch-wise double descent:** fix a large model, vary training time (training longer raises EMC, so
  the model goes under→over-parameterized within one run). Test error: decreases, increases near
  interpolation, decreases again — *training longer can correct overfitting*. Medium models follow the
  classical U (early stopping best); small models decrease monotonically.
- **Sample-wise non-monotonicity:** fix model + procedure, vary n (this crosses EMC ≈ n from the other
  side). More data shrinks error overall but shifts the peak right; near the critical regime these
  cancel (more data doesn't help) and can combine so that the hypothesis predicts *more data can
  hurt*.

Optimal early stopping removes the phenomena — consistent with the hypothesis, since stopping before
≈0 train error keeps EMC below n.

## Mechanism

At EMC ≈ n there is essentially a *unique* interpolating model, highly sensitive to label noise /
model mis-specification (forced to fit every point, slight noise destroys global structure → high
test error). Over-parameterized, many interpolating models exist and the (minimum-norm) one gradient
descent finds absorbs the noise while generalizing. Provable for linear least-squares and Random
Fourier Features, where EMC = width d exactly and the high test-error ridge follows n = d. Label noise
is a proxy for model mis-specification, not the fundamental cause; double descent appears without it
under mis-specification.

## Implementation (measurement harness)

There is no new architecture or loss; the contribution is the EMC definition and hypothesis, realized
by measuring EMC and sweeping one knob at a time.

```python
import numpy as np
EPS = 0.1  # "approximately zero" train error

def add_label_noise(labels, p, num_classes, rng):
    flip = rng.random(len(labels)) < p           # corrupt w.p. p; otherwise keep original label
    noisy = labels.copy()
    replacement = rng.integers(0, num_classes - 1, size=int(flip.sum()))
    original = labels[flip]
    noisy[flip] = replacement + (replacement >= original)              # else uniform wrong label
    return noisy                                 # drawn ONCE, fixed across epochs

def make_model(width, fixed):
    # Existing factory: ResNet18 [k,2k,4k,8k], 5-layer CNN [k,2k,4k,8k]+FC,
    # Transformer d_model with d_ff=4*d_model, or RFF width d.
    return fixed.make_model(width)

def train(model, train_data, test_data, optimizer, num_steps, fixed, record_every=None):
    # Existing training loop: cross-entropy for vision, label-smoothed CE for Transformers, MSE for RFF.
    return fixed.train(model, train_data, test_data, optimizer, num_steps, record_every=record_every)

def effective_complexity(procedure, distribution, sample_grid, trials, epsilon=EPS):
    emc = 0
    for n in sorted(sample_grid):
        if np.mean([procedure(distribution.sample(n)).train_error()
                    for _ in range(trials)]) <= epsilon:
            emc = n
    return emc

def model_size_sweep(widths, fixed):
    noisy_y = add_label_noise(fixed.y, fixed.p, fixed.num_classes, fixed.rng)
    out = []
    for w in widths:
        m = make_model(w, fixed)
        out.append((w, *train(m, (fixed.train_x, noisy_y), (fixed.test_x, fixed.test_y),
                              fixed.make_optimizer(m), fixed.num_steps, fixed)[:2]))
    return out

def training_time_sweep(width, step_budget, fixed):
    noisy_y = add_label_noise(fixed.y, fixed.p, fixed.num_classes, fixed.rng)
    m = make_model(width, fixed)
    return train(m, (fixed.train_x, noisy_y), (fixed.test_x, fixed.test_y),
                 fixed.make_optimizer(m), step_budget, fixed, record_every=1)[2]

def sample_count_sweep(width, sample_sizes, fixed):
    out = []
    for n in sample_sizes:
        x, y = fixed.dataset.subset(n)
        noisy_y = add_label_noise(y, fixed.p, fixed.num_classes, fixed.rng)
        m = make_model(width, fixed)
        out.append((n, train(m, (x, noisy_y), (fixed.test_x, fixed.test_y),
                             fixed.make_optimizer(m), fixed.num_steps, fixed)[1]))
    return out
```
