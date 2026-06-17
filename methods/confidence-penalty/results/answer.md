# Confidence penalty

The confidence penalty is an output-distribution regularizer: it adds the *negative entropy*
of the softmax predictions to the training loss, penalizing low-entropy (over-confident)
outputs. It ports the entropy bonus used in reinforcement learning to keep a policy stochastic
into supervised learning, where it discourages the peaked, near-deterministic output
distributions that accompany overfitting. It is a single-hyperparameter, closed-form term that
drops into an existing pipeline without changing the model, optimizer, or schedule.

## Problem it solves

Large classifiers overfit despite large datasets, and the symptom is visible in the output:
an over-confident network places almost all its softmax mass on one class (low entropy).
Standard regularizers (weight decay, dropout, batch norm, early stopping) all act on weights
or hidden activations and leave the output distribution unregularized. A good output
regularizer should (1) improve generalization across architectures and tasks, (2) drop in
without retuning the existing hyperparameters, (3) avoid writing an arbitrary wrong-label
target distribution into the training labels, and (4) be invariant to the network's
parameterization — the output has a natural scale, the weights do not.

## Key idea

A softmax output `p_theta(y|x)` has entropy `H(p_theta(y|x)) = - sum_i p_theta(y_i|x) log
p_theta(y_i|x)`; over-confidence is low entropy. Penalize low entropy by adding the negative
entropy to the negative log-likelihood:

```
L(theta) = - sum log p_theta(y|x) - beta H(p_theta(y|x))
```

`beta` controls the strength. Minimizing `L` maximizes `H`, i.e. penalizes confident
(low-entropy) predictions.

## Gradient (closed form, no extra passes)

With softmax Jacobian `d p_j / d z_i = p_j(delta_{ij} - p_i)`, the entropy gradient w.r.t.
logit `z_i` is

```
dH / dz_i = p_i ( - log p_i - H(p) ),
```

the **weighted deviation from the mean surprisal** (`- log p_i` is class `i`'s surprisal, `H`
its mean under `p`). Under gradient descent on the loss term `- beta H`, it pulls the dominant
(confident) logit down and, because of the `p_i` weight, barely touches near-zero classes — so
it flattens toward uniform without explicitly forcing all incorrect classes to the same target.
One extra reduction over logits already computed; no auxiliary forward/backward passes.

## Connection to label smoothing (the KL direction)

Both regularizers pull the output toward the uniform distribution `u`, in opposite KL
directions:

- **Label smoothing**: target `q'(k) = (1 - epsilon) delta_{k,y} + epsilon u(k)` gives the
  loss term `H(u, p) = D_KL(u || p) + H(u)`, i.e. the **forward** KL `D_KL(u || p)`. Its
  log-ratio is weighted by the *constant* `u_i = 1/K` — equal, fixed pressure on every class —
  and it forces every incorrect class toward the same target `epsilon/K`.
- **Confidence penalty**: `D_KL(p || u) = sum_i p_i log(p_i / u_i) = - H(p) + log K`, so up to
  the constant `log K`, penalizing `D_KL(p || u)` *is* the confidence penalty. Its log-ratio is
  weighted by the model's *own* `p_i` — adaptive pressure concentrated on the currently
  over-confident classes — and it specifies no target for incorrect classes.

So the confidence penalty is label smoothing with the KL direction reversed; the reversal is
exactly why it (a) adapts its pressure to the model's current confidence, (b) avoids explicitly
equalizing incorrect-class ratios, and (c) avoids inserting a fixed wrong-label target
distribution into every example. In its entropy form, the uniform distribution remains the
maximum-entropy reference point; choosing a non-uniform reference `u` in `D_KL(p || u)`
generalizes to a family of confidence regularizers when a task supplies such a prior.

## Annealing and thresholding

In RL the entropy bonus is on throughout (you want exploration); in supervised learning you
want quick convergence early and humility only near the end. So either anneal `beta` from weak
to strong, or use a hinge that activates only once entropy drops below a threshold `Gamma`:

```
L(theta) = - sum log p_theta(y|x) + beta max(0, Gamma - H(p_theta(y|x)))
```

When `H >= Gamma`, no penalty (no interference with convergence); when `H < Gamma`, the penalty
switches on in proportion to how far below threshold the entropy has fallen. The positive sign
is necessary under loss minimization: below threshold, the derivative with respect to `H` is
`-beta`, so minimizing pushes entropy upward. Equivalently, up to an additive constant, the
hinge is a clipped negative-entropy reward `- beta min(H, Gamma)`. The extra threshold is a
second hyperparameter, so the single-`beta` version is the simpler default.

## Choosing beta

One task-dependent knob, swept while the model's other hyperparameters are held fixed. Its value
trades off data fit against output humility, so it should be selected on validation data rather
than treated as a universal constant.

## A side regularity it explains

The one-hot cross-entropy per-logit gradient is `p - q`, bounded in `[-1, 1]`. A confident,
peaked output on a *misclassified* example yields a near-maximal gradient (true class: `p ≈ 0`,
`q = 1`). So keeping outputs from peaking also keeps gradient norms smaller and steadier
through training — the same mechanism, two benefits.

## Working code

The term fills the `compute_regularization` slot of the fixed training pipeline (data,
model, cross-entropy, optimizer, schedule all unchanged). Numerically stable form: `log_softmax`
once, then recover `p`.

```python
import torch
import torch.nn.functional as F


def compute_regularization(model, inputs, outputs, targets, config):
    """Confidence penalty: L += - beta * H(p), penalizing low-entropy
    (over-confident) softmax outputs.  outputs: [B, num_classes] logits."""
    beta = float(config.get("beta", 0.1))        # single knob, swept on validation data

    log_p = F.log_softmax(outputs, dim=-1)       # stable: one pass, no log(softmax)
    p = log_p.exp()                              # probabilities

    entropy = -(p * log_p).sum(dim=-1).mean()    # H(p) = - sum_i p_i log p_i, batch mean
    return -beta * entropy                        # negative entropy => penalize confidence
```

Thresholded (hinge) variant — penalize each example only when its entropy has fallen below
`Gamma`:

```python
def compute_regularization_thresholded(model, inputs, outputs, targets, config):
    """+ beta * max(0, Gamma - H(p)): switch on only once the model is too confident."""
    beta = float(config.get("beta", 0.1))
    Gamma = float(config.get("entropy_threshold", 0.8))
    log_p = F.log_softmax(outputs, dim=-1)
    p = log_p.exp()
    entropy = -(p * log_p).sum(dim=-1)
    return beta * torch.clamp(Gamma - entropy, min=0.0).mean()
```
