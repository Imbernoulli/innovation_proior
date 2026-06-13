# The Multilayer Perceptron (MLP), distilled

A multilayer perceptron is a layered feedforward network of neuron-like units in which each
unit forms a weighted sum of its inputs and emits a smooth nonlinear function of that sum.
Stacking a layer of *hidden* (interior) units with a nonlinear activation between the input and
output lets the network recode its inputs into a space where the required mapping becomes
linearly separable — escaping the single-hyperplane ceiling of a one-layer network. The
weights are trained by gradient descent on output error, with the error signal for hidden
units — which have no supplied target — manufactured by propagating the output error backward
through the same connections (the **generalized delta rule**, i.e. error backpropagation).

## Problem it solves

Learn a mapping from input patterns to desired output patterns when the mapping is not
linearly separable (e.g. XOR, parity), where a single-layer network of threshold/linear units
provably cannot represent the mapping, and where interior units that could represent the
needed intermediate features have no target value in the training data to learn from.

## Key ideas

1. **Hidden units defeat linear separability.** A single threshold output unit answers 1 on
   the points where `w·x + b > 0` — one side of a hyperplane. XOR (`{01,10}` vs `{00,11}`) and
   parity have no such separating hyperplane. A hidden layer recodes the input: with the right
   hidden features the recoded patterns *are* separable. (XOR example: a hidden AND unit makes
   `(x1, x2, AND)` separable.) With enough hidden units some recoding exists for any mapping.

2. **The activation must be smooth and nonlinear.** Nonlinear, or stacked layers collapse:
   with row-vector activations and biases,
   `(xW₁ + b₁)W₂ + b₂ = x(W₁W₂) + (b₁W₂ + b₂)`, a single equivalent affine map with the
   one-layer ceiling. Smooth, or there is no gradient to follow: the step function's derivative
   is 0 almost everywhere. The logistic `y = 1/(1+e^{-x})` is both, and its slope is `y(1-y)`,
   computable from the unit's own (already-stored) output.

3. **Backpropagation manufactures the hidden error signal.** Hidden units have no target, but
   gradient descent needs only `∂E/∂w`, not a target. The chain rule gives a recursion that
   pushes the output error backward through the same weights, gated by each unit's local slope.

## Final method

Forward pass, for each unit `j`:

```
x_j = sum_i y_i w_ji  (+ bias as the weight on an always-on input)
y_j = f(x_j),         f the logistic 1/(1+e^{-x_j}),   f'(x_j) = y_j(1 - y_j)
```

Objective (sum over output units `j` and cases `c`):

```
E = (1/2) sum_c sum_j (y_{j,c} - d_{j,c})^2
```

Backward pass — define the error signal `δ_j = -∂E/∂x_j`:

```
output unit:  δ_j = (d_j - y_j) · y_j(1 - y_j)            # output error × local slope
hidden unit:  δ_j = y_j(1 - y_j) · sum_k δ_k w_kj         # local slope × back-propagated δ's
```

Weight update (gradient descent, with momentum):

```
Δw_ji(t) = ε · δ_j · y_i + α · Δw_ji(t-1)
```

The derivation chain: `∂E/∂y_j = y_j - d_j`; `∂E/∂x_j = ∂E/∂y_j · y_j(1-y_j)`;
`∂E/∂w_ji = ∂E/∂x_j · y_i`, since `x_j` is linear in the weights; and for a unit `i` that
is not an output, `∂E/∂y_i = sum_j ∂E/∂x_j · w_ji`, summing over the units `i` feeds. With
`δ_j = -∂E/∂x_j`, that is exactly the backward recursion. One forward sweep then one backward
sweep (each as cheap as the other) yield the gradient of *every* weight at once, because the
per-unit sensitivities
`∂E/∂x_j` are shared across all weights into `j` and all units below — the chain rule organized
as a dynamic program, not one re-evaluation per weight.

## Why each choice

- **Squared error** is differentiable and gives `∂E/∂y = y - d` at the output; on a single
  *linear* output layer it is a convex bowl with one minimum, and the rule reduces to the
  Widrow–Hoff delta rule `Δw = ε(d-y)y_i` — the correct special case (`f` linear, no hidden
  units), confirming the construction is the right generalization.
- **Logistic units**: smooth + nonlinear + bounded, with the cheap slope `y(1-y)` reusing the
  forward activation; `y(1-y)` peaks at `y=0.5` and vanishes at `0/1`, so learning concentrates
  on uncommitted units.
- **Small random initial weights** break symmetry: equal weights make all hidden units in a
  layer compute the same thing and receive the same `δ`, so they update identically and never
  differentiate (a stationary symmetric point of the error). Small keeps units in the
  responsive midrange of the logistic.
- **Targets 0.1 / 0.9** (not 0 / 1): the logistic reaches its asymptotes only at infinite
  weights, so exact 0/1 targets would drive weights to ±∞ and never zero the error.
- **Momentum** `α ≈ 0.9` adds a fraction of the previous update, cancelling the sign-flipping
  cross-ravine oscillation and accumulating the steady along-ravine descent — so `ε` need not be
  crippled to stay stable in long curved ravines. `α = 0` recovers plain descent (same
  solutions, slower).
- **Caveat**: with hidden units and a nonlinearity the error surface is non-convex, so descent
  can stall in a local minimum; extra hidden units / connections add weight-space directions
  that tend to route around such barriers.

## Working code

A multilayer perceptron with logistic units, batch (per-sweep) backpropagation, momentum, and
symmetry-breaking initialization:

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))            # y = 1/(1+e^{-x}); slope is y(1-y)


class MLP:
    def __init__(self, layer_sizes):
        # small RANDOM init: break symmetry, stay in the logistic's responsive midrange
        self.W = [np.random.uniform(-0.3, 0.3, (a, b))
                  for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(b) for b in layer_sizes[1:]]
        self.vW = [np.zeros_like(W) for W in self.W]   # momentum buffer = last weight change
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, Y):
        activations = [Y]
        for W, b in zip(self.W, self.b):
            activations.append(sigmoid(activations[-1] @ W + b))   # x = y@W+b ; y = f(x)
        return activations

    def backward(self, activations, target):
        Y_out = activations[-1]
        delta = (target - Y_out) * Y_out * (1.0 - Y_out)          # output: (d-y)*y(1-y)
        dW, db = [None] * len(self.W), [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            dW[k] = -(activations[k].T @ delta)                   # dE/dw_ji = -y_i * delta_j
            db[k] = -np.sum(delta, axis=0)
            if k > 0:
                Y_below = activations[k]
                # hidden delta: local slope * back-propagated sum_j delta_j w_kj
                delta = (delta @ self.W[k].T) * Y_below * (1.0 - Y_below)
        return dW, db

    def train(self, data, lr=0.5, momentum=0.9, n_sweeps=1000):
        X = np.stack([inp for inp, _ in data])
        T = np.stack([t for _, t in data])
        for _ in range(n_sweeps):
            activations = self.forward(X)
            dW, db = self.backward(activations, T)
            for k in range(len(self.W)):
                self.vW[k] = -lr * dW[k] + momentum * self.vW[k]   # Δw(t) = -ε·∂E/∂w + α·Δw(t-1)
                self.vb[k] = -lr * db[k] + momentum * self.vb[k]
                self.W[k] += self.vW[k]
                self.b[k] += self.vb[k]
```

## Modern instantiation

The same architecture — a hidden layer with a nonlinearity between two linear maps, trained by
backpropagation — is the standard nonlinear prediction head. The mutation-effect head uses the
mutation-induced embedding shift as its input, applies `Linear -> Dropout -> ReLU -> Linear`,
keeps a linear readout for real-valued regression, and is trained by the same backpropagated
gradient:

```python
import torch.nn as nn
import torch.nn.functional as F


class MutationPredictor(nn.Module):
    """Single-hidden-layer MLP over delta_embedding (mutant - WT)."""

    def __init__(self, embed_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)   # input -> hidden (the recoding layer)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)           # hidden -> scalar readout

    def forward(self, embedding, delta_embedding):
        h = self.fc1(delta_embedding)
        h = self.dropout(h)
        h = F.relu(h)                                 # nonlinear hidden representation
        return self.fc2(h).squeeze(-1)                # real-valued prediction, shape [B]
```

Both are the same multilayer perceptron: a learned nonlinear hidden representation between two
affine maps, with the weights set by descending the backpropagated error gradient.
