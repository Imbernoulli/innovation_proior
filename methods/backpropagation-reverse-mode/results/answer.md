# Backpropagation as reverse-mode automatic differentiation

## Problem

Given a differentiable computation that maps many adjustable parameters to one scalar loss,

  L = F(theta_1, ..., theta_P),

compute the full gradient (dL/dtheta_1, ..., dL/dtheta_P) cheaply enough to use first-order optimization. Finite differences need one extra evaluation per parameter. Forward sensitivity propagation likewise gives one input direction per sweep. The scalar-output case calls for the opposite organization: propagate the sensitivity of the one loss backward through the computation graph.

## Core derivation

Represent the executed computation as a directed acyclic graph or operation list. Each node v_i stores a numeric value. For a scalar loss L, attach an adjoint

  vbar_i = dL/dv_i.

Initialize the final loss node with Lbar = 1. For every operation v_j = f_j(parents(j)), processed in reverse topological order, accumulate into each parent v_i:

  vbar_i += vbar_j * (dv_j/dv_i).

If a value feeds multiple later operations, the adjoint contributions add. This is the chain rule organized as dynamic programming over shared subexpressions.

For a layered neural network with

  x_j = sum_i w_ij y_i + b_j,     y_j = phi(x_j),
  E = 1/2 sum_j (y_j - d_j)^2,

the reverse recurrence is:

  output:  delta_j = dE/dx_j = (y_j - d_j) phi'(x_j)

  weight:  dE/dw_ij = delta_j y_i

  hidden:  delta_i = dE/dx_i = phi'(x_i) sum_j w_ij delta_j.

Then update weights by gradient descent:

  w_ij <- w_ij - eta dE/dw_ij.

For a batch, sum or average the per-example gradients before the update. For an unrolled iterative computation, store the forward state history and sum gradient contributions into any tied parameter shared across unrolled copies.

## Algorithm

```python
import math

class Value:
    def __init__(self, data, parents=(), local_partials=()):
        self.data = float(data)
        self.parents = tuple(parents)
        self.local_partials = tuple(local_partials)
        self.adjoint = 0.0

def add(a, b):
    return Value(a.data + b.data, (a, b), (1.0, 1.0))

def mul(a, b):
    return Value(a.data * b.data, (a, b), (b.data, a.data))

def tanh(a):
    y = math.tanh(a.data)
    return Value(y, (a,), (1.0 - y * y,))

def topo_sort(root):
    order, seen = [], set()
    def visit(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            visit(p)
        order.append(v)
    visit(root)
    return order

def compute_gradients(loss, parameters):
    topo = topo_sort(loss)
    for v in topo:
        v.adjoint = 0.0
    loss.adjoint = 1.0

    for v in reversed(topo):
        for parent, local in zip(v.parents, v.local_partials):
            parent.adjoint += v.adjoint * local

    return {p: p.adjoint for p in parameters}

def train(parameters, data, learning_rate):
    for example in data:
        loss = scalar_loss(parameters, example)
        grads = compute_gradients(loss, parameters)
        for p in parameters:
            p.data -= learning_rate * grads[p]
```

## What makes it distinct

Backpropagation is not finite differences: it never perturbs weights and never chooses a finite-difference step size. It is not ordinary symbolic differentiation: it does not expand a derivative expression and duplicate shared subterms. It evaluates the original computation once, keeps the needed intermediate values, and performs a reverse sweep of local chain-rule updates. The asymmetry is the point: for one scalar loss and many parameters, one reverse traversal obtains all partial derivatives at a small constant-factor overhead over the forward computation, with storage for the forward trace as the main cost.
