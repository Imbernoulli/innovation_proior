## Research question

The central problem is structural credit assignment in a differentiable computation with many adjustable quantities. A layered neural network makes this concrete: an input vector is pushed through affine combinations, nonlinear unit activations, and output units; a scalar error measures the mismatch between the produced output and the desired output; and learning needs to decide how every connection weight should change so that the scalar error goes down.

When the useful features are not hand-coded, the network has hidden units for which the data supplies no target values; a learning rule must infer what internal units should represent only from the final error. The setting calls for computing the partial derivative of that one scalar error with respect to every weight in a large composed computation.

## Background

A feedforward network can be read as a numerical program. For each unit j, a total input is formed as a weighted sum of lower-level unit outputs plus a bias,

  x_j = sum_i w_ij y_i + b_j,

and the unit output is a differentiable nonlinear function y_j = phi(x_j). The training criterion for one or more input-output cases is commonly a scalar squared error,

  E = 1/2 sum_c sum_j (y_{j,c} - d_{j,c})^2,

where y is an output unit's produced value and d is its desired value. Gradient descent in weight space asks for dE/dw for every connection.

The same pattern appears outside neural networks. A numerical model is a finite sequence of elementary differentiable operations: additions, multiplications, elementary functions, and assignments to intermediate variables. Wengert-style operation lists make this explicit by naming every intermediate value. The chain rule is local at each elementary operation, and a whole model may contain loops, branches, and many shared subexpressions.

Several numerical-analysis choices exist for this setting. Manual differentiation works out the derivative formulas by hand. Finite differences wrap directly around a program: a gradient in P parameters uses O(P) extra evaluations with a step size h that trades truncation error against roundoff error. Symbolic differentiation produces exact derivative expressions by algebraic rules. Forward sensitivity propagation keeps derivative values alongside the ordinary values, carrying the effect of one input direction through each sweep.

## Baselines

**Perceptron and direct delta-rule learning.** The output error is pushed directly onto each input-output weight. This applies to linearly separable mappings or models with fixed feature analyzers, where target states are available at the units being adjusted.

**Finite-difference gradients.** For a scalar objective E(w), approximate one component by

  dE/dw_i ~= (E(w + h e_i) - E(w)) / h.

This treats the model as a black box and needs no derivative code. P parameters require P perturbed evaluations, and h trades truncation error against cancellation error.

**Manual or symbolic differentiation.** Differentiate the expression for E by algebraic rules and then evaluate the resulting derivative expressions. This yields exact closed forms for small expressions.

**Forward sensitivity propagation.** Attach a tangent value to every intermediate and propagate the effect of one selected parameter through the same operation list as the primal computation. This is exact to machine precision for that selected direction and handles ordinary elementary operations; one sweep carries one parameter direction.

**Second-order optimization.** Newton-like methods use curvature to converge in fewer optimization steps, requiring more derivative quantities than first-order descent.

## Evaluation settings

The natural testbed is a differentiable layered network with input units, one or more hidden layers, output units, differentiable activations, and a scalar error over a finite set of input-output cases. The settings that expose the structural credit-assignment problem are tasks where direct input-output connections are insufficient and an intermediate representation must be learned, such as symmetry detection or relational mappings encoded through distributed internal units.

The quantities to inspect are the scalar error, the gradient components for all weights, the number of model evaluations needed to obtain one full gradient, and whether hidden units acquire useful internal features under gradient descent. A gradient procedure should also extend from ordinary feedforward layers to an iterative or recurrent computation by unfolding it through time, while accounting for the fact that tied weights appear at multiple unrolled positions.

## Code framework

The scaffold is a generic scalar-objective program with differentiable elementary operations. The forward pass and loss already exist; the open slot is the source of all partial derivatives with respect to the adjustable leaves.

```python
import math

class Value:
    def __init__(self, data, parents=(), local_partials=()):
        self.data = float(data)
        self.parents = tuple(parents)
        self.local_partials = tuple(local_partials)

def add(a, b):
    return Value(a.data + b.data, (a, b), (1.0, 1.0))

def mul(a, b):
    return Value(a.data * b.data, (a, b), (b.data, a.data))

def tanh(a):
    y = math.tanh(a.data)
    return Value(y, (a,), (1.0 - y * y,))

def scalar_loss(parameters, example):
    # TODO: build the composed numerical calculation and return one scalar Value.
    pass

def compute_gradients(loss, parameters):
    # TODO: compute d loss / d parameter for every adjustable leaf.
    pass

def train(parameters, data, learning_rate):
    for example in data:
        loss = scalar_loss(parameters, example)
        grads = compute_gradients(loss, parameters)
        for p in parameters:
            p.data -= learning_rate * grads[p]
```
