## Research question

The central problem is structural credit assignment in a differentiable computation with many adjustable quantities. A layered neural network makes this concrete: an input vector is pushed through affine combinations, nonlinear unit activations, and output units; a scalar error measures the mismatch between the produced output and the desired output; and learning needs to decide how every connection weight should change so that the scalar error goes down.

The hard case begins when the useful features are not hand-coded. Perceptron-style learning can adjust direct input-output connections, but hidden units have no target values supplied by the data. A learning rule must infer what internal units should represent only from the final error. A viable method has to compute the partial derivative of that one scalar error with respect to every weight in a large composed computation, without paying one full evaluation per weight and without asking a person to hand-derive a new formula for each architecture.

## Background

A feedforward network can be read as a numerical program. For each unit j, a total input is formed as a weighted sum of lower-level unit outputs plus a bias,

  x_j = sum_i w_ij y_i + b_j,

and the unit output is a differentiable nonlinear function y_j = phi(x_j). The training criterion for one or more input-output cases is commonly a scalar squared error,

  E = 1/2 sum_c sum_j (y_{j,c} - d_{j,c})^2,

where y is an output unit's produced value and d is its desired value. Gradient descent in weight space asks for dE/dw for every connection.

The same pattern appears outside neural networks. A numerical model is a finite sequence of elementary differentiable operations: additions, multiplications, elementary functions, and assignments to intermediate variables. Wengert-style operation lists make this explicit by naming every intermediate value. The chain rule is local at each elementary operation, but a whole model may contain loops, branches, and many shared subexpressions, so the difficulty is not whether derivatives exist locally. The difficulty is how to organize the derivative computation so the shared work in the original program is not repeated for every parameter.

The existing numerical-analysis choices have clear weaknesses for this setting. Manual differentiation is slow and error-prone. Finite differences are easy to wrap around a program, but a gradient in P parameters requires O(P) extra evaluations and depends on a step size h that trades truncation error against roundoff error. Symbolic differentiation can produce exact expressions, but expression swell duplicates common subexpressions and becomes awkward for ordinary programs with control flow. Forward sensitivity propagation keeps derivative values alongside the ordinary values, but one sweep naturally carries the effect of one input direction; a full gradient with respect to many weights takes one such sweep per direction.

## Baselines

**Perceptron and direct delta-rule learning.** With no learned hidden representation, the output error can be pushed directly onto each input-output weight. This works for linearly separable mappings or models with fixed feature analyzers, but it does not solve the hidden-unit problem because no target state is supplied for an internal unit.

**Finite-difference gradients.** For a scalar objective E(w), approximate one component by

  dE/dw_i ~= (E(w + h e_i) - E(w)) / h.

This treats the model as a black box and needs no derivative code. Its gap is cost and numerical reliability: P parameters require P perturbed evaluations, and h cannot be chosen to eliminate both truncation and cancellation error.

**Manual or symbolic differentiation.** Differentiate the expression for E by algebraic rules and then evaluate the resulting derivative expressions. This can reveal structure on small closed forms, but it becomes brittle as architectures change, and naive symbolic expansion repeatedly duplicates products and nested subexpressions that the forward computation evaluated only once.

**Forward sensitivity propagation.** Attach a tangent value to every intermediate and propagate the effect of one selected parameter through the same operation list as the primal computation. This is exact to machine precision for that selected direction and handles ordinary elementary operations. Its gap is the scalar-output, many-parameter case: to fill the whole gradient, the sweep must be repeated for each independent parameter direction.

**Second-order optimization.** Newton-like methods can use curvature to converge in fewer optimization steps, but they require many more derivative quantities than first-order descent. For large neural networks, the immediate bottleneck is already the first derivative with respect to every weight.

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
