The problem is to compute, for a differentiable computation with many adjustable parameters and a single scalar loss, the partial derivative of that loss with respect to every parameter. A layered neural network is the usual example: an input passes through weighted sums, nonlinear activations, and an output layer, and a scalar error measures how far the output is from the desired target. The difficulty is not that the derivatives are unknown in principle, but that the obvious ways to obtain them are poorly matched to the setting. Finite differences can wrap any program, but each parameter needs a separate perturbed evaluation, so a gradient step costs on the order of the number of parameters forward passes and is plagued by the trade-off between truncation error and roundoff error. Symbolic differentiation gives exact formulas, but for a composed network it duplicates shared subexpressions over and over and must be redone whenever the architecture changes. Forward sensitivity propagation is better: it carries the derivative of every intermediate with respect to one chosen parameter direction through the forward pass. Yet one such sweep answers only one directional question, so computing the full gradient still requires one sweep per parameter. For a scalar loss and many parameters, the natural direction to propagate is the reverse one.

The method is backpropagation, more precisely reverse-mode automatic differentiation. Instead of asking how each parameter affects the output, ask how the scalar loss changes when each intermediate value changes. Define the adjoint of an intermediate value v as dL/dv, the sensitivity of the loss L to that value. The final loss node is initialized with adjoint one, because dL/dL = 1. The computation graph is then traversed in reverse topological order. For each operation z = f(a, b, ...), the known adjoint of z is distributed to its inputs by the local chain rule: the contribution to the adjoint of an input is the adjoint of the output multiplied by the local partial derivative of the output with respect to that input. If one value feeds into several later operations, its total adjoint is the sum of the contributions coming back through all of them. Because the forward values and dependency edges have already been recorded, shared subexpressions are handled by accumulation rather than by duplicating derivative expressions. The cost of the reverse sweep is a small constant multiple of the forward computation, and it yields every parameter gradient at once.

In the special case of a layered feedforward network, the general adjoint rule reduces to the familiar neural-network formulas. Let x_j = sum_i w_ij y_i + b_j be the total input to unit j and y_j = phi(x_j) its output, with squared error E = 1/2 sum_j (y_j - d_j)^2. Starting from the output side, the error signal for unit j is delta_j = (y_j - d_j) phi'(x_j). The gradient for the weight w_ij is then delta_j y_i, the receiving unit's error signal times the sending unit's activity. To push the error one layer down, the sensitivity of the loss to a lower unit's output is sum_j delta_j w_ij, and converting from output sensitivity to total-input sensitivity gives delta_i = phi'(x_i) sum_j delta_j w_ij. Thus the algorithm begins at the output layer, propagates delta values backward through the weights, and accumulates weight gradients along the way. For an unrolled recurrent or iterative computation, the same logic applies to the unfolded graph, with tied weights receiving the sum of the gradients at all of their unrolled occurrences.

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

def scalar_loss(parameters, example):
    x, target = example
    # Tiny two-layer network built from elementary Value operations.
    w1, b1, w2, b2 = parameters
    h = tanh(add(mul(w1, x), b1))
    out = add(mul(w2, h), b2)
    diff = add(out, mul(Value(-1.0), target))
    return mul(diff, diff)  # squared error

def train(parameters, data, learning_rate):
    for example in data:
        loss = scalar_loss(parameters, example)
        grads = compute_gradients(loss, parameters)
        for p in parameters:
            p.data -= learning_rate * grads[p]

# Example usage:
# params = [Value(0.1), Value(0.0), Value(0.1), Value(0.0)]
# data = [(Value(2.0), Value(4.0)), (Value(3.0), Value(6.0))]
# train(params, data, learning_rate=0.01)
```

The key distinction of backpropagation is that it turns derivative computation into a local message-passing procedure over the executed computation graph rather than into a global symbolic formula or a sequence of numerical perturbations. It exploits the asymmetry that the loss is a single scalar: one reverse sweep gives the full gradient for all parameters, at the cost of storing the forward intermediate values that the backward pass needs. That organization is what makes gradient-based learning practical for deep and recurrent networks.
