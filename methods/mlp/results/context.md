# Context: supervised learning in layered networks of neuron-like units (circa 1985)

## Research question

There is a set of input patterns and, for each, a desired output pattern, and the goal is
to find a network of simple neuron-like units that produces the right output for each input
and generalizes sensibly to inputs it has not seen. A unit is cheap: it forms a weighted sum
of the values arriving on its incoming connections and emits some function of that sum. The
only adjustable quantities are the connection weights. When the units are wired straight from
input to output, the existing learning rules are reliable but the class of input-output
mappings they can represent is sharply bounded — there are simple, clearly desirable mappings
they provably cannot produce. To represent a broader class of mappings, a network can include
units between the input and the output whose values are not dictated by the data — units that
are free to represent whatever intermediate features make the mapping easy. The question is
how to adjust the weights of such interior units from output error alone, so that the network
learns to produce the desired outputs.

## Background

A network is read as a numerical machine. Each unit `j` forms a total input that is a *linear*
function of the outputs `y_i` of the units feeding it and of the weights on those connections,

```
x_j = sum_i  y_i * w_ji,
```

and emits an output `y_j = f(x_j)`, where `f` is the unit's activation function. A bias is folded
in as an extra incoming connection from a unit whose value is always 1; the weight on it is the
bias and acts as a threshold of the opposite sign, so biases need no separate treatment. Units
are arranged in layers: an input layer whose values are set by the data, zero or more interior
layers, and an output layer; each unit sends its output only to higher layers, so the output
vector is computed by a single forward sweep that fills in each layer from the ones below.

The prevailing wisdom about what such a machine can compute is set by two facts about the
single-layer case. First, when there are no interior units and the output unit is a threshold
unit, the set of inputs it maps to 1 is exactly the set on one side of a hyperplane `w·x + b > 0`
in input space — a *linearly separable* set. Second, many obviously useful mappings are not
linearly separable. The exclusive-or of two binary inputs is the canonical one: the two inputs
that differ (`01`, `10`) must give 1 and the two that agree (`00`, `11`) must give 0, and no
single hyperplane separates `{01,10}` from `{00,11}`. Parity on `n` bits is the same pathology at
scale — the most similar patterns (differing in one bit) demand opposite answers. The careful
catalog of which mappings fall outside the single-layer reach was worked out by Minsky and Papert
(*Perceptrons*, 1969), and their analysis chilled enthusiasm for these networks for a decade. The
same analysis, though, contains the constructive seed: if the input pattern is augmented with the
right interior units — for XOR, a unit that fires only when both inputs are on — the augmented
representation *is* linearly separable, and then a single output unit suffices. The interior units
change the similarity structure: they re-code the inputs into a space where the required mapping
becomes easy. Minsky and Papert observe that with enough such interior units there always exists a
re-coding that supports any required mapping.

A second pre-existing fact constrains the choice of unit. The reliable single-layer rules come in
two flavors. The threshold unit makes a clean binary decision but its output is a step: its
derivative is zero almost everywhere and undefined at the threshold. A continuous unit has a slope
but does not make a crisp decision.

A third fact: a layer of units whose activation is *linear* buys nothing. With row-vector
activations, one linear layer with bias gives `y_1 = y_0 W_1 + b_1`; a second gives
`y_2 = y_1 W_2 + b_2 = y_0 (W_1 W_2) + (b_1 W_2 + b_2)`. The composition is just one affine
map with a new weight matrix and a new bias, so any number of linear interior layers collapses to
a single equivalent direct connection. Whatever the interior units do, they cannot be merely linear.

## Baselines

**Rosenblatt's perceptron (Rosenblatt, *Principles of Neurodynamics*, 1961).** A single threshold
unit, `y = h(w·x + b)` with `h` the step function (1 if the argument exceeds 0, else 0). Its
learning procedure adjusts weights only on mistakes,

```
w(t+1) = w(t) + r * (d - y) * x,
```

where `d` is the desired output, `y` the produced output, `r` a rate. The perceptron convergence
theorem makes this a *guaranteed* rule: if the input patterns are linearly separable with margin
`gamma` and lie within radius `R`, the procedure stops after at most `(R/gamma)^2` mistakes with a
weight vector that classifies every pattern correctly.

**The delta rule / least-mean-squares (Widrow and Hoff, 1960).** Replace the perceptron's
thresholded mistake with the *continuous* error of a differentiable unit and descend the squared
error `E = (1/2) sum_j (t_j - y_j)^2`. Differentiating gives the update

```
Delta w_ji = alpha * (t_j - y_j) * f'(x_j) * y_i,
```

a step proportional to the output error, the unit's slope, and the incoming signal. For a single
layer this is honest gradient descent on a *convex* bowl: one global minimum, and a small enough
rate reaches it. This rule makes "compare output to target, scale the weight change by the slope
and the input" precise.

**Unsupervised / competitive recoding.** One way to obtain interior units is to let them
self-organize by an unsupervised rule, so useful intermediate features develop on their own.

**Fixed, hand-chosen interior representation.** Alternatively, decide on a priori grounds what the
interior units should compute and freeze them, training only the readout. This sidesteps the
credit-assignment problem entirely.

**Stochastic-unit learning with an equilibrium criterion (the Boltzmann-machine line).** Interior
units can be trained when the units are stochastic and learning is driven by the difference
between two phases in which the network is allowed to reach thermal equilibrium. This does learn
interior representations, and requires stochastic units, two separate relaxation-to-equilibrium
phases per update, and a symmetric connection structure.

## Evaluation settings

The natural yardsticks are small mappings whose difficulty is exactly the non-linear-separability
the single-layer rules cannot handle, plus storage/recoding tasks where the interior units must
invent structure:

- **Exclusive-or** of two binary inputs, and **parity** of `n` binary inputs (`n` from 2 up to a
  handful): the output is 1 when an odd number of inputs are on. The most similar inputs require
  opposite outputs, so these cannot be done without interior units. Metric: whether the network
  reaches the correct binary output on all `2^n` patterns, and how many sweeps through the pattern
  set that takes.
- **Symmetry detection**: given a one-dimensional array of binary inputs, output 1 if the pattern
  is symmetric about its center. A single input position carries no evidence about global symmetry,
  so interior units are required.
- **Encoding / structured-recall tasks**: store a set of relational triples or arbitrary
  input-output associations through a narrow interior layer, so the interior units are forced to
  compress the inputs into a re-usable code; inspect what features the interior units come to
  represent. Protocol: present the input-output pairs repeatedly (in sweeps), update weights by the
  rule under test, start from small random weights, and measure sweeps-to-criterion and the
  qualitative interior representation.

## Code framework

The machinery that already exists is a layered feedforward network with a forward sweep, a
squared-error objective, and a generic gradient-style weight-update loop. The substrate is a
generic supervised-network harness: a `Network` that holds layers and runs a forward pass, a
squared-error loss, and a training loop that sweeps the data and applies whatever per-weight
update rule is filled in.

```python
import numpy as np


def squared_error(output, target):
    """E = 1/2 * sum (output - target)^2, the training criterion."""
    return 0.5 * np.sum((output - target) ** 2)


class Network:
    """A layered feedforward network of neuron-like units.

    Each layer forms x = (input @ W) + b for its incoming weights and emits
    f(x). The forward pass fills in each layer from the ones below. How the
    INTERIOR weights should change to reduce the output error is exactly what
    is not yet known.
    """

    def __init__(self, layer_sizes):
        self.W = [np.random.uniform(-0.3, 0.3, (a, b))      # break symmetry: small random
                  for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(b) for b in layer_sizes[1:]]

    def unit(self, x):
        # TODO: the activation function for an interior unit. The reliable
        #       single-layer rules want it differentiable; nothing else about
        #       its form is settled yet.
        pass

    def forward(self, y):
        # push the input up through the layers, recording what each layer emits
        # so the update rule can use those values
        activations = [y]
        for W, b in zip(self.W, self.b):
            x = activations[-1] @ W + b
            y = self.unit(x)
            activations.append(y)
        return activations

    def weight_updates(self, activations, target):
        # TODO: the per-weight update rule we need to design.
        #       For an OUTPUT unit a target is supplied and the single-layer
        #       delta rule applies; for an INTERIOR unit no target is given,
        #       and how to set its weight change from the output error alone
        #       is the open problem. Return one update array per weight matrix.
        pass


def train(net, data, lr, n_sweeps):
    for _ in range(n_sweeps):
        for inp, target in data:                 # one sweep through the patterns
            activations = net.forward(inp)        # forward pass
            dW, db = net.weight_updates(activations, target)
            for k in range(len(net.W)):           # apply the update rule
                net.W[k] -= lr * dW[k]
                net.b[k] -= lr * db[k]
```

The forward pass and the update loop are fixed; `unit` and `weight_updates` are the two slots the
method fills in.
