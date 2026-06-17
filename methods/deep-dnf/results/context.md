# Context: learning Boolean functions with neural networks (circa 2018)

## Research question

We are handed examples `(x, y)` where `x ∈ {0,1}^n` is a Boolean input vector and
`y ∈ {0,1}` is the value of some unknown Boolean function of `x`. We want a neural model
that learns this function from gradient descent. Two things are wanted at once, and they
pull in different directions. The first is *accuracy*: the model must actually fit the
target function, including the awkward functions (parity, deeply-nested AND/OR) that small
networks famously struggle with, and it must do so when training samples are few and the
input distribution is skewed rather than a fair coin. The second is *legibility*: after
training we want to read the learned function back out as an explicit logical expression —
a list of conjunctions OR'd together, with each conjunction naming exactly which variables
(and in which polarity) it depends on — so a human can inspect, verify, and trust it.

A standard multilayer perceptron achieves neither cleanly. Even when it fits, the function
it learned is smeared across thousands of real-valued weights and biases with no procedure
to decode "which variables, in which combination" — the logic is implicit. And on certain
target functions it does not even fit reliably. The precise problem is to design a
*differentiable* model whose parameters, once trained by ordinary gradient descent, *are*
the logical formula — so that fitting the data and recovering an interpretable Boolean
expression are the same act.

## Background

It has long been known that neural networks can in principle represent Boolean functions.
A single linear-threshold unit (a perceptron with a bias and a hard threshold) computes an
AND or an OR of its inputs by choosing the weights to 0/1 and setting the bias to the count
threshold: for `x_1 ∧ x_2 ∧ x_3` set the three weights to 1 and the bias to `-2`, then pass
through a step nonlinearity. Minsky & Papert (1969) established the structural ceiling of
this: a single such unit cannot represent XOR (the classic non-linearly-separable function),
and more generally cannot represent every Boolean dependence. The resolution since then has
been depth — stack threshold units across hidden layers and any Boolean function becomes
representable.

The working extension of that idea is the additive multilayer network with bias terms and a
sharp activation. Steinbach & Kohut (2002) study exactly this: a model of Boolean functions
built from successive layers of `η(Σ_j w_j x_j + b)` units, with `η` a sigmoid/sign-type
nonlinearity, and show that with a suitable multi-layer arrangement and proper activations
any Boolean function can be learned. This is the practical state of the art for "neural
network that learns a Boolean function." Several recurring frames sit around it:

- **The continuous relaxation of Boolean algebra.** To do gradient descent at all you extend
  the truth values from `{0,1}` to the interval `[0,1]` and pick continuous surrogates for
  the connectives that agree with the truth table at the corners. The product family is the
  standard such relaxation: `NOT x = 1 - x`, `x AND y = x·y`, and, by De Morgan,
  `x OR y = 1 - (1-x)(1-y)`. At `x,y ∈ {0,1}` these reproduce the Boolean tables exactly,
  and for intermediate values they are smooth and differentiable. The disjunction form
  `1 - Π(1 - p_i)` is also the classical *noisy-OR* of probabilistic models: the probability
  that at least one of several independent causes fires.

- **Soft logic learned by gradient descent.** A line of neuro-symbolic work makes logical
  inference differentiable so it can be trained end-to-end. Logic Tensor Networks (Serafini
  & Garcez 2016) give first-order logic a real-valued (fuzzy) semantics and optimize
  satisfiability by gradient descent. Differentiable inductive logic programming (Evans &
  Grefenstette 2018) builds, from rule templates, a large set of candidate clauses, attaches
  a trainable weight to each, and runs differentiable forward-chaining deduction over a
  valuation vector of atom confidences, training the clause weights against a cross-entropy
  loss so the system tolerates noisy data. Earlier connectionist work (Hölldobler et al.
  1999; Bader et al. 2008; França et al. 2014) likewise approximates the semantics of logic
  programs with recurrent or feed-forward nets.

- **Subset selection in a differentiable model.** A separate tool from sequence modelling,
  pointer networks (Vinyals et al. 2015), uses a softmax attention to *select* elements of
  the input — a natural-looking mechanism if one wanted a layer to "pick which variables go
  into this clause."

- **Diagnostic observations about the additive design.** Two empirical facts about the
  additive-perceptron approach are load-bearing. First, on skewed input distributions it is
  fragile: when the training bits are drawn from a biased Bernoulli (e.g. `p=0.75`) rather
  than a fair coin, an additive MLP fitting a random DNF over 10-bit inputs keeps producing
  test errors and does not fully converge to the exact logical function even after long
  training, whereas at `p=0.5` it converges easily. Second, on XOR / parity over moderately
  large inputs (`n > 30`) the additive MLP fails to converge at all, sitting near 50% error.
  These are not the proposed method's results; they are observed limitations of the existing
  additive design that frame the problem.

## Baselines

These are the prior approaches a new Boolean-function learner is measured against and reacts
to.

**Additive multilayer perceptron with bias (Steinbach & Kohut 2002; Wasserman 1989).** Each
neuron computes `η(Σ_j w_j x_j + b)` with a sigmoid or sign nonlinearity; depth supplies the
expressivity Minsky & Papert showed a single layer lacks. Trained by gradient descent with a
sigmoid output and a cross-entropy / squared loss. It can represent any Boolean function in
principle. **Gaps:** (1) the learned function is *not decodable* — to express "this neuron
computes `x_1 ∧ x_2 ∧ x_3`" you must reverse-engineer a particular alignment of real-valued
weights against a bias threshold, and in a trained net the weights fluctuate and there is no
clean procedure to read off which variables, in which polarity, a unit actually depends on;
(2) it is *bias-dependent* — the count threshold lives in the bias term, which has to be
tuned per-fan-in and drifts during training, and this is exactly where the instability shows
up; (3) empirically it fails to fully converge on skewed-distribution DNF fitting and on
larger XOR/parity, as noted above.

**Single linear-threshold unit (perceptron).** The minimal AND/OR gate: weights to 1 on the
chosen inputs, bias as the threshold, step activation. **Gap:** Minsky & Papert — cannot
represent XOR or, generally, all Boolean dependence; not trainable by gradient descent
through the hard threshold.

**Differentiable ILP (Evans & Grefenstette 2018).** Generates candidate clauses from rule
templates, weights them, and performs differentiable deduction on a valuation vector,
trained with cross-entropy so it handles noise; recovers explicit rules. **Gap:** it is a
deduction engine over a templated clause set, with cost that grows steeply (quintic in the
problem size in the original analysis), restricting it to small problems; it is not a generic
differentiable AND/OR *layer* that can be stacked like an MLP and dropped into an arbitrary
architecture.

**Logic Tensor Networks (Serafini & Garcez 2016).** Real-valued/fuzzy semantics for
first-order logic, optimized for satisfiability by gradient descent. **Gap:** aimed at
grounding and satisfaction of given formulas rather than learning, from raw input/output
examples, a compact stack of AND/OR layers whose weights *are* the formula.

**Pointer-network-style subset selection (Vinyals et al. 2015).** A softmax over inputs to
select which elements participate. **Gap:** as a way to choose the literals of a clause it
requires knowing the clause size (the number of selected items) in advance, and a softmax
that must concentrate over a long input vector converges slowly when `n` is large.

**Gradient-boosted / bagged decision trees (Breiman 2001; gradient boosting).** The
classical strong baselines for Boolean/tabular data: ensembles of axis-aligned splits.
**Gap:** an ensemble of hundreds of trees is itself opaque as a single logical formula, and
these are non-differentiable, so they cannot be composed inside a larger differentiable
architecture or co-trained with one.

## Evaluation settings

The natural yardsticks for a Boolean-function learner, all pre-existing:

- **Random DNF over small inputs.** Draw a random target DNF over `n`-bit inputs (e.g.
  `n=10`), sample a training set of Boolean vectors with their labels, and measure error on a
  held-out set of fresh uniform samples. The input bit distribution is a knob: a fair coin
  `p=0.5` versus a skewed `p=0.75` Bernoulli, the skew being the harder, more realistic
  regime.
- **XOR / parity-N.** The canonical hard family. Fit XOR of an `n`-bit input and measure bit
  error as a function of training-set size, sweeping `n` (small `n` where everything works,
  vs. `n > 30` where additive MLPs collapse).
- **Families of random DNF at varying scale.** Targets parameterized by number of variables
  `n`, number of terms `s`, and term width `w`, including monotone (positive-literal-only)
  DNF and sparse juntas (the function depends on a small subset of the variables). Learner is
  fit on one fresh uniform training set and scored by held-out accuracy on a large fresh
  uniform sample, aggregated across families.
- **Optimizer / protocol.** Adam-style first-order optimization with a small learning rate
  (e.g. `1e-3`), minibatches, identical initialization across compared models, error counted
  on a held-out sample; metric is classification accuracy (or error count). No outcomes here
  — these are the settings only.

## Code framework

The model plugs into an ordinary supervised classification harness that already exists: a
data pipeline producing `(x, y)` tensors with `x ∈ {0,1}^n`, an optimizer (Adam), a binary
cross-entropy loss on a logit output, and a minibatch training loop that forwards, computes
the loss, backpropagates, and steps. What is *not* settled is the model module itself —
the layer design whose parameters are meant to encode the Boolean function is exactly the
thing to be invented, so it is left as an empty slot.

```python
import torch
from torch import nn


class BooleanFunctionModel(nn.Module):
    """A differentiable model that maps a Boolean input vector x in {0,1}^n to a
    scalar logit for P(y=1). The internal design — the layers and the meaning of
    their parameters — is exactly what we have to figure out."""

    def __init__(self, n_features: int, capacity: int):
        super().__init__()
        self.n_features = n_features
        self.capacity = capacity
        # TODO: the parameters and layers we will design go here.
        #       Whatever they are, training them by gradient descent should make
        #       the model compute the target Boolean function.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features), entries in [0,1]
        # TODO: map x to a scalar logit using the design above.
        raise NotImplementedError


def train(model, train_x, train_y, lr=1e-3, epochs=30, batch_size=512):
    """Generic supervised loop the model plugs into; nothing here is specific to
    the model's internal design."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    n = train_x.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_x[idx]).view(-1)
            loss = criterion(logits, train_y[idx])
            # TODO: any regularization implied by the model design is added here.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def predict(model, test_x) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return (model(test_x).view(-1) >= 0.0).long()
```

The single empty slot is `BooleanFunctionModel` — its layers, its parameters, and what
training those parameters is supposed to mean.
