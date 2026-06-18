# Context: learning Boolean formulas with differentiable models

## Problem setting

The task is supervised learning over binary examples. Each input is a vector
`x in {0,1}^n`, each label is a Boolean value, and the target is an unknown
Boolean function of the input bits. The useful model must satisfy two demands at
once: it should be trainable by ordinary gradient methods, and the trained
parameters should be readable as a symbolic formula rather than as an opaque
collection of real-valued weights.

The desired symbolic form is the usual rule form: a predicate is true when one
of several clauses is true, and a clause is true when all of its selected
literals are true. In propositional language this is a disjunction of
conjunctions. In logic-programming language it is the shape of several rules
with the same head.

## Additive neural baseline

A single threshold neuron can implement simple gates if it is allowed a bias.
For example, a unit with weights on selected inputs and a count threshold can
act like an AND or an OR. Depth then supplies more expressive power; the
classical perceptron limitation for XOR is avoided by moving beyond one
threshold layer.

That construction is a poor fit for rule extraction. The bias is the threshold
that decides how many selected inputs are enough, and the same Boolean behavior
can be represented by many scaled or shifted real-valued parameter settings.
After training, there is no canonical way to decide which weights are genuinely
part of a clause and which are small numerical residue. The formula is not
stored as a set of crisp include/exclude decisions.

There is also an optimization concern. In small Boolean tasks, additive MLPs may
fit easy distributions, but skewed bit distributions and parity-like functions
are standard stress tests because the learned threshold surfaces can fluctuate
or fail to settle into the exact Boolean rule.

## Logic-programming pressure

Inductive logic programming shows why explicit rules matter. A learned program
can generalize from few examples, can be inspected, and can be reused as a
program. Modern differentiable ILP systems keep the gradient-based training
benefit by relaxing discrete rule choices to continuous weights, then running
forward-chaining-style inference over fuzzy truth values.

Those systems still need a representation of candidate rules. Some generate a
large library of clauses from templates and learn weights over the generated
clauses. This keeps rule outputs explicit, but the clause library can become
large, and the learner is constrained by the template shape chosen in advance.

## Continuous Boolean algebra

A common relaxation embeds Boolean values in `[0,1]` and uses fuzzy connectives
that agree with Boolean truth tables at the corners. The product family is the
basic example:

```text
NOT x = 1 - x
x AND y = x * y
x OR y = 1 - (1 - x)(1 - y)
```

These equations preserve the Boolean cases for `x,y in {0,1}` while remaining
differentiable on soft truth values. They also connect to the noisy-OR form
used when several independent soft causes may make an output true.

## Evaluation frame

The natural experimental frame is to compare a differentiable rule learner with
an additive MLP on Boolean functions. One setting samples random Boolean
formulas over ten input bits and changes the Bernoulli parameter of the input
bits from fair (`p=0.5`) to skewed (`p=0.75`). A second setting uses parity or
XOR, where a compact special-purpose Boolean construction is possible but a
large disjunctive expansion would be expensive.

The implementation slot is therefore narrow: define a neural module whose
parameters behave like clause-membership flags, train those parameters with
Adam and cross-entropy on fuzzy truth values, and recover a formula by reading
which memberships converged near one.
