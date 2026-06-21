## Research question

Kleene recursion theorem asks how far self-reference can be made legitimate inside a formal model of computation. The central question is not whether a programmer can write a string that prints itself, but whether every effective way of turning program descriptions into new behavior has a program that can obtain its own description and use that description as an ordinary input.

In its common second-recursion-theorem form: for every partial computable two-argument operation `Q(e, x)`, there is an index `p` such that `phi_p(x) ~= Q(p, x)` for all `x`. In Rogers fixed-point form: for every total computable program transformer `f`, there is an index `p` with `phi_p ~= phi_{f(p)}`. The equality is extensional equality of partial functions, not textual equality of source code.

## Background

Computability theory treats programs as data by assigning them natural-number indices. A universal evaluator `U(e, x)` simulates the program with index `e` on input `x`; writing `phi_e(x)` for `U(e, x)` turns one effective numbering into a uniform language for talking about all partial computable functions.

The other required tool is the `s-m-n` theorem, or parameter theorem. If a program has two inputs, there is an effective way to produce a new one-input program with the first input hard-coded. In notation, from an index `q` and a value `a`, a total computable function `s(q, a)` returns an index for the function `x |-> phi_q(a, x)`.

Together these facts make program text manipulable without leaving computation.

## Baselines

**Naive self-reference.** One can imagine a program referring to "my own source" as a primitive operation. That is a language feature, not a theorem.

**Quines.** A quine prints its own text. The recursion theorem is broader: the self-description can be passed into any computable context `Q`, so self-printing, self-recognition, recursive definitions, and fixed points of program transformers are all instances of the same construction.

**Diagonalization only.** Classical diagonal arguments also feed an object to its own code, often to prove impossibility or contradiction. Kleene's theorem uses a related self-application shape, but constructs a working program instead of deriving inconsistency.

**Textual fixed points.** A program need not be textually identical to `f(p)`. The theorem guarantees that `p` and `f(p)` compute the same partial function, meaning many different descriptions can name the same behavior.

## Evaluation settings

The key correctness criterion is extensional: for every input `x`, the constructed index `p` either diverges exactly when `Q(p, x)` diverges, or returns the same value when both halt. The construction is allowed to depend effectively on an index for `Q`, but it cannot use an oracle for halting, semantic equality, or "true self-knowledge."

The conceptual test cases are program transformations. Given a computable transformer that wraps, logs, simulates, delays, or rewrites a program, Rogers' fixed-point theorem says some program has unchanged behavior after that transformation. Given a two-input computable specification `Q(e, x)`, Kleene's form says there is a program whose own index fills the first slot.

The non-paradox condition is also part of evaluation. A valid explanation must identify where self-reference enters as finite syntax and effective specialization, not as a circular definition that must already know its own value.
