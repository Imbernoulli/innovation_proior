# Rice Theorem

## Problem

Given a program, is there a general algorithm that can determine whether the semantics it computes has some property? Rice's theorem answers: as long as the property is a nontrivial extensional property, no such algorithm exists.

Here extensional means the property depends only on the partial function the program computes or the language it recognizes, not on the program text, variable names, implementation tricks, or the execution trace over a finite number of steps. Nontrivial means some program semantics satisfy the property and some do not.

## Theorem

Let `P` be a nontrivial property on partial computable functions. There is no algorithm that always halts and is always correct, that can determine for an arbitrary program index `e` whether `phi_e` has property `P`.

Therefore, natural questions about program semantics, once they are neither always true nor always false, are generally undecidable: whether it accepts some input, whether it recognizes the empty language, whether it computes a total function, whether it computes a constant function, whether it never outputs some value, and so on.

## Proof Sketch

Take the everywhere-undefined function `bottom`. Because `P` is nontrivial, one can choose a partial computable function `g` such that `g` and `bottom` have opposite truth values under property `P`.

Given an arbitrary halting instance `(M, x)`, construct program `Q`:

```text
Q(y):
    simulate M(x)
    if the simulation halts:
        run G(y), where G computes g
```

If `M(x)` halts, `Q`'s semantics is `g`. If `M(x)` does not halt, `Q` never returns on any input, and its semantics is `bottom`. So any algorithm that could decide whether `Q` has property `P` could decide whether `M(x)` halts — a contradiction.

## Key Insight

The distinctive insight of Rice's theorem is that it unifies the decision of program-semantic properties into a reduction to the halting problem and undecidability, rather than re-analyzing each property case by case. It shows that undecidability is not a peculiarity of the notion of "halting", but a structure that every nontrivial semantic judgment carries in common.

"Every nontrivial extensional property is undecidable" reveals the boundary of computational semantics: a program's true input-output behavior can encode an arbitrary halting instance, and so there is no complete, always-halting, semantic decider that is correct for all programs. Automated program analysis can only work within this boundary: doing syntactic checks, bounded-execution checks, restricted-language analysis, or giving incomplete but useful approximations.

## Boundary

Rice's theorem does not forbid all program analysis. It does not cover syntactic or bounded-execution properties such as "does the source code contain a certain token" or "does it halt within 100 steps", nor does it deny that specific programs, specific sublanguages, or conservative static analyses can be handled effectively.

The boundary it draws is more precise: as soon as a problem requires deciding the true semantics for all programs, and that semantic property really does distinguish different partial functions, the decider would be powerful enough to solve the halting problem. Hence general semantic decision is uncomputable.
