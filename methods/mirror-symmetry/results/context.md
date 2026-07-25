## Problem framing

The key point of mirror symmetry is not just "two Calabi-Yau manifolds look different yet are physically equivalent." What actually changes the methodology is that it swaps the complex moduli of one Calabi-Yau for the complexified Kahler moduli of a mirror Calabi-Yau. That swap turns string duality in physics from merely two languages describing the same theory into a machine for generating mathematical predictions.

The core question that needs explaining is: why does this duality let us translate enumerative-geometry counts into period integrals and variations of Hodge structure.

## Physical and geometric setup

In the language of the topological string, the A-model is primarily about symplectic/Kahler data, while the B-model is primarily about complex-structure data. For a mirror pair `(X, X^vee)`, mirror symmetry asserts that the A-model of `X` is equivalent to the B-model of `X^vee`; so the complexified Kahler moduli on `X` get re-expressed via the complex moduli on `X^vee`.

This is exactly what "mirror" means: not finding a space that looks similar in shape, but finding a dual space that exchanges the two types of deformation parameters. Problems that on `X` belong to Kahler geometry and curve counting become, on `X^vee`, problems about families of complex structures, holomorphic volume forms, and their periods.

## Enumerative geometry

The typical task in enumerative geometry is counting rational curves on a Calabi-Yau threefold, or, in more modern language, computing Gromov-Witten invariants. The quintic threefold is a standard object of study, involving the counting of rational curves of arbitrary degree along with quantum corrections and multiple-cover contributions.

## Mirror translation

The B-model on the mirror side turns the counting problem into a period calculation. Given the family of complex structures of the mirror Calabi-Yau, take the holomorphic three-form `Omega` and study its integral `int_gamma Omega` over three-cycles in homology. These periods vary with the complex-structure parameters, forming a variation of Hodge structures, and satisfy Picard-Fuchs differential equations.

The computational route thus becomes: solve the Picard-Fuchs equation to get the periods, choose canonical coordinates near the large complex structure limit, use the mirror map to translate the mirror complex-structure parameter back into the Kahler parameter of the original space, and then read off the enumerative invariants from the series expansion of the Yukawa coupling or the prepotential. Curve counting is rewritten as a problem of differential equations, monodromy, and series expansion.

## Methodological significance

This is a genuine conceptual breakthrough, because it does not look for a stronger local counting technique within the original enumerative space; instead it uses dual geometry to change what counts as a computable object. The enumerative-geometry quantities on the A-model side correspond, on the B-model side, to quantities governed by the Hodge theory of the complex structure, computable via periods and the Picard-Fuchs system.

So the distinctive insight of mirror symmetry is treating string duality as a mathematical translator: the same physical quantity shows up on one side as an enumerative-geometry invariant, and on the other side as computable Hodge-theoretic data. It lets physical intuition predict concrete integers and generating functions.
