## A Machine-Bound Class

By the early 1970s, nondeterministic polynomial time has a clear operational meaning. A problem belongs to the class when a nondeterministic Turing machine accepts exactly the yes-instances within a polynomial bound, or equivalently when a bounded search branch can be checked efficiently. Cook's theorem and Karp's reductions make this class central by showing that many combinatorial and logical problems share the same polynomial difficulty.

This definition is stated in terms of machines, encodings, clocks, and reductions.

## Finite Structures

Many decision problems can be treated as classes of finite structures. A graph is a finite universe with an edge relation; a database instance is a finite universe with named relations; an encoded string can be represented by an ordered finite structure with unary predicates for symbols. The intended property should not depend on the accidental names assigned to elements, so the relevant classes are closed under isomorphism.

Viewed this way, a problem becomes a class of finite structures, and one can ask which logical resources define that class.

## Spectra

Classical spectra ask which finite cardinalities occur as model sizes of first-order sentences. This connects logic to finite existence: the object being recognized is a number, the cardinality of a model.

Finite model theory also has a broader projective viewpoint. One may expand a structure by auxiliary predicates and then test a first-order condition on the expansion. This keeps the final property about the original structure while allowing the definition to quantify over extra finite relations.

## The Size Mismatch

Sizes and structures relate to complexity differently through their encodings. A model with `n` elements may carry relation tables of polynomial size in `n`, while the binary notation for the number `n` has only logarithmic length. A search over possible relations is therefore exponential in the number-input setting but polynomial relative to an encoded structure with relation tables.

## The Descriptive Question

The standing question is how the machine-defined complexity class relates to logical definability over finite structures: which logical resources over a finite structure, invariant under isomorphism of its universe, capture nondeterministic polynomial-time recognition of the structure's standard encoding. Answering it would connect a class defined by time and nondeterminism to one defined by quantifiers, auxiliary relations, and finite definability.
