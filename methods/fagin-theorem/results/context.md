## A Machine-Bound Class

By the early 1970s, nondeterministic polynomial time has a clear operational meaning. A problem belongs to the class when a nondeterministic Turing machine accepts exactly the yes-instances within a polynomial bound, or equivalently when a bounded search branch can be checked efficiently. Cook's theorem and Karp's reductions make this class central by showing that many combinatorial and logical problems share the same polynomial difficulty.

This definition is powerful, but it is tied to machines, encodings, clocks, and reductions. It does not yet say what the class looks like as a mathematical language for finite objects.

## Finite Structures

Many decision problems can be treated as classes of finite structures. A graph is a finite universe with an edge relation; a database instance is a finite universe with named relations; an encoded string can be represented by an ordered finite structure with unary predicates for symbols. The intended property should not depend on the accidental names assigned to elements, so the relevant classes are closed under isomorphism.

Once problems are viewed this way, the natural descriptive question changes. Instead of asking only which machine accepts the encodings, one can ask which logical resources define the class of structures itself.

## Spectra

Classical spectra ask which finite cardinalities occur as model sizes of first-order sentences. This connects logic to finite existence, but only indirectly to ordinary input problems: the object being recognized is a number, not a whole finite structure with relation tables already present.

Finite model theory also has a broader projective viewpoint. One may expand a structure by auxiliary predicates and then test a first-order condition on the expansion. This keeps the final property about the original structure while allowing the definition to quantify over extra finite relations.

## The Size Mismatch

The distinction between sizes and structures matters for complexity. A model with `n` elements may carry relation tables of polynomial size in `n`, while the binary notation for the number `n` has only logarithmic length. A brute-force search over possible relations can therefore look exponential in the number-input setting but polynomial relative to an encoded structure with relation tables.

Any bridge from finite definability to nondeterministic polynomial time has to account for this encoding difference rather than treating all finite existence questions alike.

## The Desired Bridge

The unresolved descriptive problem is to match the computational act of bounded nondeterministic guessing with a logical act over finite structures. The candidate bridge must let a definition add a finite witness object, verify it by local first-order constraints, and remain invariant under isomorphism of the original input.

If such a bridge exists, a machine-defined complexity class can be recognized as a logical expressiveness class. That would turn questions about time and nondeterminism into questions about quantifiers, auxiliary relations, and finite definability.
