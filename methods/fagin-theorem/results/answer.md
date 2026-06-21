# Fagin's Theorem

For finite structures under standard encodings, nondeterministic polynomial time is exactly existential second-order definability:

`NP = SO-exists`.

Equivalently, for every nonempty finite vocabulary `S` and every isomorphism-closed class `A` of finite `S`-structures,

`A` is a generalized spectrum if and only if `E(A)` is in `NP`,

where `E(A)` is the set of standard string encodings of structures in `A`.

In logical form, the defining sentence has the shape

`exists R1 ... exists Rk phi`,

where the `Ri` are auxiliary relation variables of fixed arity and `phi` is first-order over the original structure plus those auxiliary relations.

The forward direction is direct. A nondeterministic polynomial-time machine guesses the interpretations of `R1,...,Rk` and then evaluates the fixed first-order formula `phi` on the expanded finite structure.

The reverse direction encodes an accepting polynomial-time nondeterministic computation as finite relations. If the machine runs in time `n^k`, then `k`-tuples of elements index enough time instants and tape cells. Existentially quantified relations record the contents of each cell at each time and, when useful, the nondeterministic choices. The first-order formula checks the initial configuration, well-formedness, local transition consistency, and acceptance.

The theorem's content is therefore not merely that NP problems have certificates. It identifies the certificate with finite relational structure: existential second-order quantification is the logical form of nondeterministic guessing, and first-order verification is the logical form of polynomial local checking.

The empty-vocabulary case is a separate classical spectra phenomenon. There the input is a cardinality rather than an encoded finite structure, and the corresponding complexity scale is nondeterministic exponential time rather than ordinary `NP`.
