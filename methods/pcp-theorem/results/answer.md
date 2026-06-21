# The PCP Theorem

`NP = PCP(O(log n), O(1))`.

A verifier for inputs of length `n` uses `O(log n)` random bits, reads `O(1)` proof bits, accepts true statements with probability 1, and accepts every proof of a false statement with probability at most 1/2.

The key move is to make proof correctness locally testable. A classical NP witness can hide an error in one coordinate. The PCP construction rewrites the witness as a redundant algebraic object, principally low-degree polynomial evaluations over finite fields. Distance spreads errors; low-degree tests and arithmetization make the spread-out object checkable; proof composition recursively replaces long algebraic answers by locally checked claims about the verifier's small decision circuit.

The approximation connection is exact under the usual soundness-`1/2` normalization. A `(log n, O(1))` verifier gives one constant-arity constraint per random string: proof locations are variables, and accepting local views are satisfying assignments. The violated-constraint fraction is the verifier's rejection probability. Conversely, a gap-CSP reduction gives a verifier that samples one random constraint and checks it. Thus constant-query proof checking and NP-hardness of distinguishing `UNSAT(C) = 0` from `UNSAT(C) >= 1/2` are the same theorem.

ALMSS then converts the verifier to MAX-3SAT. For each random string, the accepting predicate on `q` queried bits is written as a CNF with at most `2^q` clauses of width at most `q`, then converted to 3-CNF with private auxiliary variables, giving at most `q 2^q` clauses per random string. In false instances, at least an `epsilon = 1/(q * 2^(q+1))` fraction of clauses is unsatisfied, so satisfiable formulas are separated from formulas with at most `(1 - epsilon)m` satisfiable clauses.

Grounding: AS for verifier definitions, `NP = PCP(log n, log n)`, nonadaptivity, composition, and optimality; ALMSS for `NP = PCP(c log n, q)`, MAX-3SAT gap, recursive proof checking, low-degree distance, and curve aggregation; FGLSS for the verifier-to-clique gap; Dinur for the clean PCP/gap-CSP equivalence; Arora's survey for the retrospective proof mindset.
