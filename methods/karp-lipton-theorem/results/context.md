## Research question

What should we conclude from the hypothesis that `NP` has polynomial-size circuits?

If `SAT` had a uniform polynomial-time algorithm, then `P = NP`. The subtler question is: suppose that for every input length there merely exists a polynomial-size Boolean circuit for `SAT`, with no requirement that one polynomial-time machine can construct those circuits. This is a nonuniform assumption, usually written `NP subseteq P/poly`. Does such apparently weak per-length help have any structural consequences for uniform complexity classes?

## Mathematical setting

`P/poly` can be viewed either as polynomial-size circuit families or as polynomial-time computation with polynomial-length advice depending only on the input length. The advice may encode a circuit for all inputs of that length. This makes `P/poly` much broader than `P`: it permits length-by-length information that the algorithm need not discover.

The polynomial hierarchy `PH` measures bounded alternation of polynomially long existential and universal quantifiers. A typical `Pi_2^p` language has the form

`z in L iff forall x exists y R(z, x, y)`,

where `R` is computable in polynomial time and `x, y` range over polynomial-length strings.

## Baseline intuition

The direct circuit-lower-bound approach is to prove `SAT notin P/poly`. That would imply `P != NP`, because every language in `P` has polynomial-size circuits.

A related structural tool is SAT self-reducibility: a satisfiability decision procedure can be extended to a witness-producing procedure by repeatedly fixing a variable, querying whether the restricted formula remains satisfiable, and recovering a satisfying assignment when one exists.
