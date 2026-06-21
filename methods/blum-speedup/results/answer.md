# Blum Speedup Theorem

Blum's speedup theorem says that the search for a best algorithm is not always a well-founded search. Under any Blum complexity measure, and for any chosen total computable speedup function, there exists a total computable boolean function `f` such that every program computing `f` can be replaced by another program computing the same `f` that is eventually much faster in the prescribed sense.

In one standard formulation: for a Blum complexity measure with program costs `Phi_i`, and for any total computable two-argument function `r`, there is a total computable predicate `f` such that for every program index `i` computing `f`, there is another index `j` computing `f` with

`Phi_i(n) > r(n, Phi_j(n))`

for all but finitely many inputs `n`.

The unique insight is the absence of an asymptotically optimal algorithm. This is stronger than saying that some algorithms can be improved. The theorem constructs computable problems where every algorithm remains nonfinal: once an implementation is proposed, another implementation exists that computes the same function and beats it eventually by a computably specified margin. Reapplying the theorem gives an infinite chain of improvements.

That makes Blum speedup a structural counterexample to the intuition that every problem has a "best" algorithm waiting to be found. For many natural problems, optimal algorithms and tight lower bounds are meaningful. But computability alone does not guarantee such a bottom point. In the general theory, complexity can be a hierarchy of endlessly improvable implementations rather than a single asymptotic optimum.

The theorem therefore separates two ideas that are often conflated: a problem can be perfectly computable, yet still lack a stable best complexity among its programs. Algorithm design may look like convergence for many concrete tasks, but Blum speedup shows that convergence is not built into the structure of computation itself.
