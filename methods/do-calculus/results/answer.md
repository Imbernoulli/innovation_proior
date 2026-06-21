# Do-Calculus: Causal Effects by Graphical Intervention Algebra

Do-calculus is a symbolic identification method for causal effects in graphical structural models. It treats an intervention `do(x)` as a change to the data-generating process: the structural equation for `X` is replaced by the value `x`, and the graph is modified by deleting arrows into `X`. The method then rewrites expressions involving interventions into ordinary observational probabilities whenever the graph licenses each rewrite.

Let `G_{\bar X}` denote the graph obtained from `G` by deleting arrows into `X`, and let `G_{\underline Z}` denote the graph obtained by deleting arrows out of `Z`. Let `Z(W)` be the subset of `Z` that are not ancestors of `W` in `G_{\bar X}`; Rule 3 deletes incoming arrows only for that subset, not for every member of `Z`.

The three rules are:

1. Insertion/deletion of observations:
   `P(y | do(x), z, w) = P(y | do(x), w)` if `(Y _||_ Z | X,W)` in `G_{\bar X}`.

2. Action/observation exchange:
   `P(y | do(x), do(z), w) = P(y | do(x), z, w)` if `(Y _||_ Z | X,W)` in `G_{\bar X,\underline Z}`.

3. Insertion/deletion of actions:
   `P(y | do(x), do(z), w) = P(y | do(x), w)` if `(Y _||_ Z | X,W)` in `G_{\bar X,\overline{Z(W)}}`.

Ordinary probability algebra is used between these rule applications. A derivation identifies a causal effect when these graph-indexed transformations remove all `do(...)` terms from the target expression, leaving only observational quantities.

The familiar back-door adjustment is a special case. If `Z` blocks every back-door path from `X` to `Y` and contains no descendant of `X`, then:

`P(y | do(x)) = sum_z P(y | x,z)P(z)`.

The front-door adjustment shows why the method is stronger than informal confounder adjustment. If `Z` intercepts all directed paths from `X` to `Y`, there is no unblocked back-door path from `X` to `Z`, and all back-door paths from `Z` to `Y` are blocked by `X`, then:

`P(y | do(x)) = sum_z P(z | x) sum_{x'} P(y | z,x')P(x')`.

This derivation succeeds even when `X` and `Y` share an unobserved cause, because the intervention target is decomposed into pieces whose required action-observation exchanges are licensed in different transformed graphs.

The method's output is therefore not "adjust for the right covariates." It is a proof that an interventional distribution follows from the observational distribution plus the causal graph, or a failure to find such a proof under the stated graph. Pearl's 1995 paper left open whether the three rules derive every identifiable effect; later completeness results show that, with ordinary probability manipulation, they are sufficient for every identifiable causal effect in the causal Bayesian-network setting considered by those proofs.
