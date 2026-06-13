I am starting from the scaffold's empty graph — every variable a node, no edges, the estimator that
recovers nothing — and I want the first real rung: a method that turns the observational data into a
CPDAG at all. The honest constraint frames everything. I have a data matrix `X` from a linear
Gaussian SEM, no interventions and no variable order, so all I can ask the data is whether two
variables are independent, possibly after conditioning on others. From those yes/no answers I must
manufacture a causal graph. And I cannot recover a single DAG, ever: two DAGs that entail the same
conditional independences are identical in everything I can measure, so the target is the
equivalence class of all DAGs sharing the observed independence facts, returned compactly as a
CPDAG — an edge wherever two variables are directly connected in every member, an arrowhead wherever
every member agrees on direction, an undirected edge marking genuine ignorance. The cleanest way to
get there from raw independence queries is the constraint-based route, and the canonical constraint
method is PC. That is the natural floor to put down first: it asks the data only CI questions, it
has an asymptotic correctness theorem, and it maps directly onto the one function I am allowed to
edit.

What ties the data to the graph is two assumptions I lean on the whole way. The Markov condition
says the graph's factorization holds — every variable is independent of its non-descendants given
its parents — so every independence the graph asserts is real. Faithfulness is the converse: all and
only the distribution's independences are the ones the graph entails. I need faithfulness precisely
because I am running the inference backwards, from independences to structure. Markov alone would let
the distribution carry *extra* independences from coincidental parameter cancellations, and those
would be lies about the structure — I would read an accidental zero partial correlation as a missing
edge. With both assumptions I get a two-way bridge: a graphical d-separation in `G` holds exactly
when the matching conditional independence holds in the distribution. So from here on, when a CI test
says "independent given `S`," I read it as "d-separated given `S`" in the graph I am hunting, and
vice versa.

Now, what is even identifiable, because that tells me what to aim at and what to leave undirected.
Adjacency is fixed: if `X` and `Y` are directly connected, no conditioning set can block the
one-edge path, so under faithfulness they are dependent given every set; if they are not connected,
some set separates them. So the skeleton is identifiable. Arrowheads are the subtle part. Take a
length-two path `X — Y — Z` with `X`, `Z` non-adjacent. The non-collider shapes (`X → Y → Z`,
`X ← Y ← Z`, `X ← Y → Z`) are all independence-equivalent — conditioning on `Y` blocks the path in
every one — so the data cannot distinguish them and I must leave that orientation open. But the
collider `X → Y ← Z` is different: the path is blocked unless I condition on `Y` (or a descendant),
and conditioning on `Y` *opens* it — an independence signature no non-collider shares. So unshielded
colliders are identifiable, ordinary non-collider triples are not, and two DAGs entail the same
independences exactly when they share both skeleton and unshielded colliders. My identifiable target
is therefore precisely the skeleton, plus the unshielded colliders, plus whatever further
orientations those force by logic. That is the CPDAG, and PC is built to find exactly those three
things in three phases: thin to the skeleton, orient colliders, propagate.

Phase one is the skeleton, and it is the foundation. The cleanest test from faithfulness is: `X` and
`Y` are adjacent iff dependent given *every* subset of the others. So to delete `X — Y`, search all
subsets `S` of `V \ {X, Y}`; any one making them independent proves non-adjacency. That is correct
but it is exponential — `2^{p-2}` subsets per pair — and worse, for a true edge no subset ever
separates the pair, so the search never short-circuits; the worst case is the expected case. The
escape is the Markov-condition fact that if `X` and `Y` are non-adjacent with `Y` a non-descendant
of `X`, then `X` is independent of its non-descendants given its parents, so `X ⟂ Y | Pa(X)`, and
`Pa(X)` lives among the neighbors of `X`. I do not know the neighbors yet, but if I start from the
complete graph and only ever *remove* edges, then at every moment the current adjacency set of `X` is
a superset of its true neighbors, so it still contains `Pa(X)`. Conditioning on subsets of the
*current* adjacencies therefore always includes the separator I need — the circularity is benign.
And if I test small conditioning sets first and let them prune the graph before attempting large
ones, the size of conditioning sets I ever test is bounded by the graph's degree, not by `p`: cheap
and, in the continuous Gaussian case, statistically well-conditioned.

That last point is where this task diverges from the discrete textbook version, and it is a
deliberate choice I make for *this* substrate. The data here are continuous and Gaussian, so the
right CI test is not a contingency-table chi-squared but **Fisher's z**: under joint Gaussianity, `X`
and `Y` are conditionally independent given `S` iff their partial correlation `ρ_{XY·S}` is zero, and
the Fisher z-transform `z = ½ ln((1+r)/(1-r)) · √(n − |S| − 3)` is asymptotically standard normal
under the null, giving a p-value I threshold at `α`. This is exact for linear Gaussian SEMs — the
model class the benchmark generates — so I instantiate the CI oracle as `CIT(X, "fisherz")` rather
than the `"chisq"` test the discrete lineage uses. Everything else about the skeleton phase is the
standard PC machinery: I want the order-independent variant (PC-stable), which freezes each node's
adjacency set at the start of each conditioning level and draws all subsets from the frozen set, so
that deleting one edge mid-level does not change which tests other pairs run. With an oracle this
changes nothing; on sample data it makes the output skeleton independent of the column order, which
is the difference between a stable result and one that wobbles with permutations of the variables.
The harness exposes this exactly as `SkeletonDiscovery.skeleton_discovery(..., stable=True)`, and I
keep `α = 0.05` as the conventional default — not a true significance level, since the search fires
thousands of tests, but a sparsity knob.

Phase two orients unshielded colliders, and the separating sets recorded during phase one already
tell me which triples are colliders. For an unshielded triple `X — Y — Z`, I deleted `X — Z` because
some recorded set separated them. If `Y` is a non-collider on the length-two path, leaving `Y` out
keeps that path active, so any separator must include `Y`; if `Y` is a collider, the path is already
blocked unless I include `Y`, and including it opens the path with no other interior node to close
it, so any valid separator must leave `Y` out. The rule is therefore: orient `X → Y ← Z` iff `Y` is
absent from the stored separating information for `X, Z` — a lookup, no new tests. On sample data two
collider decisions can fight over a shared edge, so I want the conservative conflict policy that only
orients when it does not contradict an arrowhead already placed; in `causallearn` this is
`UCSepset.uc_sepset(cg_1, 2, ...)`, where the priority argument `2` preserves an existing opposite
orientation instead of overwriting it, preventing overwrite cascades and spurious bidirected edges.

Phase three propagates the forced orientations. After the colliders I have a partially directed
graph, and some undirected edges are forced even though no collider touched them — forced by two
facts I may not violate: the final graph must be acyclic, and it must contain no unshielded collider
I did not already find. Those give Meek's rules. R1: `A → B`, `B — C`, `A`/`C` non-adjacent ⟹
`B → C`, else a new collider at `B`. R2: `A → B → C`, `A — C` ⟹ `A → C`, else a cycle. R3: the kite
configuration where the wrong orientation would force a new collider through two applications of R2.
I close the partially directed graph under these rules until nothing more orients; `Meek.meek(cg_2,
...)` does exactly this and returns the maximally oriented CPDAG.

So the step-1 edit is the three phases as three calls into the harness, on the Fisher-z oracle. I am
clear-eyed about why this is only the floor, and the reason names what comes next. PC's every
decision is a single thresholded CI test, and on dense or high-paths graphs the partial correlation
of a real-but-faint edge — two near-cancelling paths, a near-deterministic relation — sits close to
zero, so a finite-sample Fisher-z test reports "independent" and I delete a true edge. Worse, an
early wrong deletion shrinks a neighborhood and leaves a different false edge un-testable, so errors
ramify, and a single mis-decided collider then spreads through the Meek closure. I expect this to
show up exactly as graph density and noise climb: the small sparse scenarios should be reachable, but
the larger and denser ones — where the count of near-cancellations grows with the number of paths —
should degrade, with adjacency recall falling first (real edges missed) and arrow recall right
behind it (colliders mis-decided on a wrong skeleton). That predicted failure mode — brittle
single-edge decisions on dense, faint-dependence graphs — is precisely what motivates leaving the
constraint-based view at step 2 for a search whose decisions are more robust to faint-but-real
dependences. The full scaffold module — the three-call PC pipeline on `fisherz` — is in the answer.
