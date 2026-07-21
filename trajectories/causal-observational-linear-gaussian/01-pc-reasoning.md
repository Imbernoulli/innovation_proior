I am starting from the given empty graph — every variable a node, no edges, the estimator that
recovers nothing — and I want a first real method: one that turns the observational data into a
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
edit. It is also the most transparent — every deletion, every arrowhead traces back to one
identifiable independence fact, so whatever it gets wrong will be legible as a specific broken fact,
which is the diagnostic I want before I reach for anything cleverer.

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
vice versa. Faithfulness is an idealization about the population distribution, and my data are finite
samples of it — that gap is where this method's finite-sample failures will live, so I want the
assumption stated cleanly enough to watch where the samples stop honoring it.

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
separates the pair, so the search never short-circuits; the worst case is the expected case. On the
fifty-node scenarios `2^{48}` is not a rounding error I can absorb, it is a wall. The escape is the
Markov-condition fact that if `X` and `Y` are non-adjacent with `Y` a non-descendant of `X`, then `X`
is independent of its non-descendants given its parents, so `X ⟂ Y | Pa(X)`, and `Pa(X)` lives among
the neighbors of `X`. I do not know the neighbors yet, but if I start from the complete graph and
only ever *remove* edges, then at every moment the current adjacency set of `X` is a superset of its
true neighbors, so it still contains `Pa(X)`. Conditioning on subsets of the *current* adjacencies
therefore always includes the separator I need — the circularity is benign, because I only ever
delete an edge on a *witnessed* separation, which in the population can never remove a true edge, so
the true neighbors survive every round. And if I test small conditioning sets first and let them
prune the graph before attempting large ones, the size of conditioning sets I ever test is bounded by
the graph's degree, not by `p`. On the sparse ER graphs that degree is a handful; on the scale-free
graphs the hubs push it up, and that is where the cost concentrates: a degree-ten hub walks one
incident pair up through `∑_ℓ C(9, ℓ) ≈ 2^9 ≈ 500` conditioning subsets before the level bound saves
me, so the total work is polynomial in `p` but exponential in the local degree — and the densest
graph is where both the compute and, as I will see, the statistics get strained.

That last point is where this task diverges from the discrete textbook version, and it is a
deliberate choice I make for *this* substrate. The data here are continuous and Gaussian, so the
right CI test is not a contingency-table chi-squared but **Fisher's z**: under joint Gaussianity, `X`
and `Y` are conditionally independent given `S` iff their partial correlation `ρ_{XY·S}` is zero. The
Fisher z-transform `atanh(r)` variance-stabilizes the estimated partial correlation to
asymptotic normality with variance `1/(n − |S| − 3)`, so the standardized statistic
`atanh(r) · √(n − |S| − 3)` is asymptotically standard normal under the null and I threshold its
two-sided p-value at `α`. The `−|S|` in that count is the cost of partialling — a partial correlation
given `S` is the ordinary correlation of the residuals after regressing out `S`, and each regressor
consumes one degree of freedom — so as `|S|` grows the effective sample size `n − |S|` shrinks and
the test loses power. That is a second, statistical reason the degree-bounded conditioning of phase
one matters: the moment I am forced to condition on a large set (a hub, a dense neighborhood) I am
simultaneously running out of samples to do it with. The test is exact for linear Gaussian SEMs — the
model class the benchmark generates — so I instantiate the CI oracle as `CIT(X, "fisherz")` rather
than the `"chisq"` test the discrete lineage uses. A nonparametric kernel test (`kci`) would drop the
Gaussian assumption, but that assumption is true by construction here, not an approximation I am
making, and `kci` costs on the order of `n³` per call — buying robustness against a misspecification
that cannot occur, at a compute bill I will not pay at `n = 2000` over thousands of tests.

With the test fixed I can turn the sampling variance into a prediction about which scenarios PC
should survive, because the method's finite-sample weakness is that a *real* edge whose partial
correlation is small gets reported as zero and deleted. The two-sided test at `α = 0.05` rejects when
`|atanh(r)| · √(n − |S| − 3) > 1.96`, so the smallest partial correlation I can reliably detect is
roughly `ρ_min ≈ 1.96 / √(n − |S| − 3)`. Reading `n` off the scenario table for a representative
small `|S|`, this floor runs from about `0.044` on SF50 (`n = 2000`, the most samples per edge) up to
about `0.10` on ER20-Noisy (`n = 400`). But detectability per edge is only half the story; the other
half is how many real edges actually sit below that floor, and that count is driven by density and
noise, both of which push edge partial correlations *down*. Density does it through paths — on a dense
graph a direct edge `X → Y` competes with many indirect routes, and conditioning on the wrong
intermediate set drags the true edge's partial correlation toward zero through near-cancellation.
Noise does it directly — inflating `ε`'s standard deviation from `1.0` to `2.5` swells the residual
variance in the denominator of every partial correlation. ER20-Noisy suffers on both counts at once:
the highest detection floor *and* the noise pushing every edge toward it, on a denser-than-average ER
graph with the fewest samples. So my ranked expectation is that the two small sparse graphs are
reachable, the two scale-free graphs degrade through path-driven near-cancellation, and ER20-Noisy is
the worst case; the `SHD` and adjacency-recall columns will tell me whether that ordering is right.

Everything else about the skeleton phase is the standard PC machinery: I want the order-independent
variant (PC-stable), which freezes each node's adjacency set at the start of each conditioning level
and draws all subsets from the frozen set, so that deleting one edge mid-level does not change which
tests other pairs run. With an oracle this changes nothing; on sample data it makes the output
skeleton independent of the column order. The harness exposes this as
`SkeletonDiscovery.skeleton_discovery(..., stable=True)`, and I keep `α = 0.05` as the conventional
default. It is not functioning as a true significance level — the search fires thousands of tests, so
the family-wise error is nowhere near `0.05` — and that compounding is itself a second density
penalty: a true edge on a dense graph is tested against many candidate separating subsets, each an
independent chance to spuriously separate it, so the probability that *some* subset wrongly deletes a
faint true edge grows with degree. Raising `α` would delete fewer edges but admit more false ones;
lowering it does the reverse. Tuning it against the scenarios would be leaning on the dataset-specific
constants the task forbids, so I treat `α = 0.05` as a fixed sparsity knob and let the recall
consequences fall where the arithmetic above says they will.

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
This phase inherits every skeleton error: if phase one deleted a true edge, an unshielded triple that
should have been shielded becomes eligible for a collider verdict it does not deserve, so a single
missing edge does not just cost adjacency, it manufactures a wrong arrowhead — which is why I expect
the arrow metrics to sit *below* the adjacency ones, not merely track them.

Phase three propagates the forced orientations. After the colliders I have a partially directed
graph, and some undirected edges are forced even though no collider touched them — forced by two
facts I may not violate: the final graph must be acyclic, and it must contain no unshielded collider
I did not already find. Those give Meek's rules. R1: `A → B`, `B — C`, `A`/`C` non-adjacent ⟹
`B → C`, else a new collider at `B`. R2: `A → B → C`, `A — C` ⟹ `A → C`, else a cycle. R3: the kite
configuration where the wrong orientation would force a new collider through two applications of R2.
I close the partially directed graph under these rules until nothing more orients; `Meek.meek(cg_2,
...)` does exactly this and returns the maximally oriented CPDAG. Meek is deterministic logic, so it
adds no error of its own — but it is a *propagator*, and that cuts both ways: a correct collider
orients a whole neighborhood for free, and a wrong collider spreads that one mistake across the
neighborhood just as efficiently.

That leaves why PC and not one of its relatives. FCI is the constraint method built for latent
confounders; it would output a PAG with circle marks encoding "could be a hidden common cause here."
But the generator is explicit that every variable is observed and the noise is independent additive —
there are no latents — so FCI's extra machinery would buy robustness against a confounding that cannot
occur while costing me orientation certainty. A score-based climb like GES is legitimate prior art
here too, but I want the constraint-based version first precisely for the legibility above: every
decision is a single identifiable independence fact rather than a diffuse optimization artifact. So my
estimator is the three phases as three calls on the Fisher-z oracle:
`skeleton_discovery(..., stable=True)`, then `uc_sepset(cg_1, 2, ...)`, then `Meek.meek(cg_2, ...)`,
returning `cg.G`.

And I already know where this floor gives out: every PC decision is one thresholded CI test, so
whenever a real edge's partial correlation falls below the `1.96/√(n − |S| − 3)` floor — around `0.10`
on ER20-Noisy, and reachable on the scale-free graphs when near-cancelling paths drag it down — the
edge is deleted and never revisited, and that irreversible local deletion then ramifies through the
neighborhood and the Meek closure. So the next move, when this fails on the dense high-noise rows, is a
search whose decisions compare whole-model fits rather than thresholding one statistic. The full PC
module is in the answer.
