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
edit. It is also the honest first rung precisely because it is the most transparent — every
deletion, every arrowhead traces back to one identifiable independence fact, so whatever it gets
wrong will be legible as a specific broken fact, which is the diagnostic I want before I reach for
anything cleverer.

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
vice versa. It is worth naming now, at the top, that faithfulness is an *idealization* about the
population distribution, and the data I get are finite samples of it — that gap is where every one of
this method's future failures will live, and I want to have the assumption stated cleanly so I can
watch exactly where the finite sample stops honoring it.

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

Before I trust that collider story I want to see the numbers come out, because the whole orientation
phase rides on one arithmetic fact — that conditioning on the middle node flips the sign of the
signal at a collider and kills it at a chain. Take the collider concretely: `X, Z ~ N(0,1)`
independent, `Y = X + Z + ε` with `ε ~ N(0,1)`. Marginally `Cov(X, Z) = 0`, so a CI test at the
empty conditioning set reports independence — I would delete `X — Z` and record its separating set as
`∅`. The partial correlation given `Y` is `ρ_{XZ·Y} = (ρ_{XZ} − ρ_{XY} ρ_{ZY}) / √((1 − ρ_{XY}²)(1 −
ρ_{ZY}²))`; here `ρ_{XY} = ρ_{ZY} = 1/√3` and `ρ_{XZ} = 0`, giving `(0 − 1/3)/(2/3) = −0.5`. A
partial correlation of −0.5 is enormous — conditioning on the collision node manufactures a strong
dependence out of two marginally independent variables. Now the chain: `X ~ N(0,1)`, `Y = X + ε₁`,
`Z = Y + ε₂`. Marginally `ρ_{XZ} = 1/√3 ≠ 0` — dependent, as a chain should be. But `ρ_{XY} = 1/√2`,
`ρ_{YZ} = 2/√6`, and their product is `2/√12 = 1/√3`, exactly `ρ_{XZ}`, so `ρ_{XZ·Y} = (1/√3 −
1/√3)/… = 0`. The chain's dependence vanishes when I condition on `Y`; I delete `X — Z` and record
its separating set as `{Y}`. Same skeleton in both cases — `X — Y`, `Y — Z` — but the collider left
`Y` *out* of the separating set and the chain put `Y` *in* it. That single bit, whether the middle
node is in the recorded separator, is the entire identifiable orientation signal, and I have just
watched it come out as −0.5 versus 0. This is why the collider rule can be a lookup rather than a new
test.

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
therefore always includes the separator I need — the circularity is benign, and it is worth being
precise about why: the superset invariant is preserved by the algorithm's own contract, because I
only ever delete an edge when I have *witnessed* a separation, and a witnessed separation of a
non-adjacent pair can never delete a true edge in the population, so the true neighbors survive every
round. And if I test small conditioning sets first and let them prune the graph before attempting
large ones, the size of conditioning sets I ever test is bounded by the graph's degree, not by `p`.
On the sparse ER graphs that degree is a handful; on the scale-free graphs the hubs push it up, and
that is exactly where the cost concentrates. If a hub has degree ten, testing one of its incident
pairs walks up through `∑_ℓ C(9, ℓ)` conditioning subsets — on the order of `2^9 ≈ 500` tests for
that one pair before the level bound saves me — so the total work is polynomial in `p` but
exponential in the local degree, and the densest graph is where both the compute and, as I will see,
the statistics get strained.

That last point is where this task diverges from the discrete textbook version, and it is a
deliberate choice I make for *this* substrate. The data here are continuous and Gaussian, so the
right CI test is not a contingency-table chi-squared but **Fisher's z**: under joint Gaussianity, `X`
and `Y` are conditionally independent given `S` iff their partial correlation `ρ_{XY·S}` is zero. The
sampling distribution of the estimated partial correlation is awkward, but the Fisher z-transform
`z = ½ ln((1+r)/(1−r))` variance-stabilizes it — `atanh(r)` is asymptotically normal with variance
`1/(n − |S| − 3)`, so the standardized statistic `atanh(r) · √(n − |S| − 3)` is asymptotically
standard normal under the null, and I threshold its two-sided p-value at `α`. I want to be sure I
believe that `−|S| − 3`. The `−3` is the classical constant for the marginal correlation of two
Gaussians — set `|S| = 0` and the variance is `1/(n − 3)`, which is the textbook result, so the
formula degenerates correctly at the base case. The `−|S|` is the cost of partialling: a partial
correlation given `S` is the ordinary correlation of the residuals after regressing `X` and `Y` on
`S`, and each of the `|S|` regressors consumes one degree of freedom from the residual, so the
effective sample size is `n − |S|`. That gives me a sanity check with teeth: as `|S| → n − 3` the
degrees of freedom vanish and the test loses all power, which is a second, statistical reason the
degree-bounded conditioning of phase one matters — the moment I am forced to condition on a large set
(a hub, a dense neighborhood) I am simultaneously running out of samples to do it with. This test is
exact for linear Gaussian SEMs — the model class the benchmark generates — so I instantiate the CI
oracle as `CIT(X, "fisherz")` rather than the `"chisq"` test the discrete lineage uses. I do
consider the nonparametric kernel option, `kci`, which would not lean on the Gaussian assumption; but
it is a wrong trade here on two counts. The Gaussian assumption is not an approximation I am making,
it is *true by construction* of this generator, so a nonparametric test buys robustness against a
misspecification that cannot occur; and a kernel test costs on the order of `n³` per call, which at
`n = 2000` and thousands of tests is a compute bill I would be paying for nothing. Fisher-z is both
exact and cheap here, so it is unambiguously the test.

With the test fixed I can turn the sampling variance into a prediction about which scenarios PC
should survive, because the whole method's finite-sample weakness is that a *real* edge whose partial
correlation is small will be reported as zero and deleted. The threshold is explicit: the two-sided
test at `α = 0.05` rejects the null when `|atanh(r)| · √(n − |S| − 3) > 1.96`, so the smallest
partial correlation I can reliably detect is roughly `ρ_min ≈ tanh(1.96 / √(n − |S| − 3))`, and for
these magnitudes that is essentially `1.96 / √(n − |S| − 3)`. Reading `n` and the noise off the
scenario table and taking a representative `|S|` of a few, this comes out to about `0.088` for ER10
(`n = 500`), `0.062` for ER20 (`n = 1000`), `0.044` for SF50 (`n = 2000`), `0.062` for SF50-Hard
(`n = 1000`), and `0.099` for ER20-Noisy (`n = 400`). So on pure sample size the fifty-node SF50 is
the *easiest* per-edge — its two thousand samples let it see partial correlations down to about
`0.044` — while ER20-Noisy is the hardest, unable to resolve anything below about `0.10`. But
detectability per edge is only half the story; the other half is how many real edges actually have
partial correlations below that floor, and that count is driven by density and noise, both of which
push edge partial correlations *down*. Density does it through paths: on a dense graph a direct edge
`X → Y` competes with many indirect `X ⤳ Y` routes, and when I condition on the wrong intermediate
set the near-cancellation of those routes drags the partial correlation of the true edge toward zero.
Noise does it directly: the SEM sets each variable to `∑ w·parents + ε`, and inflating `ε`'s standard
deviation from `1.0` to `2.5` swells the residual variance in the denominator of every partial
correlation, shrinking the signal of each edge. ER20-Noisy suffers on both counts at once — the
highest detection floor *and* the noise pushing every edge toward it, on a denser-than-average ER
graph with the fewest samples — so if any scenario is going to hemorrhage true edges it is that one.
SF50 and SF50-Hard suffer the density side heavily (scale-free hubs create many competing paths)
even though SF50's sample size gives it a low floor. ER10 and ER20 are sparse enough that few edges
sit near the floor at all. My ranked expectation, then, is that the two small sparse graphs are
reachable, the two scale-free graphs degrade through path-driven near-cancellation, and ER20-Noisy is
the worst case; the `SHD` and `adjacency recall` columns will tell me whether that ordering is right.

Everything else about the skeleton phase is the standard PC machinery: I want the order-independent
variant (PC-stable), which freezes each node's adjacency set at the start of each conditioning level
and draws all subsets from the frozen set, so that deleting one edge mid-level does not change which
tests other pairs run. With an oracle this changes nothing; on sample data it makes the output
skeleton independent of the column order, which is the difference between a stable result and one
that wobbles with permutations of the variables. The harness exposes this exactly as
`SkeletonDiscovery.skeleton_discovery(..., stable=True)`, and I keep `α = 0.05` as the conventional
default. I want to be honest that `α` is not functioning as a true significance level: the search
fires thousands of tests, so under any multiple-comparisons accounting the family-wise error is
nowhere near `0.05`, and in fact this compounding is a second density penalty — a true edge on a
dense graph gets tested against many candidate separating subsets, and each one is another
independent chance to spuriously separate it, so the probability that *some* subset wrongly deletes a
faint true edge grows with the number of subsets tried, which grows with degree. Raising `α` would
delete fewer edges (helping the recall I am worried about) but admit more false ones (hurting
precision); lowering it does the reverse. I am not going to tune it against the scenarios — that would
be leaning on dataset-specific constants the task forbids — so I treat `α = 0.05` as a fixed sparsity
knob and let the recall consequences fall where the arithmetic above says they will.

Phase two orients unshielded colliders, and the separating sets recorded during phase one already
tell me which triples are colliders — I proved the arithmetic of it above, so here it is just the
bookkeeping. For an unshielded triple `X — Y — Z`, I deleted `X — Z` because some recorded set
separated them. If `Y` is a non-collider on the length-two path, leaving `Y` out keeps that path
active, so any separator must include `Y`; if `Y` is a collider, the path is already blocked unless I
include `Y`, and including it opens the path with no other interior node to close it, so any valid
separator must leave `Y` out. The rule is therefore: orient `X → Y ← Z` iff `Y` is absent from the
stored separating information for `X, Z` — a lookup, no new tests, exactly the collider-versus-chain
distinction that came out as `Y ∉ {∅}` versus `Y ∈ {Y}` in the two traces. On sample data two
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
lets it correctly orient a whole neighborhood for free, and a wrong collider lets it spread that one
mistake across the neighborhood just as efficiently. So the closure amplifies whatever the first two
phases hand it.

That leaves the question of why PC and not one of its relatives, which I want to settle deliberately
rather than by default. FCI is the constraint method built for latent confounders; it would output a
PAG with circle marks encoding "could be a hidden common cause here." But the generator is explicit
that every variable is observed and the noise is independent additive — there are no latents — so
FCI's extra machinery would buy robustness against a confounding that cannot occur while costing me
orientation certainty (circles where PC commits to arrowheads). On this substrate PC dominates FCI. I
also weigh going straight to a score-based climb like GES, which is legitimate prior art here; but
the constraint-based floor is the right *first* rung for a reason beyond conservatism — it is the
method whose every decision is a single identifiable independence fact, so when it fails, the failure
is legible as a specific broken fact rather than a diffuse optimization artifact, and that legibility
is exactly the diagnostic I want to earn before I spend a rung on anything with more moving parts. So
the step-1 edit is the three phases as three calls into the harness, on the Fisher-z oracle:
`skeleton_discovery(..., stable=True)`, then `uc_sepset(cg_1, 2, ...)`, then `Meek.meek(cg_2, ...)`,
returning `cg.G`.

I am clear-eyed about why this is only the floor, and the reason names what comes next. PC's every
decision is a single thresholded CI test, and I have now put a number on when that decision goes
wrong: whenever a real edge's partial correlation falls below the `1.96/√(n − |S| − 3)` floor —
around `0.10` on ER20-Noisy, and reachable on the scale-free graphs whenever near-cancelling paths
drag a true edge's partial correlation down — a finite-sample Fisher-z test reports "independent" and
I delete a true edge. Worse, an early wrong deletion shrinks a neighborhood and leaves a different
false edge un-testable, so errors ramify, and a single mis-decided collider then spreads through the
Meek closure. I expect this to show up exactly as graph density and noise climb: the small sparse
scenarios should be reachable, but the larger and denser ones — where the count of near-cancellations
grows with the number of paths — should degrade, with adjacency recall falling first (real edges
missed, since precision suffers less: PC rarely *adds* junk, it *drops* faint truth) and arrow recall
right behind it (colliders mis-decided on a wrong skeleton, then amplified by Meek). ER20-Noisy
should be the floor of the floor. That predicted failure mode — brittle single-edge decisions that
delete real-but-faint dependences on dense, high-noise graphs — is precisely what motivates leaving
the constraint-based view at step 2 for a search whose decisions rest on comparing whole-model fits
rather than thresholding one statistic, and are therefore robust to exactly the near-cancellations
that push a single partial correlation under the detection floor. The full scaffold module — the
three-call PC pipeline on `fisherz` — is in the answer.
