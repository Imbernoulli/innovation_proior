GRaSP did exactly what I bet it would, and it failed exactly where I flagged. The two small sparse
graphs snapped to zero — ER10 and ER20 both `SHD` 0, every metric 1.0 — confirming that moving from
PC's thresholded CI verdicts to a score-based ordering search erases the recall collapse where there
is no near-cancellation to fight. On the scale-free graphs the adjacency is now essentially right
(SF50 adjacency precision/recall 0.979/0.979, SF50-Hard 0.951/0.965) where PC was at 0.51 and 0.30
recall — the robust score recovered the real edges PC's tests deleted. But look at the two places it
is still soft, because they name the next move, and I want to read them as edge counts rather than
leave them as adjectives.

First, SF50. Adjacency is near-perfect but the arrows are not: arrow precision 0.75 with `SHD` 26.
With about 96 true edges and adjacency precision and recall both 0.979, I have roughly 94 of the 96
edges placed and only about 2 false and 2 missed — the skeleton is off by maybe four edges total. Yet
`SHD` is 26. So on the order of twenty of those twenty-six errors are *orientation* errors, not
adjacency errors: the tuck-DFS found the right skeleton and then landed on a DAG that is in the right
equivalence-class neighborhood but is not the highest-scoring member of it, so a quarter of the
arrowheads it committed point the wrong way. Second, and this is the row I explicitly said to watch,
ER20-Noisy *collapsed differently from PC*: `SHD` 58, adjacency precision 0.674, adjacency recall
0.841, arrow precision 0.459, arrow recall 0.574. Back the counts out: about 66 true edges, recall
0.841 gives roughly 56 recovered, so GRaSP found the real edges PC missed (PC had recovered about
16). But precision 0.674 means the estimate carries about `56/0.674 ≈ 83` edges, so roughly *27 false
positives*. GRaSP's adjacency precision fell *below* PC's 0.85 — meaning where PC deleted 50 true
edges and added 3 junk ones, GRaSP recovered the truth and added 27 junk ones. That is the failure
mode I predicted in the score's own terms: on the densest, lowest-sample row, where the BIC penalty
floor is highest and the sparse-Markov razor is near its limit, the plateau-crossing DFS's within-MEC
wandering has no clearly sparser truth to cross toward, so its neutral excursions drift it into a
denser-than-truth optimum and the depth-3 covered-tuck excursions cannot pull it back out.

SF50-Hard is worth a second look because at first glance it complicates the story: it is the *denser*
scale-free graph, yet its arrows came out cleaner than SF50's — arrow precision 0.897 against SF50's
0.75, with `SHD` 25 against 26. Backing out the counts, about 141 true edges, recall 0.965 gives
roughly 136 recovered against about 5 missed, precision 0.951 gives about 7 false, so the skeleton is
off by a dozen and the remaining `SHD` is again mostly orientation — but only about thirteen orientation
errors, against SF50's twenty. That the denser graph orients *better* is not a paradox once I think
about where orientation certainty comes from: arrowheads in a CPDAG are pinned by unshielded colliders
and their Meek propagation, and a denser scale-free graph with `m = 3` has more colliders per node than
the `m = 2` graph, so more of its edges are forced directed by v-structures and fewer are left as
coin-flips inside a large undirected component. So SF50's *lower* density is actually why its arrows are
softer — it has more genuinely-undirected edges whose orientation the tuck-DFS had to guess at, and it
guessed wrong on a quarter of them. This sharpens what "wrong member of the right class" means and what
the fix must do: the skeleton being right does not pin the arrows; the arrows are pinned by which
v-structures the projected DAG actually contains, and those depend on the order, because the order
decides each variable's parent set and hence which triples come out as collisions. A higher-scoring
order is one whose induced v-structures match the truth's more often, and `dag2cpdag` then recovers
more of the true arrowheads. So the arrow-precision problem is, precisely, an order-quality problem, and
a move that optimizes each variable's global position against the full score is aimed straight at it —
which is another reason to expect the move, not the score, to be the lever.

So the diagnosis is sharp, and it splits cleanly into what to keep and what to replace. GRaSP's *view*
is right — search orderings, score with local consistency, robust to almost-unfaithfulness — and the
0/0 on the sparse graphs plus the recall recovery on the scale-free ones is the evidence. What is
wrong is the *move*. The tuck is powerful but it is a DFS over sequences of covered/general tucks with
interacting depth knobs, and that machinery is exactly what produces both soft spots: the depth-3
bound means the search stops short of the best-oriented member on SF50 (the twenty stray arrowheads),
and the neutral within-MEC excursions are what let it wander dense on ER20-Noisy (the twenty-seven
spurious edges). Both symptoms trace to the same source — a relaxed, plateau-crossing search with a
depth parameter and a neutral band — so the fix should be a single change to the move, not a patch to
each symptom.

Before I design that move I should rule out the cheaper things, because if one of them worked I would
rather do it. The most tempting is to attack the ER20-Noisy over-adding directly by raising the
complexity penalty `lambda_value` above 2 — a heavier penalty charges more per parent and would
suppress the 27 spurious edges. But `lambda_value` is a single constant shared across all five
scenarios, and the task forbids leaning on dataset-specific constants, so I cannot raise it on
ER20-Noisy alone. Raising it *globally* is worse than useless: SF50 reached adjacency recall 0.979 at
`λ = 2`, and that recall lives right at the penalty floor I computed for it (around `0.087`); a
heavier penalty would push that floor up and start deleting the faint scale-free edges GRaSP just
recovered, trading the ER20-Noisy precision problem for a reopened SF50 recall problem. So the
over-adding is not fixable through the score — the score is already at the only setting that serves
all five rows — which means it has to be fixed through the search. That is a real constraint, not a
preference, and it points squarely at the move. A second option is to keep the tuck but deepen the
DFS past 3 to reach the better-oriented SF50 member; but a deeper DFS makes the *wandering* worse, not
better — more depth is more neutral excursion, which is exactly what over-added on ER20-Noisy — and it
costs more branching for the privilege. A third is to bolt the formal Backward-Equivalence phase onto
GRaSP; but that adds machinery to clean up after the wandering rather than removing the wandering, and
the problem I have is too much motion of the wrong kind, not too little cleanup. Every cheap patch
either violates the shared-constant rule or amplifies the very excursion that caused the damage. So
the move itself has to change.

Let me think about what a good move should accomplish, because the answer is hiding in how the score
depends on the order. The score depends on `π` only through, for each variable, which other variables
sit *ahead* of it — that determines its candidate parents and hence its Markov boundary. So the thing
that most changes a variable's parent set is *where that variable sits* relative to everyone else. The
tuck reorders a stretch of `π` to flip one particular edge; the adjacent transposition swaps two
neighbors and barely moves anything. Both are anchored to a local feature of the current graph, and
both need to be chained (via depth, via neutral excursions) to travel any distance. What if I instead
take one variable `v`, pull it out of the order entirely, and drop it back into the *single best
slot* — the position among all `p` possible positions that maximizes the total score? That is not a
local swap and it is not edge-anchored: sliding `v` from the end of the order to the front is, in
effect, a whole run of adjacent transpositions collapsed into one decision, so it can leap across the
shallow optima that trap single swaps in one atomic step, no depth recursion required. And it has
*zero* depth parameters — there is nothing to tune, I just evaluate every insertion point for `v` and
take the argmax. Sweep that best-position move over every variable, one at a time in shuffled order,
and repeat the whole sweep until a full pass produces no move. That is the Best Order Score Search:
best-position-per-variable, sweep to convergence.

The reason this is the right answer to GRaSP's two specific failures is worth pinning down against the
counts I just backed out. On SF50, GRaSP's arrows were soft because the tuck-DFS settled on a
not-quite-optimal member of the right class — the twenty stray arrowheads. The best-position move, by
re-optimizing each variable's *global* placement against the full score, lands on a higher-scoring
order, and a higher-scoring order projects to a better-oriented DAG, so I expect the arrow errors to
shrink. On ER20-Noisy, GRaSP over-added the twenty-seven spurious edges because its plateau-crossing
wandered denser; the best-position sweep has *no* neutral within-MEC excursion to wander on — I accept
a move only when it strictly improves the total score by more than a tolerance, so the search cannot
drift sideways into a denser optimum the way a covered-tuck DFS can. It can only ratchet toward higher
score, and a denser-than-truth graph scores worse under the BIC penalty, so the greedy monotone climb
is repelled from exactly the region GRaSP got stuck in. The move is *greedier and simpler*, and on
this task that is a feature: GRaSP's relaxation bought robustness on the moderate graphs but became a
liability on the hardest row, and BOSS trades the relaxation away for a clean, monotone, parameter-light
climb.

I want to check that "monotone climb terminates" is a real guarantee and not a hope, because a search
that could cycle would be worse than GRaSP, not better. Every accepted best-position move strictly
increases the total score by more than the tolerance; the total score is bounded above by the score of
the globally optimal order (a finite number); a strictly increasing sequence bounded above and moving
by at least a fixed positive step can only take finitely many steps. So the sweep must halt, with no
bookkeeping. Contrast GRaSP, which *needed* a flip-set `history` precisely because its neutral
excursions did not change the score and so could revisit orders forever unless it explicitly forbade
loops; BOSS needs no history because strict monotonicity forbids loops for free. That is not a minor
convenience — it is the same property, seen from the termination side, that keeps the search out of the
dense optimum: the thing that makes BOSS unable to cycle is the thing that makes it unable to wander
denser. One property, both benefits.

There is a small but real choice about where to start the climb, and it interacts with the
termination argument I just made. GRaSP shuffled its *initial* permutation, because a plateau-crossing
DFS that can wander is sensitive to where it begins — a bad start can stick it in a different basin. A
strictly monotone best-position sweep is far less sensitive: wherever it starts, it ratchets uphill to
a local optimum, and the per-sweep reshuffling of *which variable I move next* already injects the
stochasticity that lets the climb escape the artifacts of any one visiting order. So I start from the
identity order `[0, 1, …, p−1]` and let the swept moves do the work, rather than randomizing the start.
That is not a claim that the identity start is globally optimal — a monotone climb only promises a local
optimum — but on these graphs the basin structure is benign enough (the sparse rows have a unique
sparsest order, and the scale-free rows have a strong global signal) that the starting order is not
what stands between the sweep and a good result; the *move* is. Keeping the start fixed also makes the
run reproducible up to the per-sweep shuffle, which is one fewer source of variance when I read the
metrics.

Now the pieces I keep wholesale from GRaSP, because the *view* is unchanged and I want the comparison
clean. The projection is the same: given the order, each variable's parents are its score-selected
Markov boundary among its predecessors, via grow-shrink, with the same decomposable local score. For
this linear Gaussian task that score is again `local_score_BIC_from_cov` with `lambda_value = 2` —
identical to GRaSP's score, so any improvement is attributable to the *search move*, not the criterion,
which is exactly the controlled comparison I want and the reason I did not touch `λ` even though the
ER20-Noisy over-adding tempted me to. And the per-variable Grow-Shrink Tree (`GST`) is again what makes
the move affordable: when I evaluate inserting `v` at every slot, I am repeatedly asking "what is `v`'s
best parent set among a given prefix, and what does that prefix do to the variables after `v`," and the
`GST` caches those grow/shrink traces keyed by the available predecessors so the sweep does not rerun
grow-shrink from scratch at every slot.

The mechanics of the best-position move deserve care, because the naive version does not scale to fifty
variables and the efficient version is what makes the sweep feasible. The naive version reinserts `v`
at each of the `p` slots, reprojects the whole order, and rescores — that is `O(p)` full re-evaluations
per variable, `O(p²)` per variable if each re-evaluation itself touches `O(p)` families, and `O(p³)`
per full sweep. At `p = 50` that is on the order of `125,000` family scorings per sweep, and there are
many sweeps. The trick the implementation uses collapses this: compute all `p + 1` insertion scores in
two linear passes over the order. Pull `v` out. Sweep the prefix left-to-right, and at each candidate
position accumulate two things — the score `v` itself would get with the current prefix as its
available parents, and the running score of the other variables ahead of it. Then sweep right-to-left
accumulating the contribution of the variables that would fall *after* `v` at each position, since
inserting `v` earlier changes the predecessor sets of everyone behind it. Summing the forward and
backward accumulations gives the total score for every insertion slot at once, in `O(p)` `GST` traces
rather than `O(p²)`, so the whole sweep is `O(p²)` — a factor-of-`p` saving that turns the 50-node case
from `125,000` scorings into about `2,500`. I can sanity-check the two-pass bookkeeping on a
three-variable order `⟨a, b, c⟩` inserting `v`: the forward pass records, for slot 0, `v`'s score with
empty prefix; for slot 1, `v`'s score with prefix `⟨a⟩` plus `a`'s own score; for slot 2, with prefix
`⟨a, b⟩` plus `a`'s and `b`'s scores; and slot 3 with the full prefix. The backward pass then adds, to
each slot, the re-scored contribution of whichever of `a, b, c` end up *after* `v` at that slot. At
slot 0 that is all three re-scored with `v` ahead of them; at slot 3, none. Summed, each slot holds the
total score of the order with `v` at that position — which is exactly what I want to argmax, and I take
the best only if it beats `v`'s current slot by more than `1e-6`. The one place to be careful is the
index arithmetic when I actually move `v`: I remove `v` first and then insert at `best`, but removing
`v` shifts every later index down by one, so if the target slot was after `v`'s old position I insert
at `best − 1` — the `best - int(best > i)` in the fill — otherwise I would land one position too far
right. It is a one-line correction but getting it wrong would silently corrupt every move, so it earns
the check.

When the sweep converges I read the DAG off the final order — for each variable, run grow-shrink once
more against its settled prefix to fix its parents — and convert to a CPDAG with `dag2cpdag`, since the
data identify only the MEC. One design choice I am explicit about, and it is visible in the literal
fill: there is *no* Backward-Equivalence-Search cleanup phase here. The full two-phase BOSS design can
run BES after the ordering phase to nail the formal large-sample guarantee, because the projection
always returns a subgraph-minimal DAG containing `P` and BES supplies the final correctness step from
any initial order. The harness's fill is the *lean* form: the best-position sweep over the BIC-from-cov
score, then read off and convert, with no BES. I accept that I am shipping the parameter-light ordering
search without the optional backward phase — and the bet is that on these graphs the best-position climb
alone reaches a high enough order that the extra phase is not what is between me and the truth, given
that GRaSP already reached near-perfect adjacency on four of five rows without any backward phase and
the residual errors were about the *move*, not a missing cleanup.

So the delta from GRaSP is a single surgical swap: same permutation view, same BIC-from-cov score with
`lambda_value = 2`, same GST caching, same `dag2cpdag` output — but the tuck-DFS with its depth knobs
and within-MEC excursions is replaced by the best-position move swept to convergence, a greedy,
monotone, parameter-free climb. Reading GRaSP's exact numbers, the falsifiable claims are precise. The
two small sparse graphs should *stay* perfect (`SHD` 0, all 1.0) — there is no harder optimum there for
either method, so the simpler move loses nothing, and the monotone climb reaches the unique sparsest
order. On SF50 I expect the soft arrow precision (0.75 under GRaSP, the twenty stray arrowheads) to
climb substantially, with `SHD` falling from 26, because the global best-position move finds a
higher-scoring, better-oriented order; SF50-Hard should improve from `SHD` 25 for the same reason. The
decisive row is ER20-Noisy: GRaSP's collapse there (`SHD` 58, adjacency precision 0.674, the
twenty-seven spurious edges) was driven by over-adding through plateau-wandering, and a
strictly-improving best-position sweep with no neutral excursion should *not* over-add the same way — so
if BOSS is the right move, ER20-Noisy's `SHD` should drop by a large factor and its adjacency precision
should recover from 0.674 back toward the high-0.9s. If instead BOSS *also* over-adds on ER20-Noisy,
then the problem is the score or the sample size, not the move — the monotone climb would be terminating
at a genuinely denser optimum that the BIC criterion itself prefers at 400 samples — and the next rung
would have to attack the criterion rather than the search. That is the clean fork the decisive row will
resolve. The full scaffold module — the `_boss_better_mutation` two-pass best-position move, the
shuffle-and-sweep loop, and the BIC-from-cov `run_causal_discovery` with `dag2cpdag` output — is in the
answer.
