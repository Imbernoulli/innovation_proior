GRaSP did exactly what I bet it would, and it failed exactly where I flagged. The two small sparse
graphs snapped to zero — ER10 and ER20 both `SHD` 0, every metric 1.0 — confirming that moving from
PC's thresholded CI verdicts to a score-based ordering search erases the recall collapse where there
is no near-cancellation to fight. On the scale-free graphs adjacency is now essentially right (SF50
precision/recall 0.979/0.979, SF50-Hard 0.951/0.965) where PC was at 0.51 and 0.30 recall — the robust
score recovered the real edges PC's tests deleted. But two places are still soft, and they name the
next move once I read them as edge counts rather than adjectives.

First, SF50: adjacency near-perfect but arrows are not — arrow precision 0.75 with `SHD` 26. With
about 96 true edges and adjacency precision and recall both 0.979, roughly 94 of the 96 edges are
placed, only about 2 false and 2 missed — the skeleton is off by maybe four edges. Yet `SHD` is 26. So
on the order of twenty of those errors are *orientation* errors: the tuck-DFS found the right skeleton
and then landed on a DAG in the right equivalence-class neighborhood but not the highest-scoring member
of it, so a quarter of the arrowheads it committed point the wrong way. Second, the row I said to
watch, ER20-Noisy, collapsed *differently* from PC: `SHD` 58, adjacency precision 0.674, adjacency
recall 0.841, arrow precision 0.459, arrow recall 0.574. Back the counts out: about 66 true edges,
recall 0.841 gives roughly 56 recovered, so GRaSP found the real edges PC missed (PC had recovered
about 16). But precision 0.674 means the estimate carries about `56/0.674 ≈ 83` edges, so roughly *27
false positives*. GRaSP's adjacency precision fell *below* PC's 0.85: where PC deleted 50 true edges
and added 3 junk ones, GRaSP recovered the truth and added 27 junk ones. That is the residual failure
I flagged — on the densest, lowest-sample row, where the BIC penalty floor is highest and the
sparse-Markov razor is near its limit, the plateau-crossing DFS's within-MEC wandering has no clearly
sparser truth to cross toward, so its neutral excursions drift it into a denser-than-truth optimum the
depth-3 covered tucks cannot pull it back out of.

SF50-Hard is worth a second look because at first glance it complicates the story: it is the *denser*
scale-free graph, yet its arrows came out cleaner than SF50's — arrow precision 0.897 against 0.75,
`SHD` 25 against 26. Backing out: about 141 true edges, recall 0.965 gives roughly 136 recovered
against 5 missed, precision 0.951 about 7 false, so the skeleton is off by a dozen and the rest of
`SHD` is again orientation — but only about thirteen orientation errors, against SF50's twenty. That
the denser graph orients *better* is not a paradox once I ask where orientation certainty comes from:
arrowheads in a CPDAG are pinned by unshielded colliders and their Meek propagation, and a denser
scale-free graph with `m = 3` has more colliders per node than the `m = 2` graph, so more edges are
forced directed by v-structures and fewer are left as coin-flips inside a large undirected component.
So SF50's *lower* density is why its arrows are softer — it has more genuinely-undirected edges whose
orientation the tuck-DFS had to guess, and it guessed wrong on a quarter of them. This sharpens what
"wrong member of the right class" means: the skeleton being right does not pin the arrows; the arrows
are pinned by which v-structures the projected DAG contains, and those depend on the order, because the
order decides each variable's parent set and hence which triples come out as collisions. A
higher-scoring order induces v-structures matching the truth's more often, and `dag2cpdag` recovers
more true arrowheads. So the arrow-precision problem is an order-quality problem, and a move that
optimizes each variable's global position against the full score is aimed straight at it.

So the diagnosis splits cleanly into what to keep and what to replace. GRaSP's *view* is right — search
orderings, score with local consistency, robust to almost-unfaithfulness — and the 0/0 on the sparse
graphs plus the recall recovery on the scale-free ones is the evidence. What is wrong is the *move*.
The tuck is a DFS over sequences of covered/general tucks with interacting depth knobs, and that
machinery produces both soft spots: the depth-3 bound stops short of the best-oriented member on SF50
(the twenty stray arrowheads), and the neutral within-MEC excursions let it wander dense on ER20-Noisy
(the twenty-seven spurious edges). Both trace to the same source — a relaxed, plateau-crossing search
with a depth parameter and a neutral band — so the fix should be a single change to the move.

Before designing it I should rule out the cheaper things. The most tempting is to attack the
ER20-Noisy over-adding directly by raising `lambda_value` above 2 — a heavier penalty would suppress
the 27 spurious edges. But `lambda_value` is a single constant shared across all five scenarios, and
the task forbids leaning on dataset-specific constants, so I cannot raise it on ER20-Noisy alone.
Raising it *globally* is worse than useless: SF50 reached adjacency recall 0.979 at `λ = 2`, right at
the penalty floor I computed for it (around `0.087`); a heavier penalty would push that floor up and
start deleting the faint scale-free edges GRaSP just recovered, trading the ER20-Noisy precision problem
for a reopened SF50 recall problem. So the over-adding is not fixable through the score — the score is
already at the only setting that serves all five rows — which points squarely at the search. A second
option, keeping the tuck but deepening the DFS past 3 to reach SF50's better-oriented member, makes the
*wandering* worse: more depth is more neutral excursion, exactly what over-added on ER20-Noisy, at more
branching cost. A third, bolting a formal Backward-Equivalence phase onto GRaSP, adds machinery to
clean up after the wandering rather than removing it, and my problem is too much motion of the wrong
kind, not too little cleanup. Every cheap patch either violates the shared-constant rule or amplifies
the very excursion that caused the damage. So the move itself has to change.

What should a good move accomplish? The answer is in how the score depends on the order: it depends on
`π` only through, for each variable, which other variables sit *ahead* of it — that fixes its candidate
parents and hence its Markov boundary. So the thing that most changes a variable's parent set is
*where that variable sits* relative to everyone else. The tuck reorders a stretch of `π` to flip one
edge; the adjacent transposition swaps two neighbors and barely moves anything; both are edge-anchored
and need chaining to travel any distance. What if I instead take one variable `v`, pull it out of the
order entirely, and drop it back into the *single best slot* — the position among all `p` that
maximizes the total score? That is not a local swap and not edge-anchored: sliding `v` from the end to
the front is a whole run of adjacent transpositions collapsed into one decision, so it leaps across the
shallow optima that trap single swaps in one atomic step, no depth recursion. And it has *zero* depth
parameters — I just evaluate every insertion point for `v` and take the argmax. Sweep that best-position
move over every variable, one at a time in shuffled order, and repeat until a full pass produces no
move. That is the Best Order Score Search: best-position-per-variable, sweep to convergence.

Why this is the right answer to GRaSP's two specific failures is worth pinning against the counts.
On SF50, the arrows were soft because the tuck-DFS settled on a not-quite-optimal member of the right
class; re-optimizing each variable's *global* placement against the full score lands on a
higher-scoring order, which projects to a better-oriented DAG, so the arrow errors should shrink. On
ER20-Noisy, GRaSP over-added because its plateau-crossing wandered denser; the best-position sweep has
*no* neutral within-MEC excursion to wander on — I accept a move only when it strictly improves the
total score beyond a tolerance, so the search cannot drift sideways into a denser optimum. It can only
ratchet toward higher score, and a denser-than-truth graph scores worse under the BIC penalty, so the
greedy monotone climb is repelled from exactly the region GRaSP got stuck in. The move is greedier and
simpler, and here that is a feature: GRaSP's relaxation bought robustness on the moderate graphs but
became a liability on the hardest row, and BOSS trades it for a clean, monotone, parameter-light climb.
That strict monotonicity also gives termination for free — every accepted move raises the total score
by more than the tolerance, the score is bounded above by the globally optimal order, so the sweep
halts with no bookkeeping. GRaSP *needed* its flip-set `history` precisely because its neutral
excursions could revisit orders forever; BOSS needs none, and the same monotonicity that gives
termination is what keeps it out of the dense optimum.

There is a small choice about where to start the climb. GRaSP shuffled its initial permutation because
a plateau-crossing DFS is sensitive to where it begins. A strictly monotone best-position sweep is far
less so: wherever it starts it ratchets uphill, and the per-sweep reshuffling of *which variable I move
next* already injects the stochasticity that escapes any one visiting order's artifacts. So I start
from the identity order `[0, 1, …, p−1]` and let the swept moves do the work, which also makes the run
reproducible up to the per-sweep shuffle — one fewer source of variance. It is not a claim the identity
start is globally optimal (a monotone climb only promises a local optimum), but on these graphs the
basin structure is benign enough that the start is not what stands between the sweep and a good result;
the *move* is.

The pieces I keep wholesale from GRaSP make the comparison clean. The projection is the same: given the
order, each variable's parents are its score-selected Markov boundary among its predecessors, via
grow-shrink, with the same decomposable score — again `local_score_BIC_from_cov` with
`lambda_value = 2`, *identical* to GRaSP's, so any improvement is attributable to the search move, not
the criterion, and the reason I did not touch `λ` even though the ER20-Noisy over-adding tempted me.
And the per-variable Grow-Shrink Tree (`GST`) again makes the move affordable: evaluating inserting `v`
at every slot repeatedly asks "what is `v`'s best parent set among a given prefix, and what does that
prefix do to the variables after `v`," and the `GST` caches those traces keyed by available
predecessors so the sweep does not rerun grow-shrink from scratch at every slot.

The mechanics of the best-position move deserve care, because the naive version does not scale to fifty
variables and the efficient version is what makes the sweep feasible. Naively, reinserting `v` at each
of `p` slots and reprojecting is `O(p)` full re-evaluations per variable, `O(p³)` per sweep — about
`125,000` family scorings at `p = 50`, over many sweeps. The trick collapses this to two linear passes.
Pull `v` out. Sweep the prefix left-to-right, and at each candidate position accumulate two things: the
score `v` itself would get with the current prefix as its available parents, and the running score of
the other variables ahead of it. Then sweep right-to-left accumulating the contribution of the
variables that would fall *after* `v` at each position, since inserting `v` earlier changes the
predecessor sets of everyone behind it. Summing forward and backward gives the total score for every
insertion slot at once, in `O(p)` `GST` traces rather than `O(p²)`, so the whole sweep is `O(p²)` — the
50-node case drops from `125,000` scorings to about `2,500`. The one place to be careful is the index
arithmetic when I actually move `v`: I remove `v` first and then insert at `best`, but removing `v`
shifts every later index down by one, so if the target slot was after `v`'s old position I insert at
`best − 1` — the `best - int(best > i)` in the fill. It is a one-line correction, but getting it wrong
would silently corrupt every move, so it earns the care.

When the sweep converges I read the DAG off the final order — for each variable, run grow-shrink once
more against its settled prefix to fix its parents — and convert to a CPDAG with `dag2cpdag`, since the
data identify only the MEC. One choice I am explicit about: there is *no* Backward-Equivalence-Search
cleanup phase here. The full two-phase BOSS design can run BES after the ordering phase to nail the
formal large-sample guarantee, because the projection always returns a subgraph-minimal DAG containing
`P` and BES supplies the final correctness step from any initial order. I ship the *lean* form — the
best-position sweep over the BIC-from-cov score, then read off and convert — on the bet that on these
graphs the climb alone reaches a high enough order that the extra phase is not what stands between me
and the truth, given that GRaSP already reached near-perfect adjacency on four of five rows without any
backward phase and the residual errors were about the *move*, not a missing cleanup.

So the delta from GRaSP is one surgical swap: the tuck-DFS with its depth knobs and within-MEC
excursions, replaced by the best-position move swept to convergence — a greedy, monotone,
parameter-free climb over the same score and the same GST caching. Against GRaSP's numbers, the sparse
graphs should stay perfect and the scale-free rows' soft arrows should firm up as the global move
lands a higher-scoring, better-oriented order. The decisive row is ER20-Noisy: its collapse there was
over-adding driven by plateau-wandering, and a strictly-improving sweep with no neutral excursion
should not over-add the same way, so its `SHD` should drop and adjacency precision recover well out of
the 0.6s. If instead BOSS *also* over-adds there, the fault is the score or the sample size rather than
the move — the monotone climb would be halting at a genuinely denser optimum the BIC criterion prefers
at 400 samples — and the next move would have to attack the criterion, not the search. The full BOSS
module is in the answer.
