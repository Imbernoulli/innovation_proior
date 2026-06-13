GRaSP did exactly what I bet it would, and it failed exactly where I flagged. The two small sparse
graphs snapped to zero — ER10 and ER20 both SHD 0, every metric 1.0 — confirming that moving from
PC's thresholded CI verdicts to a score-based ordering search erases the recall collapse where there
is no near-cancellation to fight. On the scale-free graphs the adjacency is now essentially right
(SF50 adjacency precision/recall 0.979/0.979, SF50-Hard 0.951/0.965) where PC was at 0.51 and 0.30
recall — the robust score recovered the real edges PC's tests deleted. But look at the two places it
is still soft, because they name the next move. First, SF50 *arrow* precision is only 0.75 with SHD
26 — the skeleton is nearly perfect but the orientations are not, so the tuck-DFS is landing on a DAG
in roughly the right equivalence-class neighborhood but not consistently picking the
highest-scoring member. Second, and this is the row I explicitly said to watch, ER20-Noisy
*collapsed*: SHD 58, adjacency precision 0.674, arrow precision 0.459, arrow recall 0.574. At 400
samples and noise 2.5 on a denser ER graph, GRaSP's adjacency precision fell *below* PC's (0.85),
meaning the score-based search is now over-adding edges — the opposite of PC's problem. So the
plateau-crossing DFS, on the hardest, lowest-sample row, wanders into a denser-than-truth optimum and
the depth-3 covered-tuck excursions cannot pull it back out. That is the failure mode I predicted:
where the sparse-Markov razor is near its own limit, the tuck-DFS's within-MEC wandering is a
liability, not an asset.

So the diagnosis is sharp. GRaSP's *view* is right — search orderings, score with local consistency,
robust to almost-unfaithfulness — and I am keeping all of it. What is wrong is the *move*. The tuck
is powerful but it is a DFS over sequences of covered/general tucks with interacting depth knobs, and
that machinery is exactly what makes it both fiddly and, on the noisy row, prone to wandering into a
dense optimum it cannot escape at depth 3. I want GRaSP's accuracy target with its machinery gone:
a move that still leaps far enough in one step to escape shallow optima, but that has no depth
parameters to tune and no within-MEC excursion to get lost in.

Let me think about what a good move should accomplish, because the answer is hiding in how the score
depends on the order. The score depends on `π` only through, for each variable, which other
variables sit *ahead* of it — that determines its candidate parents and hence its Markov boundary.
So the thing that most changes a variable's parent set is *where that variable sits* relative to
everyone else. The tuck reorders a stretch of `π` to flip one particular edge; the adjacent
transposition swaps two neighbors and barely moves anything. Both are anchored to a local feature of
the current graph. What if I instead take one variable `v`, pull it out of the order entirely, and
drop it back into the *single best slot* — the position among all `p` possible positions that
maximizes the total score? That is not a local swap and it is not edge-anchored: sliding `v` from the
end of the order to the front is, in effect, a whole run of adjacent transpositions collapsed into
one decision, so it can leap across the shallow optima that trap single swaps. And it has *zero*
depth parameters — there is nothing to tune, I just evaluate every insertion point for `v` and take
the argmax. Sweep that best-position move over every variable, one at a time in shuffled order, and
repeat the whole sweep until a full pass produces no move. That is the Best Order Score Search:
best-position-per-variable, sweep to convergence.

The reason this is the right answer to GRaSP's specific failures is worth pinning down. On SF50,
GRaSP's arrows were soft because the tuck-DFS settled on a not-quite-optimal member of the right
class; the best-position move, by re-optimizing each variable's *global* placement against the full
score, lands on a higher-scoring order, and a higher-scoring order projects to a better-oriented DAG
— so I expect arrow precision to lift. On ER20-Noisy, GRaSP over-added because its plateau-crossing
wandered denser; the best-position sweep has no neutral within-MEC excursion to wander on — every
accepted move strictly improves the score by more than a tolerance, so it cannot drift into a denser
optimum the way a covered-tuck DFS can. The move is *greedier and simpler*, and on this task that is
a feature: GRaSP's relaxation bought robustness on the moderate graphs but became a liability on the
hardest row, and BOSS trades the relaxation away for a clean, monotone, parameter-light climb.

Now the pieces I keep wholesale from GRaSP, because the *view* is unchanged. The projection is the
same: given the order, each variable's parents are its score-selected Markov boundary among its
predecessors, via grow-shrink, with the same decomposable local score. For this linear Gaussian task
that score is again `local_score_BIC_from_cov` with `lambda_value = 2` — identical to GRaSP's score,
so any improvement is attributable to the *search move*, not the criterion, which is exactly the
controlled comparison I want. And the per-variable Grow-Shrink Tree (`GST`) is again what makes the
move affordable: when I evaluate inserting `v` at every slot, I am repeatedly asking "what is `v`'s
best parent set among a given prefix, and what does that prefix do to the variables after `v`," and
the `GST` caches those grow/shrink traces keyed by the available predecessors so the sweep does not
rerun grow-shrink from scratch at every slot.

The mechanics of the best-position move deserve care, because the efficient version is what makes a
50-node sweep feasible. The naive version reinserts `v` at each of `p` slots, reprojects, and
rescores — `O(p)` full re-evaluations per variable. The trick the implementation uses is to compute
all `p + 1` insertion scores in two linear passes over the order. Pull `v` out. Sweep the prefix
left-to-right accumulating, for each candidate position, the score `v` would get with that prefix as
its available parents, plus the running score of the other variables ahead of it; then sweep
right-to-left accumulating the contribution of the variables that would fall *after* `v` at each
position. Summing the two passes gives the total score for every insertion slot at once, and I take
the best. If the best slot beats `v`'s current position by more than a small tolerance (`1e-6`), I
remove `v` and reinsert it there (adjusting the index for the removal); otherwise `v` stays. That is
the single-variable mutation; the outer loop shuffles the variable list and sweeps until a full pass
moves nothing.

When the sweep converges I read the DAG off the final order — for each variable, run grow-shrink once
more against its settled prefix to fix its parents — and convert to a CPDAG with `dag2cpdag`, since
the data identify only the MEC. One design choice I am explicit about, and it is visible in the
literal fill: there is *no* Backward-Equivalence-Search cleanup phase here. The full two-phase BOSS
design can run BES after the ordering phase to nail the formal large-sample guarantee (because the
projection always returns a subgraph-minimal DAG containing `P`, and BES supplies the final
correctness step from any initial order). The harness's fill is the *lean* form: the best-position
sweep over the BIC-from-cov score, then read off and convert, with no BES. I accept that I am
shipping the parameter-light ordering search without the optional backward phase — and the bet is
that on these graphs the best-position climb alone reaches a high enough order that the extra phase is
not what is between me and the truth.

So the delta from GRaSP is a single surgical swap: same permutation view, same BIC-from-cov score
with `lambda_value = 2`, same GST caching, same `dag2cpdag` output — but the tuck-DFS with its depth
knobs and within-MEC excursions is replaced by the best-position move swept to convergence, a greedy,
monotone, parameter-free climb. Reading GRaSP's exact numbers, the falsifiable claims are precise.
The two small sparse graphs should *stay* perfect (SHD 0, all 1.0) — there is no harder optimum there
for either method, so the simpler move loses nothing. On SF50 I expect the soft arrow precision (0.75
under GRaSP) to climb substantially, with SHD falling from 26, because the global best-position move
finds a higher-scoring, better-oriented order; SF50-Hard should improve from SHD 25 for the same
reason. The decisive row is ER20-Noisy: GRaSP's collapse there (SHD 58, adjacency precision 0.674)
was driven by over-adding through plateau-wandering, and a strictly-improving best-position sweep with
no neutral excursion should *not* over-add the same way — so if BOSS is the right move, ER20-Noisy's
SHD should drop by a large factor and its adjacency precision should recover from 0.674 back toward
the high-0.9s. If instead BOSS also over-adds on ER20-Noisy, then the problem is the score or the
sample size, not the move, and the next rung would have to attack the criterion rather than the
search. The full scaffold module — the `_boss_better_mutation` two-pass best-position move, the
shuffle-and-sweep loop, and the BIC-from-cov `run_causal_discovery` with `dag2cpdag` output — is in
the answer.
