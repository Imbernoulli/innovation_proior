PC's numbers tell me exactly what I feared, and the pattern is the lever. On the two small sparse
graphs PC is fine on adjacency precision — 1.0 on ER10, 0.857 on ER20 — but its *recall* is already
sagging (0.857, 0.585), and on the larger and denser scenarios it collapses: SF50 recall 0.51,
SF50-Hard recall 0.30, ER20-Noisy recall a catastrophic 0.246 with adjacency precision still 0.85.
Read that precision/recall split: PC is not adding junk edges, it is *missing real ones*, and it
misses more as density and noise climb. That is the exact signature I predicted — a single
thresholded Fisher-z test deletes a true edge whenever the partial correlation of a real-but-faint
dependence sits near zero, and on dense graphs (more paths, more near-cancellations) and high noise
(ER20-Noisy at noise 2.5, only 400 samples) that happens all over the graph at once. The arrow
numbers are even worse (SF50 arrow recall 0.19, ER20-Noisy 0.044) because they sit downstream of the
broken skeleton: a missed edge or a wrong deletion ramifies through the collider phase and the Meek
closure, so an early skeleton error multiplies into the orientation. The SHD line says the same in
one number — 8, 31, 93, 143, 69 — climbing with density. PC's correctness theorem is true; the
finite-sample assumption it rests on, faithfulness, is what breaks, and it breaks exactly where edges
crowd together. So I do not want a faster faithful algorithm. I want decisions that are robust to
faint-but-real dependences, and I want them to scale to the fifty-node graphs where PC fell apart.

What tolerates almost-unfaithfulness? The line of work that searches *orderings* instead of edges.
Take a permutation `π` of the variables, treat it as an acyclic order, and build a DAG `G_π` by
giving each variable parents drawn only from the variables ahead of it in `π`. Among all
permutations, return the one whose induced DAG has the fewest edges. The appeal is the assumption it
needs — u-frugality, "the true DAG is the uniquely sparsest Markovian DAG" — which is strictly weaker
than faithfulness. It survives precisely the near-cancellation cases that sink PC, because it asks a
global sparsity question rather than a local independence one: a faint dependence that hides an edge
just makes the true graph look sparser, and sparsity is exactly what is being optimized. The catch is
brutal: it enumerates all `p!` permutations, dead at nine variables. So the *robustness* lives in the
permutation view, and my whole problem becomes how to move through permutation space without
enumerating it.

Two things are needed: how to turn one `π` into a scored DAG cheaply and correctly, and how to walk
from one `π` to a better one. Take the projection first, because everything calls it. Given `π`, the
parents of variable `k` should be its Markov boundary within `pre(k)`, the smallest predecessor set
rendering `X_k` independent of the rest of the predecessors. I could draw `j → k` whenever a CI test
says they are dependent given the other predecessors — but that puts me right back into the fragile
CI-test regime that just sank PC. The escape is a *score*. I want a decomposable local score,
`score(G_π) = Σ_v score(X_v, Pa(v))`, with *local consistency*: in the large-sample limit, adding
`j → k` raises the score iff `X_j ⟂̸ X_k | Pa(k)`. That property means the score already encodes the
independence facts a CI test would give me, but as a comparison of two model scores rather than a
thresholded statistic — the same information, far more stable. For this linear Gaussian task the
right such score is the Gaussian SEM-BIC, `local_score_BIC_from_cov`, with the complexity multiplier
`lambda_value = 2` (a heavier penalty than the textbook BIC's 1, which on the dense scale-free graphs
guards against over-adding). It reads the data only through the covariance, so per-family scoring is
a cheap linear-algebra call.

Finding the Markov boundary by score is grow-then-shrink. Grow: start `M = ∅`, repeatedly add the
predecessor that most improves `score(X_k, M ∪ {Y})`, stop when no addition helps — by local
consistency this only pulls in `Y` when `X_k ⟂̸ Y | M`, i.e. genuinely informative predecessors,
and overshoots into a superset of the boundary. Shrink: from the grown set, repeatedly remove the
`Y` whose removal most improves the score, peeling off the redundancies — removing `Y` helps exactly
when `X_k ⟂ Y | M \ {Y}`. In the limit this returns the unique Markov boundary, and the per-node
shrink scores sum to `score(G_π)`. So I have `project(π)` and `score(π)` entirely from local score
calls, no CI tests — exactly the robustness PC lacked. And there is a structural guarantee I lean on:
every `G_π` is Markovian and subgraph-minimal, and a DAG is subgraph-minimal iff it is the
projection of *some* permutation, so the true MEC stays reachable inside the permutation image.

Now the real question PC could not answer: how to move through permutation space toward sparser DAGs.
The crudest move is Teyssier-Koller's adjacent transposition — swap two neighbors in `π` — whose one
virtue is that a single swap only changes the local scores of the two swapped variables, so it is
cheap. But it is *too* local: a single swap barely changes the induced DAG, so hill-climbing with
adjacent swaps stalls in shallow optima on exactly the hard, dense graphs I care about, and it has no
consistency guarantee. I need a move that leaps far enough in one step to escape those shallow optima
while staying tied to the equivalence-class structure a correctness argument would need. The handle
is Chickering's theory of covered edges. An edge `j → k` is *covered* when `j` and `k` share all
their other parents, `Pa(j) = Pa(k) \ {j}`; reversing a covered edge keeps the DAG in the same MEC,
and any two Markov-equivalent DAGs differing in `k` orientations are linked by exactly `k`
covered-edge reversals. So a search whose moves are covered-edge reversals can traverse an entire
equivalence class — the navigation a correctness proof rides.

The clean realization is to perform a covered-edge reversal *entirely in permutation space*, with no
detour through the DAG. Take `π` and a covered edge `j → k` in `G_π`. Write `π = ⟨δ1, j, δ2, k,
δ3⟩`. I want a new permutation whose induced DAG is `G_π` with `j → k` reversed. Slide `k` left past
each vertex of `δ2`: sliding past `i` is an adjacent transposition that leaves `G_π` unchanged
exactly when `i` is *not* a parent of `k`. And here "covered" earns its keep — if `j → k` is covered,
`k`'s parents are `j` plus `j`'s parents, all of which precede `j`, so no vertex of `δ2` is a parent
of `k`. So I can slide `k` all the way to just after `j`, every step DAG-preserving, and then the
single adjacent transposition swapping the now-adjacent `j` and `k` flips `j → k` to `k → j` and,
because reversing a covered edge stays in the MEC, lands the DAG with exactly that one arrow
reversed. That operation is the **tuck**: in general, split `δ2` into `γ` (the ancestors of `k`,
which must travel with `k`) and `γ^c` (the rest), and `tuck(π, j, k) = ⟨δ1, γ, k, j, γ^c, δ3⟩`. A
single tuck can fuse a reversal with edge deletions, so it is strictly more efficient than a
Chickering DAG-walk, and it never increases the edge count nor loses an independence.

This is the move PC never had: not a local single-edge verdict, but a structural reordering that
walks the equivalence class. The search is a depth-first traversal driven by tucks. From an initial
permutation I scan candidate edges and tuck them. A tuck that *strictly improves* the score is
accepted and the search restarts from the improved order; a tuck that is *score-neutral* (a
within-MEC reversal) is explored one level deeper, because a neutral move can set up a later strictly
improving one — this is the relaxation that lets the search cross plateaus that trap a pure
hill-climber. At the DFS root I am allowed to tuck *any* parent edge of a variable; deeper in the
recursion I restrict to *covered* tucks only, which keep me inside the current MEC while I look for an
exit, and a flip-set `history` records the set of reversed pairs so the within-MEC wandering cannot
loop forever. The depth bound matters: unbounded DFS carries the full correctness theorem, but in
finite samples a shallow `depth = 3` captures essentially all the benefit cheaply, so that is what I
run.

The implementation detail that makes the per-tuck re-scoring affordable is the Grow-Shrink Tree. A
tuck only perturbs a contiguous block of the permutation — the vertices between positions `j` and `i`
— so only those families change which predecessors they see; everything else is untouched. So I cache
each vertex's grow/shrink work in a `GST` keyed by the available predecessors, and after a tuck I
re-derive just the affected block against the cached trie rather than rerunning grow-shrink from
scratch. One `GST` per variable, built once over the score. The order object carries the working
permutation, each vertex's parents, its cached local score, and a running edge count; its `__init__`
seeds each vertex's score from `score.score(y, [])` — and here is a `causal-learn` convention worth
naming, because it is in the literal scaffold fill: that setup score is *negated* (`-score.score(y,
[])`), while the values that actually drive grow, shrink, and the DFS accept/reject are the
higher-is-better numbers returned by `GST.trace(...)`. I keep that exactly as the harness exposes it;
the negation is a bookkeeping quirk of the initializer, not the scoring used for moves.

When the DFS converges — a full pass with no improving or productive-neutral tuck — I read the DAG
off the final order (each vertex's selected parents) and convert it to a CPDAG with `dag2cpdag`,
since the data identify only the MEC. Note what is *not* here relative to the general method: there is
no explicit Backward-Equivalence cleanup phase bolted on, and there is no tier-by-tier escalation
exposed as a knob — the harness's fill runs the single tuck-DFS at `depth = 3` over the BIC-from-cov
score and reads off the CPDAG. The within-MEC/covered-tuck structure of the DFS is doing the work the
tier relaxation would do; I accept that the lean fill is the operational form here.

So the delta from PC is total: I have left the constraint-based world entirely. Instead of thousands
of independent thresholded Fisher-z verdicts, every decision is now a comparison of decomposable
model scores (local consistency), and instead of a fixed skeleton that an early wrong deletion
poisons, the search is a plateau-crossing DFS of covered/general tucks over orderings, robust under
an assumption weaker than the faithfulness that PC's recall violated. Reading PC's exact shape, the
falsifiable claims are on recall and on the dense scenarios. If the ordering/score view is the right
fix, the two small sparse graphs (ER10, ER20) — where PC was already near-perfect on precision and
only its recall sagged — should snap to essentially zero SHD, because there the score's global
sparsity search has no near-cancellation to fight. The real test is the dense and noisy rows where PC
collapsed: SF50, SF50-Hard, and especially ER20-Noisy (PC adjacency recall 0.246, SHD 69). I expect
the score-based search to lift adjacency recall sharply there — recovering the real edges PC's CI
tests deleted — with SHD falling by a large factor on SF50 and the noisy graphs. Where I am *not*
sure GRaSP wins cleanly is the very densest, lowest-sample regime: ER20-Noisy's 400 samples at noise
2.5 strain even a robust score, and the tuck-DFS's plateau-crossing can still wander on a graph where
the sparse-Markov razor itself is near its limit — so if anything survives as the next failure mode,
I expect it to be the densest/noisiest scenario, not the scale-free ones. That is the row to watch
against the next rung. The full scaffold module — `_GraspOrder`, the tuck/ancestors/update helpers,
the `_grasp_dfs` with the covered-edge restriction and flip-set history, and the BIC-from-cov
`run_causal_discovery` at `depth = 3` — is in the answer.
