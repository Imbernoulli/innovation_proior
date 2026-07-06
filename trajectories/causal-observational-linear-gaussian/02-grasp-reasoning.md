PC's numbers tell me exactly what I feared, and the pattern is the lever. On the two small sparse
graphs PC is fine on adjacency precision — 1.0 on ER10, 0.857 on ER20 — but its *recall* is already
sagging (0.857, 0.585), and on the larger and denser scenarios it collapses: SF50 recall 0.51,
SF50-Hard recall 0.30, ER20-Noisy recall a catastrophic 0.246 with adjacency precision still 0.85.
Read that precision/recall split: PC is not adding junk edges, it is *missing real ones*, and it
misses more as density and noise climb. I want to make that quantitative rather than leave it as a
gestalt, so let me back the edge counts out of the metrics. The generator's expected edge counts are
`p·C(n,2)` for the ER graphs and `m·(n−m)` for the scale-free ones: about 13.5 for ER10, 38 for
ER20, 96 for SF50, 141 for SF50-Hard, and 66 for ER20-Noisy. With `recall = TP/(TP+FN)` and
`precision = TP/(TP+FP)` I can recover the confusion counts. On ER20-Noisy: `TP ≈ 0.246 × 66 ≈ 16`
true edges found, `FN ≈ 50` real edges missed, and from precision 0.85 the estimate has about
`16/0.85 ≈ 19` edges total, so only `≈ 3` false positives. Fifty missed against three spurious — the
`SHD` of 69 is almost entirely a hole where true edges should be. SF50 tells the same story: `TP ≈
0.51 × 96 ≈ 49`, `FN ≈ 47`, `FP ≈ 12`; SF50-Hard is starker, `TP ≈ 42`, `FN ≈ 99`, `FP ≈ 25`. Across
every hard scenario the missed-edge count dwarfs the false-edge count. That is the exact signature I
predicted — a single thresholded Fisher-z test deletes a true edge whenever the partial correlation
of a real-but-faint dependence sits near zero, and on dense graphs (more paths, more
near-cancellations) and high noise (ER20-Noisy at noise 2.5, only 400 samples) that happens all over
the graph at once.

The arrow numbers are even worse, and they are worse in a way that confirms the cascade rather than
just echoing it. SF50 arrow recall is 0.19 against adjacency recall 0.51; ER20-Noisy arrow recall is
0.044 against adjacency recall 0.246. Arrow recall can never exceed adjacency recall — I cannot
orient an edge I never found — so half the arrow loss is just the missing skeleton. But the arrow
number sits *well* below that ceiling: on SF50 I recovered about 51% of the adjacencies yet only 19%
of the arrowheads, so even among the edges I did find, most colliders came out wrong. That is the
downstream half of the failure I flagged — a missed edge unshields a triple that should have stayed
shielded, phase two reads a collider off a corrupted separating set, and Meek's closure then
propagates that one wrong arrowhead across the neighborhood. The `SHD` line says the whole thing in
one number — 8, 31, 93, 143, 69 — climbing with density, and now I can read each entry as
mostly-missed-edges plus an amplified arrow error on top. PC's correctness theorem is true; the
finite-sample assumption it rests on, faithfulness, is what breaks, and it breaks exactly where edges
crowd together. So I do not want a faster faithful algorithm. I want decisions that are robust to
faint-but-real dependences, and I want them to scale to the fifty-node graphs where PC fell apart.

What tolerates almost-unfaithfulness? Before I answer with a method I should be honest about what
"robust" can and cannot mean here, because it is easy to tell myself a comforting story that the
arithmetic won't support. The tempting story is: "swap the thresholded CI test for a smarter,
lower-threshold criterion and the faint edges come back." Let me check whether a score can even be
more sensitive per edge than Fisher-z was. For a linear Gaussian family, adding one parent `j → k`
improves the log-likelihood by about `−(n/2)·log(1 − ρ²)` where `ρ` is the partial correlation of the
new edge given the current parents, and a BIC-style penalty charges `λ·log(n)` per added parameter,
so a score keeps the edge only when `n·ρ² ≳ λ·log(n)`, i.e. down to `ρ_BIC ≈ √(λ·log(n)/n)`. With the
complexity multiplier `λ = 2` I intend to use, that floor is about `0.087` at SF50's `n = 2000` and
about `0.17` at ER20-Noisy's `n = 400`. Fisher-z at `α = 0.05` detected down to about `0.044` and
`0.10` on those same rows. So a heavier-penalty score is *less* sensitive per edge than the CI test
was, not more — if per-edge detectability were the whole game, moving to a score would make recall
*worse*. That kills the comforting story and forces me to locate the actual source of PC's failure
somewhere other than raw per-edge sensitivity. And it is locatable: PC's fatal property is not the
threshold height, it is *irreversibility plus locality*. PC deletes an edge the first time any one
subset spuriously separates its endpoints, and it never revisits that deletion; the deletion then
shrinks a neighborhood so a neighboring true edge becomes untestable, and the error ramifies. A
method that re-derives each variable's entire parent set from scratch every time it reconsiders the
structure — never committing an irreversible local deletion — can recover a locally-faint edge the
moment the global picture prefers keeping it, even with a *higher* per-edge floor, because it is
never trapped by an early mistake. So the robustness I am chasing is cascade-freedom and global
re-optimization, not a lower threshold, and I should build toward a search that reconsiders whole
neighborhoods rather than one that thresholds one statistic at a time.

The line of work that fits that description searches *orderings* instead of edges. Take a permutation
`π` of the variables, treat it as an acyclic order, and build a DAG `G_π` by giving each variable
parents drawn only from the variables ahead of it in `π`. Among all permutations, return the one
whose induced DAG has the fewest edges. The appeal is the assumption it needs — u-frugality, "the
true DAG is the uniquely sparsest Markovian DAG" — which is strictly weaker than faithfulness. It
survives precisely the near-cancellation cases that sink PC, because it asks a global sparsity
question rather than a local independence one: a faint dependence that hides an edge just makes the
true graph look slightly sparser, and sparsity is exactly what is being optimized, so the search is
not thrown by a single partial correlation dipping under a threshold. The catch is brutal: it
enumerates all `p!` permutations, dead at nine variables — `9! ≈ 3.6×10⁵`, `20!` beyond any budget,
`50!` not worth writing down. So the *robustness* lives in the permutation view, and my whole problem
becomes how to move through permutation space without enumerating it. That reframing also lets me set
aside the other obvious candidate, a straight score-based climb like GES over equivalence classes
with local insert/delete operators. GES has the robust score I want, but its moves are still
edge-local greedy edits, and its forward phase is exactly the kind of process that can over-insert on
a dense graph and then fail to back the additions out — it swaps PC's one weakness (irreversible
deletion) for a different one (greedy insertion into a local optimum) without giving me the
whole-neighborhood reconsideration that the ordering view does structurally. The ordering search
keeps the robust score *and* gets cascade-freedom for free, so that is the lane.

Two things are needed to make the ordering view operational: how to turn one `π` into a scored DAG
cheaply and correctly, and how to walk from one `π` to a better one. Take the projection first,
because everything calls it. Given `π`, the parents of variable `k` should be its Markov boundary
within `pre(k)`, the smallest predecessor set rendering `X_k` independent of the rest of the
predecessors. I could draw `j → k` whenever a CI test says they are dependent given the other
predecessors — but that puts me right back into the fragile CI-test regime that just sank PC. The
escape is a *score*. I want a decomposable local score, `score(G_π) = ∑_v score(X_v, Pa(v))`, with
*local consistency*: in the large-sample limit, adding `j → k` raises the score iff `X_j ⟂̸ X_k |
Pa(k)`. That property means the score already encodes the independence facts a CI test would give me,
but as a comparison of two model scores rather than a thresholded statistic — the same information,
and crucially reversible, because I compute it fresh from the current parent set every time rather
than reading a frozen verdict. For this linear Gaussian task the right such score is the Gaussian
SEM-BIC, `local_score_BIC_from_cov`, with the complexity multiplier `lambda_value = 2` — the same
heavier-than-textbook penalty whose per-edge floor I just computed. I keep it at 2 deliberately: on
the dense scale-free graphs the number of candidate parents is large, and a textbook `λ = 1` would
let the search over-add spurious families; the arithmetic above says `λ = 2` costs me some raw
per-edge sensitivity, but the cascade-freedom is what is doing the recall work, so I would rather
spend the penalty on guarding precision than chase a lower floor I do not need. It reads the data
only through the covariance, so per-family scoring is a cheap linear-algebra call.

Finding the Markov boundary by score is grow-then-shrink, and I want to watch it work on a tiny case
so I trust it. Grow: start `M = ∅`, repeatedly add the predecessor that most improves `score(X_k, M ∪
{Y})`, stop when no addition helps — by local consistency this only pulls in `Y` when `X_k ⟂̸ Y | M`,
i.e. genuinely informative predecessors, and overshoots into a superset of the boundary. Shrink: from
the grown set, repeatedly remove the `Y` whose removal most improves the score, peeling off the
redundancies — removing `Y` helps exactly when `X_k ⟂ Y | M \ {Y}`. Take an order that puts `X` and
`W` ahead of `Y`, with the truth `X → Y` and `W → Y` and `X`, `W` marginally independent. Scoring
`Y`: grow adds whichever of `X`, `W` most improves the fit, say `X`; then with `M = {X}`, `W` still
improves the fit because `W ⟂̸ Y | X` (both are genuine parents), so grow adds `W` too; no third
predecessor helps, and shrink removes neither because dropping either loses a real parent. The
boundary comes back `{X, W}`, the correct parent set, from local score calls alone — no CI test in
sight. In the limit this returns the unique Markov boundary for any order, and the per-node shrink
scores sum to `score(G_π)`. So I have `project(π)` and `score(π)` entirely from local score calls —
exactly the reversible, whole-neighborhood evaluation PC lacked. And there is a structural guarantee
I lean on: every `G_π` is Markovian and subgraph-minimal, and a DAG is subgraph-minimal iff it is the
projection of *some* permutation, so the true MEC stays reachable inside the permutation image.

One more thing I want to confirm before I trust this route: that I am not quietly giving up PC's
asymptotic correctness in exchange for finite-sample robustness. In the population limit the score's
local consistency says adding `j → k` helps iff `X_j ⟂̸ X_k | Pa(k)`, which is the *same*
independence fact PC's Fisher-z would resolve, so grow-shrink on the true order recovers the true
Markov boundaries and `G_π*` is the true DAG — the identifiable target is unchanged, still the
skeleton-plus-colliders MEC I derived at step one. Take the collider skeleton again, `X → Y ← Z` with
`X ⟂ Z`: on any order that places `X` and `Z` before `Y`, growing `Y` pulls in both because each is
dependent on `Y` given the other, and shrink drops neither, so the projection reconstructs the two
edges and the collision at `Y` — exactly the structure PC's separating-set lookup would orient. So
the two methods aim at the identical object; what differs is purely the finite-sample path to it,
where PC thresholds one statistic irreversibly and the ordering search compares whole-family fits and
can revisit. That is the trade I want — keep the target, change the robustness — and it means any
metric gain over PC is attributable to the finite-sample behavior, not to aiming somewhere new.

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
covered-edge reversals. I can check the reversal claim on the smallest case: `X → Y` with no other
parents is trivially covered (`Pa(X) = ∅ = Pa(Y)\{X}`), and reversing it to `Y → X` gives the same
skeleton `{X — Y}` and no collider either way, so both DAGs sit in the same one-element MEC whose
CPDAG is the single undirected edge `X — Y`. The reversal moved through the equivalence class without
leaving it — which is exactly the navigation a correctness proof rides.

The clean realization is to perform a covered-edge reversal *entirely in permutation space*, with no
detour through the DAG. Take `π` and a covered edge `j → k` in `G_π`. Write `π = ⟨δ1, j, δ2, k,
δ3⟩`. I want a new permutation whose induced DAG is `G_π` with `j → k` reversed. Slide `k` left past
each vertex of `δ2`: sliding past `i` is an adjacent transposition that leaves `G_π` unchanged
exactly when `i` is *not* a parent of `k`, because the only families whose predecessor sets change
are those of `k` and `i`, and if `i ∉ Pa(k)` neither one's Markov boundary moves. And here "covered"
earns its keep — if `j → k` is covered, `k`'s parents are `j` plus `j`'s parents, all of which
precede `j`, so no vertex of `δ2` is a parent of `k`. So I can slide `k` all the way to just after
`j`, every step DAG-preserving, and then the single adjacent transposition swapping the now-adjacent
`j` and `k` flips `j → k` to `k → j` and, because reversing a covered edge stays in the MEC, lands
the DAG with exactly that one arrow reversed. That operation is the **tuck**: in general, split `δ2`
into `γ` (the ancestors of `k`, which must travel with `k`) and `γ^c` (the rest), and `tuck(π, j, k)
= ⟨δ1, γ, k, j, γ^c, δ3⟩`. A single tuck can fuse a reversal with edge deletions, so it is strictly
more efficient than a Chickering DAG-walk, and it never increases the edge count nor loses an
independence.

This is the move PC never had: not a local single-edge verdict, but a structural reordering that
walks the equivalence class. The search is a depth-first traversal driven by tucks. From an initial
permutation I scan candidate edges and tuck them. A tuck that *strictly improves* the score is
accepted and the search restarts from the improved order; a tuck that is *score-neutral* (a
within-MEC reversal) is explored one level deeper, because a neutral move can set up a later strictly
improving one — this is the relaxation that lets the search cross plateaus that trap a pure
hill-climber. At the DFS root I am allowed to tuck *any* parent edge of a variable; deeper in the
recursion I restrict to *covered* tucks only, which keep me inside the current MEC while I look for an
exit, and a flip-set `history` records the set of reversed pairs so the within-MEC wandering cannot
loop forever. The depth bound matters: unbounded DFS carries the full correctness theorem, but the
recursion branches, so cost grows steeply with depth, and in finite samples a shallow `depth = 3`
captures essentially all the benefit cheaply — the first level does the strict improvements, the
next two buy the plateau-crossing that a pure hill-climber lacks, and beyond that the marginal
recovered structure does not pay for the branching. So `depth = 3` is what I run.

The implementation detail that makes the per-tuck re-scoring affordable is the Grow-Shrink Tree. A
tuck only perturbs a contiguous block of the permutation — the vertices between positions `j` and `i`
— so only those families change which predecessors they see; everything else is untouched. So I cache
each vertex's grow/shrink work in a `GST` keyed by the available predecessors, and after a tuck I
re-derive just the affected block against the cached trie rather than rerunning grow-shrink from
scratch. That locality is what keeps the fifty-node scenarios feasible: a tuck near the front of a
50-long order might re-score a block of a dozen families, not all fifty, and the `GST` turns each of
those re-scores into a trie descent rather than a fresh covariance solve. One `GST` per variable,
built once over the score. The order object carries the working permutation, each vertex's parents,
its cached local score, and a running edge count; its `__init__` seeds each vertex's score from
`score.score(y, [])` — and here is a `causal-learn` convention worth naming, because it is in the
literal scaffold fill: that setup score is *negated* (`-score.score(y, [])`), while the values that
actually drive grow, shrink, and the DFS accept/reject are the higher-is-better numbers returned by
`GST.trace(...)`. I keep that exactly as the harness exposes it; the negation is a bookkeeping quirk
of the initializer, not the scoring used for moves, and getting it wrong would silently invert the
search, so it is worth stating that I checked which number drives the comparisons.

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
sparsity search has no near-cancellation to fight and no plateau deep enough to trap it. The real
test is the dense and noisy rows where PC collapsed: SF50, SF50-Hard, and especially ER20-Noisy (PC
adjacency recall 0.246, `SHD` 69, and I backed out about 50 missed edges against only 3 false ones).
I expect the score-based search to lift adjacency recall sharply there — recovering the real edges
PC's CI tests deleted — with `SHD` falling by a large factor on SF50 and the noisy graphs. Where I am
*not* sure GRaSP wins cleanly is the very densest, lowest-sample regime, and now I can say why in the
score's own terms. On ER20-Noisy the BIC penalty floor I computed sits around `0.17` — higher than
anywhere else — so the score is at its most conservative exactly where the data are thinnest, and the
tuck-DFS's within-MEC wandering, which is an asset when it crosses a plateau toward a sparser truth,
becomes a liability when the sparse-Markov razor itself is near its limit and there is no clearly
sparser truth to cross toward: the neutral excursions can drift the search into a *denser*-than-truth
optimum instead. So if PC's problem was deleting real edges, I would not be shocked if GRaSP's residual
problem on that one row is the opposite — over-adding — with adjacency precision, not recall, as the
metric that gives it away. That is the row to watch against the next rung. The full scaffold module —
`_GraspOrder`, the tuck/ancestors/update helpers, the `_grasp_dfs` with the covered-edge restriction
and flip-set history, and the BIC-from-cov `run_causal_discovery` at `depth = 3` — is in the answer.
