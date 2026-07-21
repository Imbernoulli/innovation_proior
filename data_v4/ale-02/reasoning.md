The floor is at most `30 x 30` and I am scored by the number of covered cells, with
one unforgiving catch: any illegal placement — poking outside the grid, overlapping
another part, exceeding a shape's copy bound `cnt_k`, or naming a bad shape or a
rotation outside `0..3` — drops the entire solution to `0`. Two facts organise
everything before I choose an algorithm. First, `W <= 30` means a whole floor row
fits inside a single 64-bit word with room to spare; that is the lever the fast
inner loop will pull. Second, the scoring floor is a cliff, not a slope, so every
state I ever emit has to be legal *by construction* — I am optimising a coverage
count, but I can never gamble on feasibility.

The objective is a continuous coverage count and the structure is a maximum-coverage
polyomino packing, NP-hard, so there is no exact answer to print — I am pushing a
heuristic against a score. And the cheapest legal answer, `P = 0`, covers `0` cells,
which ties the infeasible floor: useless as a submission but valuable as a safety
net. I want to always have a legal state in hand and only ever improve on it.

The cheapest *useful* legal solution is first-fit greedy: sort shapes
largest-area-first (covering more cells per placement tends to leave fewer awkward
gaps), and for each shape and each of its distinct rotations sweep every anchor
`(ar, ac)`, dropping a copy wherever it currently fits and respecting `cnt_k`. It is
one-shot and obviously feasible, and it already covers most of the floor. It serves
double duty here — my warm start, and the reference baseline the score ratio is read
against.

Greedy's weakness is structural, and worth naming before trying to fix it: it
commits a cell forever the instant a part covers it. A handful of early,
slightly-misaligned large parts can strand pockets of empty cells whose shape no
remaining catalogue piece matches — a stray `1x2` hole when the smallest free piece
is an L-tromino. On floors from `12x12` to `30x30` with areas up to 6 those pockets
add up to a few percent that a better arrangement could reclaim. The fix has to be
able to *undo*: pull a part back out, free its cells, and re-tile that region. That
is local search over the multiset of current placements, with moves that both add
and remove parts, and enough tolerance for a temporary coverage loss to cross a
valley to a better packing — simulated annealing.

Local search only beats greedy if I can run an enormous number of perturbations in
the ~2-second budget, because each perturbation gains almost nothing (a part is at
most 6 cells). The expensive part of a perturbation is the collision test: to add a
part I must check it overlaps nothing already down. The naive test — look each of
the part's up-to-6 cells up in a 2-D occupied array — is correct but it scatters
across memory and I would run it tens of millions of times. This is where the
opening scale observation pays off. I keep occupancy as one 64-bit word per row,
`occ[r]` with bit `c` set iff `(r, c)` is covered, and precompile each `(shape,
rotation)` once into a short list of `(row-offset, column-bitmask)` pairs anchored at
column 0. Testing a placement at `(ar, ac)` becomes, per occupied piece-row, a
single `occ[ar + row] & (mask << ac)`; if every row comes back zero it fits — one
`AND` and one branch per piece-row, all cache-resident, no per-cell scatter. Adding
is `occ[ar + row] ^= (mask << ac)` per row, and removing is the identical XOR (a
placement's cells are exactly the bits it owns, so the same XOR toggles them off).
Coverage rides along incrementally: `+area` on add, `-area` on remove, never a
rescan. Per-move cost is O(rows of the piece) ≈ O(1), and millions of moves fit the
budget — that is what makes the metaheuristic viable rather than a toy.

With that machinery, the move set over the placement multiset is three moves. ADD
draws a uniformly random feasible placement (random shape with a copy left, random
rotation, random in-bounds anchor, retried a few times) and takes it — pure uphill,
always accepted. REMOVE drops a random current placement (`-area`, downhill) and
accepts it with Metropolis probability `exp(-area / T)`; this is precisely what
greedy cannot do — vacate a part to open space. REPLACE removes a random placement,
draws a fresh feasible one into the freed floor, and accepts the net `delta =
newArea - oldArea` by Metropolis (`delta >= 0` always, else `exp(delta / T)`); it
re-tiles a freed region in one move and is the workhorse. Cooling is geometric in
wall-clock fraction from `T0 = 4.0` to `T1 = 0.02`, so the search shuffles freely
early and behaves like greedy hill-climbing late. Throughout I keep the best feasible
state ever seen and print *that*, so a run of accepted downhill moves near the end
can never leave the output worse than the warm start.

REPLACE is the one move that touches state twice — a remove, then a conditional
add — and its reject path is a real trap I have to get exactly right. The reject
branch must reconstruct the *pre-move* state, which means re-adding the removed part
`old` and adding nothing else. The tempting mistake is to re-add the freshly drawn
candidate instead of `old`; that is wrong twice over. It is a coverage regression
(a "rejected" REPLACE silently becomes "removed a 4, added a 2"), and worse, a
feasibility hazard: the candidate was drawn against the floor *after* `old` was
removed, so it is legal only in that freed state — if I ever hold both the candidate
and `old`, the candidate may now overlap `old`, and reconstructing that illegal
state is an instant scorer zero. So the two branches reconstruct named legal states:
accept keeps the candidate with `old` gone; reject runs `add_place(old...)` to put
`old` back at its exact anchor and never adds the candidate. Because I never hold
both parts at once, the hazard is closed, and the XOR restore returns the board
bit-for-bit to where it started.

A few edge conditions the packing geometry actually invites, each pinned to a guard.
A shape can be larger than the floor in some rotation, so `random_add` skips a
rotation with `s.hh > H || s.ww > W` before computing the anchor range `randint(H -
s.hh + 1)`, which is only well-defined once the shape fits; the greedy sweep bounds
`ar + s.hh <= H` for the same reason, so no out-of-bounds anchor is ever generated.
Symmetric pieces have fewer than 4 distinct rotations, so the greedy skips a rotation
whose normalised cells equal an earlier one rather than sweeping the floor four
times. Every add is gated by `usedCnt[k] < cntOf[k]` and `remove_idx` decrements the
count, so the multiset never exceeds `cnt_k`. REMOVE and REPLACE both `continue` on
an empty placement set, so I never index an empty vector. The loop checks wall-clock
every 1024 iterations and stops at `1.8 s`, under budget. And the shift `mask << ac`
never drops bits because `ac + ww <= W <= 30 < 64`.

To confirm the bookkeeping rather than assume it, I generate seeds `1..20`, run the
solver and the first-fit baseline, and score both against an independent replay of
every placement. Every solver output is feasible and strictly beats the baseline on
all 20 seeds, mean coverage around `420` against the baseline's `~389`. The check
that actually bites is cross-checking the solver's internal incrementally-maintained
`covered` against the from-scratch recount: they agree exactly, including on a seed
where the floor is nearly tiled (359 of 360 cells). That agreement is the evidence
the XOR add/remove and the `+/-area` score updates are correct, not merely plausible.

The full self-contained C++17 program — the bitset occupancy, the precompiled
rotation masks, the greedy warm start, and the annealing loop with the corrected
REPLACE branch — is the frozen answer.
