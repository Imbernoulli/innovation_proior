The greedy region vindicated the representation and indicted its engine, and the rung-2 table puts
numbers on both. The cell-built net beat the rectangle where it mattered: seed 2 from `2572` to
`3188`, seed 1 from `2134` to `2362`. But how seed 2 won exposes the engine. The rung-1 box held
`4366` mackerel and `1795` sardine; the grown region came in at `3618` and `431`. So the greedy shed
`1364` sardine but paid `748` mackerel to do it — a single connected blob with no remove move
orphans the mackerel stranded on the far side of a carved pocket. The net gain was real, `+616`, but
only about a third of the recoverable sardine turned into score, and `748` mackerel were sacrificed.
That sacrifice is the measured price of irreversibility.

The rest of the table agrees. Seeds 3 and 4 fell back to the box because they are already clean —
seed 4 at `b = 114` has nothing to carve. Seed 5 dipped `8` points below rung 1: it carries the
second-most recoverable sardine yet the one-shot greedy failed to find a carve beating the box and
fell back. And the two seeds that carved, 1 and 2, both saturated their perimeter (`393334`,
`393332`) on ragged boundaries — the greedy spends its length on whatever boundary it stumbles into
and has nothing left to repair it. So I keep the representation and replace the greedy with local
search that can add *and* remove cells and is willing to step downhill. This is the benchmark's
strong baseline: simulated annealing with cheap incremental scoring, which took ALE-Agent to fifth
place at `2880`.

The move set answers the greedy's deficiency directly: at each step add an outside cell adjacent to
the region or remove a boundary cell — a single-cell flip. The remove is the move the greedy never
had, letting the search release a sardine-heavy cell; the add lets it re-capture an orphaned
mackerel. Each flip changes `a − b` by exactly that cell's count, so with precomputed per-cell
mackerel and sardine counts any candidate's delta is an O(1) lookup. A block move — a whole row at
once — could vault a plateau, but its score becomes `O(block)`, its validity a whole-boundary check,
and its perimeter a full recomputation; annealing convergence depends on sheer proposal volume, so I
keep the single-cell flip, the one move for which both score and validity are O(1).

O(1) scoring is only half of it. What forced the greedy to be one-shot was its `O(G²)` hole check,
which a search proposing tens of millions of flips cannot pay, so the whole viability rests on a
*local* validity test. Perimeter I handle with the running `4 − 2s` edge count. Connectivity and
hole-freeness together come from the digital-topology simple-point test: a cell can be flipped
without changing topology iff, walking the eight-cell ring around it, the foreground transitions
from outside to inside exactly once. An isolated add sees zero transitions (rejected — it would be a
second component); an add bridging two arms sees two (rejected — it merges components or seals a
hole); an ordinary staircase step sees one. The one trap it misses is two cells touching only at a
corner, whose boundary pinches into a figure-eight; the evaluator demands a simple polygon, so I add
a `2 × 2` diagonal-pinch guard forbidding that checkerboard. Both checks are O(1) — the property the
flood-fill denied the greedy.

Now the search. I anneal: propose a random legal flip, take it if it improves `a − b`, else take it
with probability `exp(Δ/T)`, cooling `T` geometrically. I calibrate temperature to the actual scale
of `Δ`: with `10^4` fish over a `50 × 50` grid a cell holds about four fish, so a typical boundary
flip moves `a − b` by a single-digit amount. At `T0 = 8` a `Δ = −4` downhill flip is accepted with
`exp(−0.5) ≈ 0.61`, so the region wanders freely early and reshapes out of the greedy's basins; at
`T1 = 0.05` that same flip has probability `≈ 0`, so only improving flips survive and the region
settles. I keep the best region ever seen.

The flip budget makes or breaks the bet and forces a language decision. Annealing needs the region
proposed-at millions of times, each flip a few dozen operations; in C++ that is tens of nanoseconds,
so a `~5`-second budget buys `10^7`–`10^8` flips where Python would deliver `10^5`–`10^6` — far too
few for the temperature to cool through a meaningful anneal. The flip volume *is* the method, so I
move to C++ with a fast xorshift RNG.

Where to start? Perimeter is the hard limit, and a rectangle is the most perimeter-efficient shape
there is — a compact region of area `A` spends only about `4√A` on its outline. Growing from a dot
would burn the flip budget just inflating a blob. So I warm-start from the best perimeter-constrained
rectangle (prefix-sum sweep) and let the search spend its budget on what it is uniquely good at:
cutting notches to release sardine and extending to grab outlying mackerel. On seed 2 this starts
already holding the `4366` mackerel the box caught, so the job is to shed sardine while re-adding
orphaned mackerel. Sitting on a `−6` sardine-heavy inside cell with a `+7` mackerel cell just
outside, the search removes the `−6` and adds the `+7` for a `+13` swing — the *sequence* the
one-shot greedy structurally cannot perform, and at high temperature it will even accept a worse
intermediate to get from one to the other.

The two directions are asymmetric. An add with one or two inside neighbors holds or lengthens the
boundary (`4 − 2s` is `0` or `+2`), so adds are gated by perimeter and refused near the limit; a
remove almost always shortens the boundary and is essentially never blocked. So a region pressed
against its length limit — as seeds 1 and 2 are — can freely shed but struggles to extend, drifting
toward shedding sardine rather than capturing outlying mackerel. That is fine for seed 2, whose
problem is excess sardine, but wrong for seed 1, whose remaining gain lives in mackerel just outside
a saturated boundary.

I fix `G = 50` — cell side `2000`, which divides `10^5` exactly. The search reuses one grid for
millions of flips so I cannot try three resolutions; `50` is fine enough to carve useful notches and
coarse enough that the budget still wraps a large region. Because it divides evenly, every boundary
unit-edge is exactly `2000`, so the traced perimeter is `bedges × 2000` — an identity, unchanged by
collinear merging — which makes the running edge count an exact perimeter gate rather than a
heuristic. I hold a safe margin of `20` cell-sides under the cap, and a final true-perimeter check
falls back to the best rectangle if a traced polygon is ever degenerate, so the rung never emits an
illegal net.

Warm-starting from the box makes the seed predictions testable. Since the search stores only
improvements, seeds 3 and 4 should not merely tie the box but can be carved slightly past it. Seed 2
should gain most: the search can recover much of the `748` mackerel the greedy sacrificed while
holding sardine low, pushing `a` back toward `4366`. Seed 5 should improve modestly or hold. Seed 1
is the one I worry about — it does not warm-start from the greedy's strong `50`-vertex carve but from
the box (`a = 2643`), and must rediscover the carve by undirected random flips under a slightly
tighter safe budget, so it could land at or below rung 2 even as the mean rises. If it does, that is
a symptom of the next ceiling, not a defect of the method.

That ceiling is already visible in how I sample. To propose an add I pick a random region cell and a
random neighbor; to propose a remove I pick a random region cell. This is cheap and structureless
but blunt: a remove is only legal on a boundary cell, yet I sample uniformly over all region cells,
so once the region is large most remove proposals land on interior cells and are thrown away, and an
add toward a random neighbor is as likely to point at already-correct boundary as at a misclassified
fish. Each wasted proposal is O(1) and I have tens of millions, but the fraction aimed at an actual
error shrinks as the region grows. Compounded with the perimeter gate refusing the extend-moves seed
1 needs, that undirectedness is what I expect to surface as a plateau.
