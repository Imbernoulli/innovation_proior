I want a step function whose worst overlap against its own complement is as small as possible — that is
the whole game, and lower is better. But before I try to be clever, I should fix the floor: the simplest
legal profile, because I need to know what a value of the overlap even feels like on this problem, and I
need a baseline that every later rung has to beat. The simplest function in the class is the flat one:
every cell half-full, `v_i = 1/2` for all `i`. It is the only profile with no internal structure at all,
and its overlap I can reason about entirely by hand.

So let me do that, because the point of this rung is to *understand* the functional, not just read a
number off the evaluator. The complement of the flat profile is itself: if `v_i = 1/2` everywhere, then
`1 − v_i = 1/2` everywhere too. The overlap at a shift `k` is the sum over the overlapping cells of
`v_i · (1 − v_{i−k})`, which is a sum of `(1/2)·(1/2) = 1/4` over however many cells overlap at that
shift. The number of overlapping cells is largest at zero shift — there all `n` cells line up — so the
worst overlap is `n` cells each contributing `1/4`, i.e. `n/4`. The evaluator rescales by `2/n`, giving
`(n/4)·(2/n) = 1/2`. The flat profile scores exactly `1/2`, and — crucially — this is *independent of
how finely I discretize it*. A flat vector of `10` half-cells and a flat vector of `1000` half-cells both
have the same triangular overlap envelope peaking at the center, and both score `1/2` exactly. The number
of pieces is a red herring on its own; only the *shape* of the heights moves the bound.

I should sanity-check the balance constraint while I am here, because it is the one thing that makes this
problem non-trivial. The rule is `Σ v_i = n/2` — exactly half the total cell-mass on the `A` side. The
flat profile has every `v_i = 1/2`, so its sum is `n·(1/2) = n/2`, exactly on the constraint. Good: the
flat profile is feasible, sitting right at the center of the feasible region. That is reassuring, because
it means the floor I am about to measure is a genuine interior point, not some degenerate corner.

That invariance to discretization is the real lesson of this rung, and it tells me something about why
the problem is hard. The piece count `n` is not itself a lever — refining a flat profile buys nothing,
because every refinement of `1/2`-cells is still all `1/2`-cells. What moves the bound is making the
overlap envelope *less peaked at the worst shift*: I want the worst `c_k` to come down, which means I want
the mass of `A` to align *badly* with the complement-mass of `B` at every shift simultaneously, so that
no single shift can pile up a large overlap. The flat profile does exactly the wrong thing — at zero shift
every cell aligns perfectly with itself, and since `A`-mass and `B`-mass are identical halves everywhere,
that perfect alignment is also the worst overlap. To beat `1/2` I have to break the symmetry: push some
cells toward `0` and others toward `1`, so that where `A` is heavy `B` is light and the products
`v_i(1 − v_{i−k})` cannot all be large at once.

I notice the structure of what I am fighting. The constraint forces the *average* of `v` to be `1/2`, so
I cannot lower every product by globally shrinking the heights — the mass has to go somewhere. What I can
do is *redistribute* it: a cell at `1` contributes nothing to the overlap when aligned with another cell
at `1` (because `1 − 1 = 0`), and a cell at `0` contributes nothing when it is the `v_i` factor. So a
spiky, near-binary profile — cells near `0` or near `1` rather than all at `1/2` — can make many of the
products vanish, and the art is arranging the spikes so that the *worst* shift still has a small total.
But none of that is available to a flat vector, which has nothing to vary. The flat function is a strict,
rigid baseline: it is the unique maximally-symmetric feasible profile, parked at the top of the achievable
bound-range with no internal degree of freedom to spend.

I should also pin the ceiling honestly so I know how far I am from the bottom before I start descending.
On the *lower* side, the true constant `C5` is provably at least `0.379005` (White, by convex
programming), so no construction can ever certify a bound below that. On the *upper* side, the published
record is around `0.380869`, reached by elaborate optimized step functions with several hundred pieces.
So the whole window the constant lives in is `0.379005 ≤ C5 ≤ 0.380869`, a sliver of width about
`2×10^{-3}`. My flat floor at `0.5` is `0.12` above the top of that window — an enormous distance in the
units of this problem — and essentially *all* of it has to be bought by optimizing the heights into a
non-trivial, asymmetric, near-binary profile. The flat function buys none of it; it just establishes the
starting altitude.

What I expect from the evaluator, then, is `C = 0.5` exactly, confirming both the by-hand overlap
analysis and the discretization invariance — and confirming that the scaffold's notion of "flat = floor"
matches my hand computation, so I can trust the harness on the harder rungs. The limitation this rung
exposes is sharp and sets up the next one: the flat profile is a local nothing, with a worst overlap
locked to `1/2` by perfect self-alignment at zero shift, and no direction to move because every cell is
identical. To descend at all I have to introduce *variation* among the heights and let some search
procedure discover which non-flat profile flattens the overlap envelope — pushing the heights toward `0`
and `1` while holding the sum at `n/2`. The cleanest place to start that search is at a small piece count,
where the height vector is short enough that a constrained optimizer can explore the shape space
thoroughly and find the kind of profile that already clears `0.39`, a first real drop off the `0.5`
floor.
