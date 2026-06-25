I want a step function whose worst overlap against its own complement is as small as possible — that is
the whole game, and lower is better. But before I try to be clever, I should fix the floor: the simplest
legal profile, because I need to know what a value of the overlap even feels like on this problem, and I
need a baseline that every later rung has to beat. The simplest function in the class is the flat one:
every cell half-full, `v_i = 1/2` for all `i`. It is the only profile with no internal structure at all,
and its overlap I can work out entirely by hand and then check against the evaluator.

So let me do that, because the point of this rung is to *understand* the functional, not just read a
number off the evaluator. The complement of the flat profile is itself: if `v_i = 1/2` everywhere, then
`1 − v_i = 1/2` everywhere too. The overlap at a shift `k` is the sum over the overlapping cells of
`v_i · (1 − v_{i−k})`, which is a sum of `(1/2)·(1/2) = 1/4` over however many cells overlap at that
shift. The number of overlapping cells is largest at zero shift — there all `n` cells line up — so I'd
predict the worst overlap is `n` cells each contributing `1/4`, i.e. `n/4`, and the evaluator rescales by
`2/n`, giving `(n/4)·(2/n) = 1/2`.

I don't want to trust that prediction blind, so let me actually look at the cross-correlation the
evaluator builds. Take `n = 10`. `np.correlate(v, 1 − v, mode='full')` of two all-`1/2` vectors should be
the auto-correlation of a length-10 box scaled by `1/4`, which is a triangle. Working it out, the lag
profile is

```
[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25]
```

— nineteen entries, ramping up by `0.25` per step to a single peak of `2.5` at the center index (zero
shift) and back down. The peak is `2.5 = 10/4 = n/4`, exactly as the hand argument said, and `2.5 · 2/10
= 0.5`. So the evaluator returns `0.5` for `n = 10`, and the maximizing shift is `k = 0`, the dead-center
self-alignment. That is the concrete object behind the phrase "worst overlap": a symmetric triangle of
lag-overlaps peaking where the profile sits exactly on top of itself.

Now the discretization question. I claimed refining the flat profile changes nothing; I should confirm
that rather than assume it. Running the evaluator on `v = full(n, 1/2)` for `n = 2, 3, 7, 50, 600, 5000`
gives `0.5` in every case — the triangle just gets taller (`n/4`) and wider in lockstep, and the `2/n`
rescale cancels the height exactly. So the piece count `n` is genuinely not a lever for a flat profile:
every refinement of `1/2`-cells is still all `1/2`-cells, and the bound is pinned at `0.5` regardless of
resolution. Only the *shape* of the heights can move it.

I should also check the balance constraint while I am here, because it is the one thing that makes this
problem non-trivial. The rule is `Σ v_i = n/2` — exactly half the total cell-mass on the `A` side. The
flat profile has every `v_i = 1/2`, so its sum is `n·(1/2) = n/2`, and the evaluator's feasibility assert
(`|v.sum() − n/2| < 1e-9`) passes for all the `n` above. The flat profile is feasible, sitting right at
the center of the feasible region rather than at some degenerate corner.

So the floor is `0.5`, and it is a hard, structural floor for *this* profile. What I actually want to know
is the direction out of it. The constraint forces the *average* of `v` to be `1/2`, so I cannot lower
every product by globally shrinking the heights — the mass has to go somewhere. The escape has to be
redistribution: a cell at `1` contributes nothing to the overlap when aligned with another cell at `1`
(because `1 − 1 = 0`), and a cell at `0` contributes nothing as the `v_i` factor. So my instinct is that a
spiky, near-binary profile — cells near `0` or `1` rather than all at `1/2` — makes many products vanish
and should beat `0.5`.

Let me test that instinct before I lean on it, because "introduce variation" is vague and I want to see
whether the obvious way to do it actually helps. The crudest near-binary feasible profile at `n = 10` is
five `1`s then five `0`s — it sums to `5 = n/2`, perfectly balanced, and it is maximally spiky. I expected
it to drop below `0.5`. It does the opposite. Its lag profile is

```
[1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, ...]
```

peaking at `5` at shift `k = 4`, so `C = 5 · 2/10 = 1.0` — the worst score the functional can return. The
reason is exactly the mechanism I thought I was exploiting, running backwards: at shift `4`, each of the
five `A`-cells holding a `1` lands on a `B`-cell where `1 − v = 1` as well, so every product is `1·1 = 1`
and they all pile up at the *same* shift. Strict alternation `1,0,1,0,…` does the same thing and also
returns `1.0`. So breaking the symmetry is not automatically good — done naively it concentrates the
overlap into one catastrophic shift and is *twice* as bad as flat. The flat profile spreads its overlap
evenly across all shifts and pays `0.5`; a careless spike dumps it all onto one shift and pays `1.0`.

That sharpens what the real objective is. It is not "make the heights extreme" — it is "arrange the
heights so that *no single shift* accumulates a large total," which is a much more delicate, almost
adversarial, balancing act against the worst `k`. To see that the gain is genuinely available once the
arrangement is done with care, I let a constrained random search over continuous heights at a larger piece
count look for feasible profiles (renormalized to `Σ v = n/2`). Even an unsophisticated search of this
kind finds profiles around `C ≈ 0.42` at `n = 24` — comfortably below the flat `0.5` and below `0.43` —
while the same evaluator still returns exactly `0.5` on the flat vector at that `n`. So the descent below
`0.5` is real, but it lives in the *shape*, has to be searched for, and the naive moves point the wrong
way.

Let me pin the ceiling honestly so I know how far this floor sits from the bottom. On the *lower* side, the
true constant `C5` is at least `0.379005` (White, by convex programming), so no construction can certify a
bound below that. On the *upper* side, the published record is around `0.380869`, reached by elaborate
optimized step functions with several hundred pieces. So the constant lives in `0.379005 ≤ C5 ≤
0.380869`, a sliver of width about `2×10^{-3}`. My flat floor at `0.5` is `0.5 − 0.380869 ≈ 0.119` above
the top of that window — an enormous distance in the units of this problem — and essentially all of it has
to be bought by optimizing the heights into a non-trivial, asymmetric, near-binary profile that survives
the worst shift. The flat function buys none of it; it just establishes the starting altitude.

So this rung's contribution is the floor itself, measured and not merely asserted: `C = 0.5`, matching
Erdős's own 1955 upper bound `C5 ≤ 1/2`, reproduced exactly by the evaluator at every piece count, with
the maximizing shift at `k = 0` where the profile sits on its own complement. The flat vector is the
unique maximally-symmetric feasible point, with no internal degree of freedom to spend — which is why it is
the right baseline and also why it cannot improve. The two things I now trust going into the harder rungs
are concrete: the harness's "flat = 0.5" agrees with the hand computation, so I can rely on its numbers;
and the worst-shift mechanism is the real enemy, so any search I run next has to be judged by what it does
to the *peak* of that lag triangle, not by how spiky the profile looks. The cleanest place to start that
search is at a small piece count, where the height vector is short enough that a constrained optimizer can
explore the shape space thoroughly while holding the sum at `n/2`.
