I want a step function whose worst overlap against its own complement is as small as possible — lower is
better, that is the whole game. Before trying to be clever I should fix the floor: the simplest legal
profile, so I know what an overlap value feels like on this problem and have a baseline every later step
must beat. More than that, I want to understand the functional I am about to spend the ladder minimizing.
The simplest function in the class is the flat one, every cell half-full, `v_i = 1/2`. It is the only
profile with no internal structure, and its overlap I can reason about entirely by hand.

The complement of the flat profile is itself: `1 − v_i = 1/2` everywhere. The overlap at shift `k` is `c_k
= Σ_i v_i (1 − v_{i−k})`, a sum over the overlapping cells of `(1/2)(1/2) = 1/4`. Sliding a length-`n`
vector against its complement by `k`, exactly `n − |k|` cells sit on top of one another, so `c_k = (n −
|k|)/4` — a triangle peaking at `k = 0` with `c_0 = n/4` and falling linearly to zero at `k = ±n`. The
worst shift is zero, the raw worst overlap `n/4`, and the evaluator's `2/n` rescale gives `1/2`. That is
exactly Erdős's own 1955 bound `C5 ≤ 1/2`, and it holds at `n = 10, 100, 600` alike.

That invariance is structural, and worth naming because I will lean on it at every later resolution. The
domain `[0,2]` is cut into `n` cells of width `2/n`; the continuous overlap is `∫ h(x)(1 − h(x+k)) dx`,
and the discrete `c_k` is its Riemann sum without the width factor, so `2/n` is exactly the cell width
that turns the sum into the integral. Refining a flat profile subdivides each `1/2`-cell into more
`1/2`-cells — the underlying step function `h ≡ 1/2` is literally unchanged — so the score reproduces
`1/2` at every resolution. The piece count is a red herring on its own; only the *shape* of `h` moves the
bound.

Two symmetries fall straight out of the definition. Complementation `v → 1 − v` sends `c_k → c_{−k}`, and
since the worst overlap maximizes over a symmetric shift range, `C(1 − v) = C(v)`. Reflection `v_i →
v_{n−1−i}` also maps `c_k → c_{−k}` and leaves `C` fixed. The flat profile is the unique fixed point of
complementation — `1 − v = v` forces `v ≡ 1/2` — which is exactly why its complement equals itself and its
worst overlap lands at zero shift by perfect self-alignment. Any profile that beats `1/2` must leave that
self-complementary point, and since `C(v) = C(1 − v)` the good profiles come in complement-pairs and can
be taken asymmetric without loss. Symmetry is the enemy; every bit of improvement is a symmetry broken.

Now something sharper than "flat scores `1/2`." There is a conservation fact I get for free. Sum `c_k`
over every shift: each `c_k` sums products `v_i (1 − v_j)` with `j = i − k`, and as `k` and `i` range, the
ordered pair `(i, j)` runs over every pair of cells exactly once. So

```
Σ_k c_k = Σ_{i,j} v_i (1 − v_j) = (Σ_i v_i)·(Σ_j (1 − v_j)) = (n/2)(n/2) = n²/4,
```

for *every* feasible profile whatsoever — it does not depend on the shape at all. I can move overlap
between shifts, but I can never remove any of it.

That gives a floor. The worst shift is a max over `2n − 1` shifts summing to `n²/4`, and a max beats the
average, so `max_k c_k ≥ (n²/4)/(2n − 1)` and after rescale `C ≥ n/(2(2n − 1)) → 1/4` (`0.2553` at `n =
24`, `0.2502` at `n = 600`). So any step function has worst overlap at least about `1/4`, and the flat
profile at `1/2` sits at exactly twice this floor — peak-to-mean `(2n − 1)/n ≈ 2`, the whole `n²/4` piled
into one triangle at zero shift. What I am chasing is not *small* overlap — impossible, the total is
pinned — but *evenly spread* overlap, every `c_k` near the average so no single shift towers. The constant
near `0.38` corresponds to peak-to-mean `≈ 1.5`; the best constructions press the envelope from twice its
mean down to about one-and-a-half times, never to the flat ideal of `1.0`. That irreducible peakedness is
the open problem, and it is why White's provable `0.379005` sits far above my `≈ 0.25` averaging floor.

Where does the redistribution come from mechanically? The zero-shift term is `c_0 = Σ v_i (1 − v_i) = n/2
− Σ v_i²`. For the flat profile `Σ v_i² = n/4`, so `c_0 = n/4` — the peak. To pull `c_0` down I want `Σ
v_i²` large, and subject to `Σ v = n/2` that is maximized by pushing every height to `0` or `1`: a
balanced binary profile has `Σ v_i² = n/2`, so `c_0 = 0`, killing the self-overlap entirely. Every cell at
`1/2` is the *worst* case for `Σ v_i²`. So the descent from `1/2` must run toward the box corners, toward
a near-binary profile.

But killing `c_0` does not kill the worst overlap — conservation says the mass removed from zero shift
reappears at other shifts. A naive binary is not just insufficient, it can be catastrophic: a block of
`1`s followed by a block of `0`s lines its ones up against the complement's ones at one shift and piles
the whole half-mass there, far worse than `1/2`. Binarity is a direction, not a destination; arranging the
corners so the leftover overlap spreads flat rather than re-piling is the entire content of the
optimization, and it is why a hand formula gives way to search.

The flat profile has none of that freedom, and I can see how rigid it is by perturbing. Take `v_i = 1/2 +
δ_i` with `Σ δ_i = 0`: then `c_0 = n/4 − Σ δ_i²`, dropping quadratically in any feasible direction. So the
flat point is a strict local *maximum* of the very functional I am minimizing — every feasible direction
descends, the first optimization step is guaranteed to help. But the easy descent has short range: push a
random jitter harder and the worst shift jumps off zero — `c_0` falls but a neighboring `c_{−1}` climbs
past it and the score bounces back above `1/2`. That is the onset of the minimax I will fight for the rest
of the ladder: the objective is a max over many shifts, smooth within each shift's region, kinked where
two tie, and a move that lowers today's worst shift generically raises tomorrow's.

One last look at the balance constraint, because it is what makes this Erdős's problem rather than a
triviality. Without `Σ v = n/2` the minimizer is trivial and useless — set every height to `0`, every
product has a zero factor, the bound is a meaningless `0`. The constraint is the discrete image of the
`A`/`B` balance, and the flat profile sits exactly on it, dead center of the feasible polytope, so `1/2`
is a genuine interior value, not a boundary artifact.

The whole descent from `0.5` to White's `0.379005` is only `~0.12`, essentially all of it bought by
breaking the flat symmetry into an asymmetric near-binary profile whose overlap spreads flat. Conservation
even predicts the *shape* of that descent: emptying the towering `c_0` peak is cheap and happens first, so
I expect one big drop off `0.5` into the `0.38`s, then a long grind of ever-smaller gains as the tied
envelope is pressed against the fixed total, never reaching `0.25`. To descend at all I have to introduce
variation and let a search find which near-binary arrangement flattens the envelope — cleanest to begin at
a small piece count, short enough to explore the shape space thoroughly. And the flat baseline leaves a
control: its bound is invariant to piece count, so any later drop as the pieces grow is unambiguously the
work of shape, not resolution granting free progress.
