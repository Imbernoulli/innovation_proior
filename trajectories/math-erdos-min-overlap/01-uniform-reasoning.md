I want a step function whose worst overlap against its own complement is as small as possible — that is
the whole game, and lower is better. But before I try to be clever, I should fix the floor: the simplest
legal profile, because I need to know what a value of the overlap even feels like on this problem, I need
a baseline that every later rung has to beat, and — most of all — I need to understand the functional I
am about to spend the rest of the ladder minimizing before I hand it to any optimizer. The simplest
function in the class is the flat one: every cell half-full, `v_i = 1/2` for all `i`. It is the only
profile with no internal structure at all, and its overlap I can reason about entirely by hand.

So let me do exactly that, slowly, because the point of this rung is comprehension, not a number. The
complement of the flat profile is itself: if `v_i = 1/2` everywhere, then `1 − v_i = 1/2` everywhere too.
The overlap at a shift `k` is `c_k = Σ_i v_i · (1 − v_{i−k})`, a sum over the cells that actually overlap
at that shift of the product `(1/2)·(1/2) = 1/4`. How many cells overlap at shift `k`? When I slide the
length-`n` height vector against its length-`n` complement by `k` cells, exactly `n − |k|` cells sit on
top of one another. So `c_k = (n − |k|)/4` — a perfect triangle in `k`: it peaks at `k = 0`, where all `n`
cells line up, at `c_0 = n/4`, and falls off linearly to zero at `k = ±n`. The worst shift is therefore
the zero shift, the raw worst overlap is `n/4`, and the evaluator's `2/n` rescale gives `(n/4)·(2/n) =
1/2`. The flat profile scores exactly `1/2`.

I want to check the closed form on the smallest cases by hand, because a clean formula is easy to get
subtly wrong at the boundaries. Take `n = 2`, `v = (1/2, 1/2)`: the full cross-correlation of `(1/2,1/2)`
with `(1/2,1/2)` has three entries `(1/4, 1/2, 1/4)`, peaking at `1/2`, and `1/2·(2/2) = 1/2`. Take `n =
4`: the full correlation runs `(1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4)`, a clean triangle peaking at `1`, and
`1·(2/4) = 1/2`. Both match the `(n − |k|)/4` prediction exactly — peak `n/4` (which is `1` at `n = 4`),
linear falloff, rescale to `1/2`. And it keeps saying `1/2` at `n = 10`, `100`, `600`: the same triangular
envelope every time, just sampled at more points along the same triangle. That invariance is not an
accident of arithmetic; it is telling me something structural, and I want to nail down what.

Here is the reason, and it is worth stating precisely because I will lean on it at every later resolution.
The domain is `[0, 2]` cut into `n` equal cells, so each cell has width `2/n`. The continuous overlap
Haugland's equivalence is about is `∫ h(x)(1 − h(x+k)) dx`; the discrete `c_k = Σ_i v_i(1 − v_{i−k})` is
the Riemann sum of that integrand *without* the width factor, so multiplying by the cell width `2/n` turns
the sum into the integral. The `2/n` is not an arbitrary normalization — it is exactly the cell width that
makes the discrete score converge to the continuous functional. And that is precisely why the flat score
is `n`-invariant: refining a flat profile subdivides each `1/2`-cell into more `1/2`-cells, the underlying
step function `h ≡ 1/2` on `[0,2]` is *literally unchanged*, so its integral overlap is unchanged, and the
discrete score reproduces the same `1/2` at every resolution. The piece count `n` is a red herring on its
own; only the *shape* of `h` moves the bound. That is the first real lesson, and it is a units check I can
trust: the number is resolution-independent because `2/n` is the width and `h` is the same function.

Before I leave the flat point I want to understand *why* it is the special one, because "most symmetric"
is vague and I would rather have the exact symmetries in hand. The score `C` has two symmetries I can read
straight off the definition. First, complementation: replacing `v` by `1 − v` sends `c_k = Σ v_i(1 −
v_{i−k})` to `Σ (1 − v_i) v_{i−k} = c_{−k}`, and since the worst overlap maximizes over all shifts and the
shift range is symmetric, `max_k c_k` is unchanged — so `C(1 − v) = C(v)`. Second, reflection: reversing
the vector, `v_i → v_{n−1−i}`, also maps `c_k → c_{−k}` and leaves `C` invariant. The flat profile is the
*unique fixed point of complementation* — `1 − v = v` forces `v ≡ 1/2` — and it is fixed by reflection
too. That is the precise sense in which it is maximally symmetric: it is the one profile that is its own
complement, which is exactly why its complement equals itself and its worst overlap lands at zero shift by
perfect self-alignment. It also tells me something about the destination: any profile that beats `1/2`
must *leave* the self-complementary point, and since `C(v) = C(1 − v)`, the good profiles come in
complement-pairs and can be taken asymmetric without loss. Symmetry is the enemy here; the flat profile is
pure symmetry, and every bit of improvement is a symmetry broken.

Now I want something sharper than "the flat profile is `1/2`." I want to know how much of the interval
below `1/2` is even reachable — what the *hardest possible* worst overlap is, as a floor no construction
can cross — because that tells me whether the descent from `1/2` is a long road or a short one, and it
gives me a yardstick for reading every later number. There is a conservation fact I can get for free. Sum
`c_k` over every shift `k`, from `−(n−1)` to `n−1`. Each `c_k` is a sum of products `v_i (1 − v_j)` with
`j = i − k`; as `k` ranges over all shifts and `i` over all cells, the ordered pair `(i, j)` runs over
*every* pair of cells exactly once — the pair `(i, j)` contributes to the single shift `k = i − j`. So

```
Σ_k c_k = Σ_{i,j} v_i (1 − v_j) = (Σ_i v_i)·(Σ_j (1 − v_j)).
```

The first factor is the balance constraint itself, `Σ v = n/2`. The second is `Σ(1 − v_j) = n − n/2 =
n/2`. So the total overlap summed over all shifts is `(n/2)(n/2) = n²/4`, *for every feasible profile
whatsoever* — it does not depend on the shape at all. I checked this two ways to be sure I have not fooled
myself at the boundaries: a random feasible vector at `n = 24` gives a shift-sum of exactly `144 = 24²/4`,
and the flat triangle `Σ_k (n−|k|)/4` also collapses to `n²/4` when I add up the two arms. The total is
conserved. This is a genuine handle, and a surprising one — I can move the mass around between shifts, but
I can never remove any of it.

That immediately gives a floor. The worst shift is a maximum over `2n − 1` shifts of numbers that are
forced to sum to `n²/4`, and a maximum is always at least the average, so `max_k c_k ≥ (n²/4)/(2n − 1)`.
After the `2/n` rescale, `C ≥ (n²/4)·(2/n)/(2n−1) = (n/2)/(2n−1) = n/(2(2n−1))`, which tends to `1/4` as
`n → ∞`. Numerically the floor is `0.2553` at `n = 24`, `0.2510` at `n = 120`, `0.2502` at `n = 600`. So
*any* step function, no matter how cleverly shaped, has worst overlap at least about `1/4`. That reframes
the whole problem. The flat profile sits at `1/2`, which is exactly twice this floor — its peak-to-mean
ratio is `(n/4)/((n²/4)/(2n−1)) = (2n−1)/n ≈ 2`. The flat triangle is as *peaked* as the conservation
budget lets a symmetric profile be: it piles the entire `n²/4` of overlap into one tall triangle centered
at zero shift. The unreachable ideal would be a profile whose overlap is *perfectly flat across all
shifts*, every `c_k` equal to the average, scoring the floor near `1/4`. That is the shape I am really
chasing — not "small overlap," which is impossible because the total is pinned at `n²/4`, but *evenly
spread* overlap, so that no single shift towers over the rest.

And the crude `1/4` floor is loose: the true constant lives around `0.38`, a peak-to-mean of roughly
`0.38/0.25 ≈ 1.5`. So the best constructions in the world get the envelope down from twice its mean to
about one-and-a-half times it, but never to the flat ideal of `1.0`. There is an irreducible peakedness
that the conservation law permits but the actual geometry of these overlaps enforces, and squeezing that
last factor is the seventy-year-old open problem. Reading it this way also tells me that White's provable
lower bound `0.379005` is doing real work: it is far above my `≈ 0.25` averaging floor, so convex
programming has already excluded almost a quarter of overlap-space that the naive conservation argument
leaves open.

There is one more feature of the flat envelope I want to name, because it is the structural opposite of
what a good profile looks like and it foreshadows the difficulty of the search. At the flat point the
worst shift is *isolated*: `c_0 = n/4`, and the next-highest shifts `c_{±1} = (n−1)/4` sit a full `1/4`
below it in raw units, i.e. `(2/n)·(1/4) = 1/(2n)` below in the rescaled score. At `n = 24` that is a gap
of about `0.021` between the worst shift and its nearest competitor; the peak stands alone on top of the
triangle. A profile that scores near the `1/4` floor is the reverse — its overlap envelope is nearly flat,
so *many* shifts sit essentially tied at the maximum. Descending from `1/2` toward the floor therefore
means converting a single isolated peak into a broad plateau of tied shifts, all held down at a common
level: I am not shaving one number, I am flattening a whole envelope and then pressing the entire plateau
down against the fixed total `n²/4`. That is why the objective becomes a genuine minimax with a large set
of near-worst shifts as I improve, and why the early, easy descent (one dominant peak to lower) is nothing
like the late grind (hundreds of tied shifts to lower together).

Where does the redistribution come from mechanically? Look at the zero-shift term by itself, `c_0 = Σ_i
v_i (1 − v_i) = Σ v_i − Σ v_i² = n/2 − Σ v_i²`. For the flat profile `Σ v_i² = n·(1/4) = n/4`, so `c_0 =
n/2 − n/4 = n/4` — the peak. To pull `c_0` down I want `Σ v_i²` *large*, and subject to `Σ v_i = n/2` the
sum of squares is maximized by pushing every height to the extremes `0` or `1`: a binary profile with
exactly `n/2` ones has `Σ v_i² = n/2`, so `c_0 = n/2 − n/2 = 0`. A binary balanced profile kills the
self-overlap at zero shift *entirely*. That is the mechanism the flat profile cannot touch — every cell at
`1/2` is the *worst* case for `Σ v_i²`, the exact point where the zero-shift self-overlap is maximal. The
descent from `1/2` must run toward the corners of the box, toward a near-binary profile, so that at zero
shift a heavy cell (`v` near `1`) meets a light complement (`1 − v` near `0`) and the product vanishes.

There is a catch, though, and it is the catch that tells me the problem is not trivial: killing `c_0` does
*not* kill the worst overlap, because the conservation law says the mass I removed from zero shift has to
reappear at other shifts. Emptying the peak just pushes overlap sideways; some other shift becomes the new
worst. So a binary profile is necessary but wildly insufficient — the balance constraint plus binarity
fixes the total at `n²/4` and empties the zero shift, but says nothing about whether the redistributed
mass lands *evenly* (good, near the `1/4` floor) or piles right back up at a single new worst shift (bad).
Arranging the `0`s and `1`s so the leftover overlap spreads as flatly as possible across all nonzero
shifts is the entire content of the optimization, and it is why a hand formula gives way to search. I can
already feel, without running anything, that a *naive* binary profile could be catastrophic: a long block
of `1`s followed by a block of `0`s would, at the shift that lines the block of `1`s up against the
complement's block of `1`s, pile the whole half-mass into one shift — far worse than the flat `1/2`. The
flat profile at least *spreads*; a badly arranged binary concentrates. So binarity is a direction, not a
destination, and the destination is an arrangement I cannot write down in closed form.

The flat profile has none of this freedom to spend, and I can see exactly how rigid it is by perturbing
it. Take a zero-sum perturbation `v_i = 1/2 + δ_i` with `Σ δ_i = 0`, so the balance constraint stays
satisfied. Then `c_0 = n/2 − Σ(1/2 + δ_i)² = n/2 − Σ(1/4 + δ_i + δ_i²) = n/4 − Σ δ_i²`: the zero-shift
overlap drops *quadratically*, by `‖δ‖²`, for *any* direction I move. So the flat point is the unique
maximizer of the smooth zero-shift term — moving off it in any feasible direction lowers `c_0`. In the
rescaled score that is a drop of `C ≈ 1/2 − (2/n)‖δ‖²`, i.e. `C ≈ 1/2 − 2·(mean square perturbation)`. I
tried this on paper and then checked it: a zero-sum jitter of per-cell standard deviation `0.1` at `n =
20` gives `Σ δ_i² = 0.2`, predicting `c_0 = 5 − 0.2 = 4.8` and `C = 0.48`, and the evaluator returns
exactly `0.48000` with the worst shift still at `k = 0`. So near the flat point the descent is real but
slow and purely quadratic, and it is governed entirely by the collapsing central peak while the other
shifts barely stir — to first order, the off-zero `c_k` move only through the boundary of the perturbation,
their bulk being second order.

But that easy descent has a short range, and watching where it ends is instructive. If I push the same
random jitter harder — standard deviation `0.2` at `n = 20` — the worst shift *jumps off zero*: `c_0` has
fallen to `4.2`, but a neighboring shift `k = −1` has risen past it, and the rescaled score bounces back up
to `0.519`, above the flat floor. A random direction stops being a descent direction the moment a
different shift takes over as the maximum. That is the onset of the minimax structure I will be fighting
for the rest of the ladder: the objective is a max over many shifts, smooth within each shift's region but
kinked where two shifts tie for worst, and a naive move that lowers today's worst shift generically raises
tomorrow's. Lowering `c_0` cheaply gets me a little way off `1/2`; going further requires lowering the
*worst* shift while watching all the others that are climbing to meet it — a genuine constrained minimax,
not a jitter.

Putting the two computations together gives a clean geometric statement of what the baseline actually is.
Within the small ball where zero shift stays the worst, the objective *equals* `c_0 = n/4 − ‖δ‖²` rescaled,
which is a smooth downward bump — so the flat point is a strict *local maximum* of the very functional I am
trying to minimize. That is a stronger and more useful statement than "it has no internal structure": it
means every feasible direction out of the flat point descends, so the first optimization step is
guaranteed to help no matter where it heads, and the only questions are how *far* the local descent runs
before a shift-swap turns the max over to a different `c_k`, and how deep the basin on the other side of
that kink goes. The flat baseline is thus the worst feasible interior point in its own neighborhood — the
top of the hill — which is precisely why it is the right floor to start from: there is nowhere to go but
down, and the descent is guaranteed to exist even though it is shallow and short-ranged near the peak.

The balance constraint deserves one last look, because it is the one thing that makes this the Erdős
problem rather than a triviality. Without `Σ v = n/2` the minimizer is obvious and useless: set every
height to `0`, then every product `v_i(1 − v_{i−k})` has a zero factor, the total overlap is `0`, and the
"bound" is a meaningless `0` certifying nothing about `C5`. The constraint — exactly half the cell-mass on
the `A` side — is what forbids that escape; it is the discrete image of `∫ f = 1` with `f` the density of
`A`. The flat profile sits *exactly* on it (`n·(1/2) = n/2`), dead center of the feasible polytope
`[0,1]^n ∩ {Σ v = n/2}`, so the floor I am measuring is a genuine interior point, not a degenerate corner.
That is reassuring: `1/2` is the honest value of the most symmetric feasible profile, not an artifact of
some boundary.

Now I can pin the ceiling honestly and know how far the descent runs. On the lower side `C5 ≥ 0.379005`
(White, by convex programming), a floor no construction can cross and far tighter than my `≈ 0.25`
conservation floor. On the upper side the published record is around `0.380869`, reached by elaborate
optimized step functions with several hundred pieces. So the constant lives in the sliver `0.379005 ≤ C5 ≤
0.380869`, width about `2×10⁻³`. My flat floor at `0.5` is about `0.12` above the *top* of that window — an
enormous distance in the units of this problem — and essentially all of it must be bought by breaking the
flat symmetry: pushing heights to the corners of the box into an asymmetric, near-binary profile, and
arranging those corners so the redistributed overlap spreads flat instead of re-piling at a new worst
shift.

The conservation picture also lets me predict the *shape* of the descent, which I can check against the
ladder as it unfolds. The reachable range of the peak-to-mean ratio runs from the flat `2.0` down to the
best-known `≈ 1.5`, with the geometric ideal `1.0` forbidden. Breaking the symmetry at all — moving from a
concentrated peak to a roughly flattened envelope — should capture most of that range in one step, because
the first thing an optimizer does is empty the towering zero-shift peak and spread the mass, and emptying
`c_0` is worth a lot (`c_0` falls from `n/4` toward `0`). But conservation guarantees sharply diminishing
returns after that: once the envelope is roughly flat, every further gain requires lowering *all* the tied
shifts simultaneously against a total that cannot shrink, so the marginal improvement per unit of effort
collapses. So I expect the ladder to show one big drop off `0.5` — into the `0.38`s almost immediately —
followed by a long grind of ever-smaller gains as the profile is refined, never reaching the `0.25` floor
and asymptoting somewhere near `0.38`. That is a falsifiable qualitative prediction, and the sequence of
bounds the later rungs report will either bear it out or force me to revise the redistribution picture.

The flat function buys none of that; it just establishes the starting altitude and, more valuably, the
map. I now understand the objective as a redistribution game under a fixed total `n²/4`, with a hard
conservation floor near `1/4`, a symmetric-and-maximal starting point at `1/2` sitting at twice that
floor, and a target band near `0.38` where the peak sits about `1.5×` the mean. What I expect from the
evaluator on this rung is `C = 0.5` exactly — confirming both the triangle-envelope computation and the
discretization invariance, and confirming that the harness's notion of "flat = floor" matches my hand
calculation, so I can trust it on the harder rungs where I cannot check by hand.

The limitation this rung exposes is sharp and sets up the next one. The flat profile is a rigid nothing:
worst overlap locked at `1/2` by perfect self-alignment at zero shift, no direction to move because every
cell is identical, zero internal degrees of freedom. To descend at all I have to introduce *variation*
among the heights and let a search procedure discover which non-flat, near-binary arrangement flattens the
overlap envelope — pushing heights toward `0` and `1` while holding the sum at `n/2`, and, harder,
arranging them so the leftover overlap does not simply re-concentrate at a new worst shift. The cleanest
place to begin that search is at a small piece count, where the vector is short enough that an optimizer
can explore the shape space thoroughly and find the first profile that already clears the low `0.4`s and
drops into the `0.38`s — a first real step off the `0.5` floor, and the first test of whether the
redistribution picture I just built actually predicts where the descent goes. And I keep the flat baseline's
own signature as a control: its bound is invariant to the piece count, so if later rungs lower the bound as
the pieces grow, that drop is unambiguously the work of *shape*, not of resolution granting free progress —
the flat floor has already proved that more pieces, by themselves, buy nothing at all.
