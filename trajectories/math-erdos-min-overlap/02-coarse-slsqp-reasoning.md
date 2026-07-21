The flat floor gave me a map, not just a number: the objective is a redistribution game under a fixed
total `n²/4`, with a conservation floor near `0.25`, a symmetric start at `0.5`, and a target band near
`0.38`. The only lever is the shape of the heights, and the only way down is to break the
self-complementary symmetry — push cells toward `0` and `1` so `c_0 = n/2 − Σ v_i²` collapses — while
keeping the conserved mass from re-piling at a new worst shift. So now I have to optimize. I keep the piece
count small — two dozen cells — because I want to understand what an optimized profile looks like before
committing to a long vector: at `n = 24` the minimax has `47` shifts, restarts are cheap, and I can read
the shape off the result. The `~600`-piece records are the destination, but I should start where the
search is fast and legible.

Before reaching for an optimizer I test the cheapest idea the flat analysis hands me: a balanced binary
profile drives `c_0` to zero, so maybe I can hand-write one. The
alternating `1,0,1,0,…` scores `C = 1.0`; twelve `1`s then twelve `0`s scores `1.0`; period-six
`1,1,1,0,0,0` scores `1.0`. Every clean hand pattern is catastrophic — twice the flat floor. The mechanism
is conservation biting back: binarizing empties `c_0`, but the `n²/4` that lived there reappears, and a
regular pattern lines its block of `1`s against the complement's block of `1`s at one shift, piling the
whole half-mass there. Concretely, `v = (1^{12}, 0^{12})` has its ones at `0…11` and the complement's ones
at `12…23`, so every `v`-one meets a complement-one at the single shift `k = −12`: `c_{−12} = 12`, `C =
12·(2/24) = 1.0`, the entire half-mass in one spike of height `12` against the flat triangle's `6`. So I
try randomness instead of structure — `2000` random balanced binaries — and the best is `0.5`, the mean
`0.628`, none beating the flat floor, the worst near `0.92`. Binarity alone, structured or random, is not
a descent but usually a disaster, and the landscape is wildly variable, neighboring arrangements differing
by half in the bound.

That wall teaches two things the flat analysis did not. The arrangement of the corners is everything, and
the good arrangements are a vanishing fraction a search must find deliberately — blind sampling of a
jagged terrain loses to a gradient method that can descend from wherever it starts. And pure binary is too
rigid: with only `0`s and `1`s I cannot finely detune the shifts competing to be worst. So the sub-`0.5`
profiles must be *near*-binary — mostly at the corners to keep `c_0` small, with a minority of interior
cells that fine-tune and shave tied shifts apart. That rules out a combinatorial search over binary
strings and points at a continuous constrained optimizer over the box `[0,1]`.

What am I minimizing, and why is it hard? The score is `max_k c_k` over `47` shifts, rescaled — a minimax,
the pointwise max of `47` smooth functions, piecewise-smooth with kinks where two shifts tie. A plain
gradient chatters at those kinks, and a solver that sees only the hard `max` gets no gradient about the
near-worst shifts about to become binding, so it lowers today's worst straight into tomorrow's — the same
shift-swap I watched at the flat point. The fix is to replace the hard `max` with a smooth log-sum-exp
surrogate at sharpness `β`,

```
C̃_β(v) = ( m + (1/β) log Σ_k exp(β (c_k − m)) )·2/n,     m = max_k c_k,
```

which feels all near-worst shifts at once and converges to the true max as `β → ∞`. But I have to watch
how faithfully it tracks the number I report: log-sum-exp overshoots the true max by at most
`log(#shifts)/β`, so the rescaled surrogate sits below the true worst overlap by at most `(2/n)·log(2n−1)/β`.
At `n = 24`, even at the sharpest `β = 1200` I plan to use, that gap is about `2.7×10⁻⁴` — the same order
as the entire distance I expect between a coarse `~0.3812` and the record `0.38087`. So I cannot optimize
the surrogate and quote it; two consequences follow. I anneal `β` up a ladder `(60, 150, 300, 600, 1200)`,
warm-starting each solve from the last, so the landscape deforms gradually from a soft nearly-convex bowl
to the kinked true minimax — soft early to reorganize freely, sharp late to hug the number I care about.
And I always score the true hard-max of whatever comes back, never the surrogate.

The constraints are what make this Erdős's problem: heights in `[0,1]`, summing to exactly `n/2`. The sum
equality is load-bearing — the flat analysis showed that without it the optimizer drives every height to
`0` for a meaningless bound of `0`. So I need a solver handling box-plus-linear-equality natively, and
SLSQP is exactly that, and the tool the agentic-search record (AutoEvolver) reports on this problem. Its
QP subproblem is roughly cubic in the variables, nothing at `24` cells. After each solve I re-project onto
the constraint set — clip to `[0,1]`, then spread the residual `n/2 − Σ v` across the strictly interior
cells and iterate — because pushing mass into a cell pinned at `1` would just get clipped back off; that
converges in a few passes since each either closes the residual or saturates another cell. (I note in
passing that scipy finite-differences the surrogate at `O(n)` evaluations per gradient, each an `O(n²)`
correlation — trivial here, but exactly the cost that will make SLSQP the bottleneck at record
resolution.)

The last issue is local minima, and the binary catastrophe already showed how vicious the landscape is —
profiles at `1.0` next to profiles at `0.38`. One SLSQP ladder finds one basin with no reason to trust it,
so I multi-start: `12` random feasible starts under a fixed seed, keep the best true overlap. A dozen
restarts at two dozen cells is a trivial budget and a reliable read on what the coarse resolution reaches;
if the result looks basin-limited rather than resolution-limited, the cheap fix is more starts, not a
longer vector. That arithmetic turns against me as `n` grows — more cells, more and narrower basins, a
dozen blind starts no longer enough — which is another reason this legible pass belongs at small `n`.

I expect this to be the big first drop the flat analysis predicted. A genuinely optimized near-binary
profile should be worth most of the descent — the earliest optimized step functions already crossed `0.5 →
0.4 → 0.38` — so I expect the low `0.38`s, a touch above Haugland's `0.380927`, but not the frontier: two
dozen wide cells cannot resolve the fine structure a near-optimal profile needs. Landing near `0.3812`
would close about `98%` of the whole `0.5 → 0.379005` window in one step, leaving the entire remaining
enterprise in the last `~2%`, where every gain is a ten-thousandth — the same size as the `2.7×10⁻⁴`
surrogate gap.

The cap here is resolution, and I can say why in interior-knob terms. To keep `c_0` small most cells pin to
the corners; the sum constraint is then carried by the minority of interior cells, which are the only knobs
for detuning tied shifts. With perhaps a quarter to a third of `24` cells pinned, I have on the order of a
dozen interior knobs to flatten a `47`-shift envelope — a coarse instrument where each knob moves several
shifts at once. More cells mean more knobs and finer detuning, which is the lift the next step buys. That
gives a clean test: if the cap is genuinely resolution, adding pieces should lower the bound while extra
coarse restarts should not; if I merely missed a basin, more starts at `n = 24` would find something lower.
The two make opposite predictions, and I expect the former.
