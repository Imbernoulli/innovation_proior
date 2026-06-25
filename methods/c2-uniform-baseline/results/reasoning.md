I want a non-negative step function whose autoconvolution is as "spread out yet peaked" as possible, in
the precise sense the ratio measures: lots of `L_2` energy relative to the product of the peak height and
the total mass. Before I try to be clever, I should fix the floor — the simplest legal function — because
I need to know what a value of `R` even feels like on this problem, and I need a baseline that every later
rung has to beat. The simplest function in the class is a flat one: all heights equal, `v_n = 1` for every
piece. That is the discretized indicator of an interval, and its autoconvolution is the cleanest object I
can reason about by hand. Whatever number it produces, I want to derive it from the geometry myself, not
just read it off the evaluator — partly to understand the functional, partly so that when the harness
agrees with me I have a calibration point I can trust on the harder rungs.

Take `f` to be the indicator of an interval of length `1` (dilation invariance means the length does not
matter, so `1` is convenient). Its autoconvolution is the classic triangle — the "tent" — supported on
`[0, 2]`, rising linearly from `0` to a peak at the midpoint and falling back to `0`. I can get the peak
height directly: `f*f(1) = ∫ f(t)f(1−t) dt = ∫_0^1 1 dt = 1`, so the curve tops out at `1` and
`||f*f||_inf = 1`. The area under the triangle is base times height over two, `2·1/2 = 1`, so `||f*f||_1 =
1`. And `||f*f||_2^2` is the integral of the triangle squared. By symmetry that is twice the integral over
the left half, where the tent is just the line `y = x` for `x ∈ [0,1]`: `2·∫_0^1 x^2 dx = 2·(1/3) = 2/3`.
So the ratio comes out to `(2/3) / (1·1) = 2/3`. The flat function should score exactly `2/3`.

I do not want to take that on faith from one continuous calculation, so let me check it against the
discrete machinery the evaluator actually uses, on a tiny case I can hold in my head. For `f` a flat
vector of `N` ones, the node values are `L_j = (v*v)_{j−1}`, the discrete autocorrelation of `(1,…,1)`
with itself. For `N = 3` that convolution is `1,2,3,2,1`, so with the boundary zeros the node sequence is
`L = (0,1,2,3,2,1,0)` — exactly the integer triangle I expected, climbing to `N` at the centre and back
down. Feeding that into the exact piecewise-linear norms:
`||f*f||_inf = max L = 3`;
`||f*f||_1 = ½ Σ_j (L_j + L_{j+1}) = ½·(1+3+5+5+3+1) = 9`;
`||f*f||_2^2 = ⅓ Σ_j (L_j^2 + L_j L_{j+1} + L_{j+1}^2) = ⅓·54 = 18`;
so `R = 18 / (3·9) = 18/27 = 2/3`. The discrete formula lands on the same `2/3`, which is a good sign:
my continuous tent and the evaluator's piecewise-linear bookkeeping agree, and the peak height `3 = N`
cancelled out cleanly against the area. I run the actual code to be sure I have not fooled myself, and it
returns `0.6666666666666666` for the flat vector — matching to machine precision.

Now the more interesting question: does the number of pieces matter? My `N = 3` computation produced the
peak `L = N` exactly because every node value scaled with `N` the same way, and the ratio is invariant
under that overall scale. So I would expect a flat vector of any length to give the same `2/3`. Let me
not just expect it — let me sweep `N`. Running the flat vector at `N = 1, 2, 3, 5, 10, 50, 1000` returns
`0.6666666666666666` in every case (the last one drops by one unit in the last decimal place, pure
floating-point noise). So refining a flat profile buys nothing: a flat vector of ten ones and a flat
vector of a thousand ones both have a triangular autoconvolution and both score `2/3` exactly. The piece
count `N` is a red herring on its own; only the *shape* of the heights can move the ratio.

That tells me where the gains have to come from, but I want to confirm it rather than assert it, so let me
perturb the shape and watch `R` respond. A few concrete non-flat vectors: `[1,2,3,2,1]` (mass piled toward
the middle) gives `0.6996`; `[1,3,1]` (a sharp central spike) gives `0.6473`; the split block
`[1,1,1,0,1,1,1]` gives `0.5494`. So shape genuinely matters and it cuts both ways — some profiles beat
the flat floor and some fall well below it. The ones that win are the ones that make the autoconvolution
*less triangular*: a tent spends a lot of `L_1` mass and a lot of width on the thin tails near the base,
mass that contributes little to `L_2` but inflates the `L_1` denominator. Concentrating the heights toward
the centre fattens the cap of the autoconvolution and steepens its sides, pushing the curve toward the
indicator that Hölder says would give `1` — `[1,2,3,2,1]` already does a little of this and clears `0.69`.

Here I almost wrote down that the flat profile is a "local nothing" with no gradient to follow, because
every piece is identical and there is nothing to vary. But that is a claim about the local geometry of `R`,
and I just saw shape changes move the value in both directions, so I should actually check it instead of
assuming. I take the flat vector of `20` ones and apply two thousand tiny random perturbations of size
`~0.02` to the heights, counting how many raise `R` above `2/3` and how many lower it. The result is `1010`
up and `990` down — essentially a coin flip. So the flat point is **not** a local maximum and **not** a
flat plateau: it is a *saddle*. There is no shortage of gradient here; ascent directions are everywhere,
they just sit alongside equally many descent directions. I can even name a clean ascent direction directly:
nudge the two central heights up and the rest down, and `R` rises monotonically off the floor — a symmetric
middle-bump on a flat-10 vector gives `0.6696` at amplitude `0.05`, `0.6721` at `0.10`, `0.6781` at `0.50`,
while the mirror move (dipping the middle, piling mass at the ends) drives `R` down to `0.6105`. The
distinguished thing about the flat profile, then, is not that it is stuck — it is that it is the unique
maximally-symmetric member of the class, sitting exactly at the symmetric critical point where the
mass-concentration direction and its reverse are perfectly balanced. The moment a search breaks that
symmetry toward the centre, it climbs.

I should also pin down the ceiling honestly so I know how far the climb has to go. Hölder gives `R ≤ 1`,
but that is unattainable here — the autoconvolution of a non-negative function is continuous and
bump-shaped, never a sharp indicator, so `R = 1` is forever out of reach and the true supremum `C2` is
strictly below `1`. The published lower bounds say `C2` is at least around `0.96`, reached by elaborate
optimized step functions with tens of thousands of pieces. So the distance from this flat floor at
`0.6667` to the frontier near `0.96` is enormous, and essentially *all* of it has to be bought by
optimizing the heights into a non-trivial, asymmetric, structured profile. The flat function buys none of
it; it just establishes the starting altitude — and, now that I have checked it against both the
continuous tent and the discrete formula and confirmed its `N`-invariance, a trustworthy starting altitude
I can rely on as a sanity check when the harder rungs report their numbers.

What this rung settles, then, is the baseline: a flat non-negative height vector, which the analysis above
fixes at `R = 2/3` exactly, independent of piece count. The limitation it exposes points straight at the
next rung. The flat profile is not a dead end for lack of gradient — it is a saddle, and the productive
direction off it is to concentrate mass toward the centre and break the symmetry. The cleanest place to
let a search exploit that is at a small piece count, where the height vector is short enough that a
stochastic local search can explore the shape space thoroughly, follow the ascent directions I just
located, and rediscover the kind of `~20`-piece profile already known to clear `0.88` — a first real jump
off the `0.6667` floor.
