I want a non-negative step function whose autoconvolution is as "spread out yet peaked" as possible, in
the precise sense the ratio measures: lots of `L_2` energy relative to the product of the peak height and
the total mass. Before I try to be clever, I should fix the floor — the simplest legal function — because
I need to know what a value of `R` even feels like on this problem, and I need a baseline that every later
rung has to beat. The simplest function in the class is a flat one: all heights equal, `v_n = 1` for every
piece. That is the discretized indicator of an interval, and its autoconvolution is the cleanest object I
can reason about by hand.

So let me reason about it by hand, because the whole point of this rung is to *understand* the functional,
not just to read a number off the evaluator. Take `f` to be the indicator of an interval of length `1`
(dilation invariance means the length does not matter, so I take it to be `1`). Its autoconvolution is the
classic triangle — the "tent" — supported on `[0, 2]`, rising linearly from `0` to a peak of height `1` at
the midpoint and falling back to `0`. Everything I need is read off that triangle. The peak is `1`, so the
sup-norm is `1`. The area under the triangle is base times height over two, `2·1/2 = 1`, so the `L_1` norm
is `1`. And the `L_2` norm squared is the integral of the triangle squared: by symmetry that is twice the
integral of a line of slope one squared from `0` to `1`, which is `2·∫_0^1 x^2 dx = 2/3`. So the ratio is
`(2/3) / (1·1) = 2/3`. The flat function scores exactly `2/3`, and — crucially — this is *independent of
how finely I discretize it*. A flat vector of `10` ones and a flat vector of `1000` ones both have a
triangular autoconvolution and both score `2/3` exactly. The number of pieces is a red herring on its own;
only the *shape* of the heights moves the ratio.

That last fact is the real lesson of this rung, and it tells me something about why the problem is hard.
The piece count `N` is not itself a lever — refining a flat profile buys nothing. What moves `R` is making
the autoconvolution *less triangular*: I want a curve that holds more of its mass near its peak (raising
the `L_2`-to-`L_1·L_inf` ratio toward `1`) rather than tapering away linearly on both sides the way a tent
does. A triangle is, in a sense, the worst case among unimodal autoconvolutions — it spends a lot of `L_1`
mass and a lot of width on the thin tails near the base, mass that contributes little to `L_2` but inflates
the `L_1` denominator. To beat `2/3` I have to reshape the heights so the autoconvolution becomes flatter
on top and steeper on the sides — closer to the indicator that Hölder says would give `1` but that no
autoconvolution can actually be.

I notice the parity of the situation matters too. The triangle's `L_2^2/L_1` is exactly `2/3` because the
tent is so symmetric and so linear; any *asymmetry* or *internal structure* in the heights `v_n` will, in
general, create a more complicated piecewise-linear autoconvolution with a flatter cap, and that is where
the gains will come from. But none of that is available to a flat vector, which has nothing to vary. The
flat function is a strict, rigid baseline: it is the unique maximally-symmetric member of the class, and it
is parked at the bottom of the achievable range with no internal degree of freedom to spend.

I should also pin down the ceiling honestly so I know how far I am from the top before I start climbing.
Hölder gives `R ≤ 1`, but that is unattainable here — the autoconvolution of a non-negative function is
continuous and bump-shaped, never a sharp indicator, so `R = 1` is forever out of reach and the true
supremum `C2` is strictly below `1`. The published lower bounds say `C2` is at least around `0.96`, reached
by elaborate optimized step functions with tens of thousands of pieces. So the distance from this flat
floor at `0.6667` to the frontier near `0.96` is enormous, and essentially *all* of it has to be bought by
optimizing the heights into a non-trivial, asymmetric, structured profile. The flat function buys none of
it; it just establishes the starting altitude.

What I expect from the evaluator, then, is `R = 0.6667` exactly, confirming both the triangle analysis and
the dilation/refinement invariance — and confirming that the scaffold's notion of "flat = floor" matches
my hand computation, so I can trust the harness on the harder rungs. The limitation this rung exposes is
sharp and sets up the next one: the flat profile is a local nothing, with no gradient to follow because
every piece is identical and the autoconvolution is locked to a triangle. To move at all I have to
introduce *variation* among the heights and let some search procedure discover which non-flat profile bends
the autoconvolution away from the tent. The cleanest place to start that search is at a small piece count,
where the height vector is short enough that a stochastic local search can explore the shape space
thoroughly and rediscover the kind of `~20`-piece profile that is already known to clear `0.88` — a first
real jump off the `0.6667` floor.
