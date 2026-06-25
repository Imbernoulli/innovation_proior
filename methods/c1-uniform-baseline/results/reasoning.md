I want a non-negative step function whose autoconvolution has its peak as *low as possible* relative to the
square of the function's mass — that is the whole content of the ratio I am minimizing. Before I try to be
clever I should fix the ceiling: the simplest legal function, so I know what a value of `R` even feels like
on this problem and what every later rung has to beat by going *below* it. This is the opposite reflex from a
maximization problem — here the baseline is a ceiling I am trying to push down, not a floor I am trying to
climb. The simplest function in the class is a flat one: all heights equal, `a_n = 1` for every piece. That is
the discretized indicator of an interval, and its autoconvolution is the cleanest object I can reason about by
hand.

So let me reason about it by hand, because the point of this rung is to *understand* the functional, not just
read a number off the evaluator. Take `f` to be the indicator of an interval; by dilation invariance the length
does not matter, so I take the support to be the unit interval. Its autoconvolution is the classic triangle —
the "tent" — supported on an interval of twice the width, rising linearly from `0` to a single peak at the
midpoint and falling back to `0`. The peak of that triangle is the quantity that matters, because the numerator
of `R` is exactly the peak of the autoconvolution. For the indicator of width `w` and height `h`, the
autoconvolution peaks at the value `h^2 · w` (the full overlap at the center), while the mass is `∫ f = h·w`, so
the square of the mass is `h^2 w^2`. The continuous ratio `peak / (∫f)^2` is then `h^2 w / (h^2 w^2) = 1/w`. The
discrete evaluator carries a normalization factor that is supposed to turn this dimensionless ratio into the
constant on the fixed interval `[-1/4,1/4]`: with `N` unit pieces the factor is `2N`. I do not want to take that
factor on faith, so let me check the discrete arithmetic explicitly.

With `a_n = 1` for `n = 0 … N−1`, the discrete self-convolution `b = a*a` should be the triangular sequence that
rises `1, 2, 3, …` up to its peak and falls back symmetrically. Let me actually compute it for a couple of small
`N` rather than assert the shape. For `N = 5` the evaluator's `fftconvolve(a,a)` returns `[1, 2, 3, 4, 5, 4, 3,
2, 1]`, and for `N = 6` it returns `[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]` — the triangle, peaking at the center
value `N`, exactly as the hand picture says. So `max(b) = N`. The mass is `Σ a_n = N`, hence `(Σ a_n)^2 = N^2`,
and the score is `2N · max(b) / (Σ a)^2 = 2N · N / N^2 = 2`. When I run the evaluator it returns
`2.0000000000000000` for `N = 5, 10, 100` and `2.0000000000000004` for `N = 600, 1000` — the tiny `4e-16` tail
is just FFT round-off in the larger transforms, not a real deviation. So the flat function scores `2`, and the
score is the same whether I discretize with `10` pieces or `1000`. The number of pieces is a red herring on its
own; only the *shape* of the heights moves the ratio. That tells me something about why the problem is hard:
refining a flat profile buys nothing, because the triangle — and the value `2` — is locked in by the constant
height, not by the resolution.

That last fact is the real content of this rung, and it forces the question of *what* does move `R`. The triangle
puts a lot of total overlap into one central node; to get below `2` I have to reshape the heights so that the
self-convolution is less peaked relative to its mass — so that `max_k b_k` falls while `Σ a_n` is held fixed. The
tempting move is to guess a closed-form shape for the heights. My first instinct is that the central peak comes
from every piece overlapping every other piece near zero shift, so I should *thin out the middle* and put weight
near the ends — a U-shaped profile, heavier at the boundary, lighter in the center, which intuitively should
reduce the central self-overlap. Let me not just trust that picture; let me put a U-shape through the evaluator
at `N = 30` and see.

The result is the opposite of what I guessed. A flat profile scores `2.0`; a mild U-shape `a = 1 + c·x^2`
(with `x` ranging over `[-1,1]`) scores `2.007` at `c = 0.2`, `2.036` at `c = 0.5`, `2.110` at `c = 1.0`, and
`2.276` at `c = 2.0` — monotonically *worse* the more I pull weight to the ends. So "heavier near the ends" is
flatly wrong-signed. And the symmetric alternative is no better: a middle-heavy bump `a = 1 − c·x^2` rises the
same way, `2.009`, `2.075`, `2.355` as `c` grows. I check *why* by looking at where the peak of `a*a` sits for
the U-shape, expecting maybe it has moved off-center where I could exploit it — but `argmax(a*a)` is still at
index `29` of `59`, dead center, exactly where the flat triangle peaks. A smooth symmetric perturbation in either
direction keeps the dominant node at the center *and* makes it taller relative to the mass. The clean intuition
("spread the weight to the ends") does not survive contact with the evaluator; both obvious one-parameter
families sit strictly above `2`.

So the lever is not a smooth closed-form shape at all — it has to be a genuinely *irregular* height profile, the
kind a search finds and an ansatz misses. To confirm that something below `2` is even reachable, I run a crude
stochastic local search at `N = 30`: start from the flat vector, add Gaussian jitter, clip to non-negative, keep
the best of twenty thousand draws. That gets down to `R ≈ 1.817` — comfortably below the flat ceiling, and well
above the published frontier near `1.50286`. The drop is real, but it took an unstructured, searched profile to
get it; neither monotone family I tried by hand could. That is the lesson this rung hands forward: the gain lives
in irregular height vectors, found by search, not in any tidy U-shape or bump I can write down.

I should pin down the rest of the landscape honestly so I know how far I am from the bottom before I start
descending in earnest. The provable lower bound on `C1` is `1.28`, so no construction can ever certify below that
— the true constant lives somewhere in `[1.28, 1.5028…]`. The published upper-bound record is `1.5028628969`,
reached by elaborate optimized step functions with tens of thousands of pieces. So the distance from this flat
ceiling at `2.0` down to the frontier near `1.50286` is large — my twenty-thousand-draw search at `N = 30`
covered only the first third of it — and essentially *all* of it has to be bought by optimizing the heights into
a non-trivial, irregular profile. The flat function buys none of it; it just establishes the starting altitude.

What this rung delivers, then, is a verified ceiling and a verified harness: `R = 2.0` exactly for the flat
profile, matching the triangle-by-hand computation across piece counts, so the scaffold's `2N` normalization is
the right one and I can trust the machine on the harder rungs. (As a further cross-check I would still want to
push the published record sequence through this same evaluator and confirm it scores `1.50286`, so I know the
harness agrees with the literature on *both* ends, not just at the ceiling — that I have not run here.) The
limitation this rung exposes is sharp and sets up the next one: the flat profile is a local nothing, with no
gradient to follow because every piece is identical and the autoconvolution is locked to a triangle; and the two
nearest closed-form escapes both move the score the wrong way. To make progress I have to introduce *variation*
among the heights and let a search procedure discover which irregular profile suppresses the autoconvolution's
peak. The cleanest place to start that search is at a small piece count, where the height vector is short enough
that a stochastic local search can explore the shape space thoroughly and find a first real drop off the `2.0`
ceiling — which the quick `N = 30` probe already shows is there to be found.
