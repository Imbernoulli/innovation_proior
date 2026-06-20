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
midpoint and falling back to `0`. The peak of that triangle is the key quantity, because the numerator of `R`
is exactly the peak of the autoconvolution. For the indicator of width `w` and height `h`, the autoconvolution
peaks at the value `h^2 · w` (the full overlap at the center), while the mass is `∫ f = h·w`, so the square of
the mass is `h^2 w^2`. The continuous ratio `peak / (∫f)^2` is then `h^2 w / (h^2 w^2) = 1/w`. The discrete
evaluator carries a normalization factor that turns this dimensionless ratio into the constant on the fixed
interval `[-1/4,1/4]`: with `N` unit pieces the factor is `2N`, and a short check on the discrete heights makes
the arithmetic concrete.

Let me do that discrete check, because I want to *trust* the `2N` factor rather than take it on faith. With
`a_n = 1` for `n = 0 … N−1`, the discrete self-convolution `b = a*a` is the triangular sequence that rises
`1, 2, 3, …` up to its peak and falls back symmetrically; its maximum is `N` (the center term, where all `N`
overlaps line up). The mass is `Σ a_n = N`, so `(Σ a_n)^2 = N^2`. The evaluator returns `2N · max(b) / (Σ a)^2
= 2N · N / N^2 = 2`. So the flat function scores exactly `2`, and — crucially — this is *independent of how
finely I discretize it*. A flat vector of `10` ones and a flat vector of `1000` ones both have a triangular
autoconvolution and both score `2` exactly. The number of pieces is a red herring on its own; only the *shape*
of the heights moves the ratio. This is the same lesson the sibling maximization taught at its own floor, and
it tells me the same thing about why this problem is hard: refining a flat profile buys nothing.

That last fact is the real content of this rung. The piece count `N` is not itself a lever — refining a flat
profile leaves the triangle, and the value `2`, untouched. What moves `R` *down* is making the autoconvolution
*less peaked* relative to its mass: I want a height profile whose self-convolution spreads its energy out so the
single tallest node is suppressed, rather than concentrating it into one sharp central spike the way the tent
does. A triangle is, in this minimization, a bad case among unimodal autoconvolutions — it puts a lot of its
total overlap into one central peak. To get below `2` I have to reshape the heights so the autoconvolution
becomes flatter-*topped* in the sense of having no single dominant node — the peak shared across a plateau, or
pushed off-center, so that `max_k b_k` falls even as the mass `Σ a_n` is held fixed.

I notice the parity of the situation matters too, and it points the opposite way from the sibling problem. There
the gains came from making the autoconvolution *more* indicator-like (a flatter cap raising an L2-ratio). Here
the gains come from making the *peak* of the autoconvolution low relative to the squared mass — which, for a
fixed mass budget, means spreading the heights so that no single shift of `f` against itself lines up too much
overlap. Intuitively that wants an *asymmetric* or *structured* height profile: a flat function maximizes the
central self-overlap (every piece aligns with every piece at zero shift's neighborhood), which is exactly the
wrong thing. Some non-flat arrangement — heavier near the ends, lighter in the middle, or genuinely irregular —
should lower the worst-case overlap. But none of that is available to a flat vector, which has nothing to vary.
The flat function is a strict, rigid baseline parked at the top of the achievable range with no internal degree
of freedom to spend.

I should pin down the rest of the landscape honestly so I know how far I am from the bottom before I start
descending. The provable lower bound on `C1` is `1.28`, so no construction can ever certify below that — the
true constant lives somewhere in `[1.28, 1.5028…]`. The published upper-bound record is `1.5028628969`, reached
by elaborate optimized step functions with tens of thousands of pieces. So the distance from this flat ceiling
at `2.0` down to the frontier near `1.50286` is large, and essentially *all* of it has to be bought by
optimizing the heights into a non-trivial, asymmetric, structured profile. The flat function buys none of it;
it just establishes the starting altitude.

What I expect from the evaluator, then, is `R = 2.0` exactly, confirming both the triangle analysis and the
dilation/refinement invariance — and confirming that the scaffold's `2N` normalization matches my hand
computation, so I can trust the harness on the harder rungs. (As a cross-check I would also confirm the
published record sequence scores `1.50286` through this same evaluator, so I know the machine agrees with the
literature on both ends.) The limitation this rung exposes is sharp and sets up the next one: the flat profile
is a local nothing, with no gradient to follow because every piece is identical and the autoconvolution is
locked to a triangle. To move at all I have to introduce *variation* among the heights and let some search
procedure discover which non-flat profile suppresses the autoconvolution's peak. The cleanest place to start
that search is at a small piece count, where the height vector is short enough that a stochastic local search
can explore the shape space thoroughly and find a first real drop off the `2.0` ceiling.
