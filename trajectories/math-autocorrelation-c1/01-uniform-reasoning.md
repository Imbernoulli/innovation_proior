I want a non-negative step function whose autoconvolution has its peak as *low as possible* relative to the
square of the function's mass — that is the whole content of the ratio I am minimizing. Before I try to be
clever I should fix the ceiling: the simplest legal function, so I know what a value of `R` feels like on this
problem and what every later rung has to beat by going *below* it. This is the opposite reflex from a
maximization problem — here the baseline is a ceiling I am pushing down, not a floor I am climbing. The
simplest function in the class is a flat one, all heights equal, `a_n = 1` for every piece: the discretized
indicator of an interval, whose autoconvolution I can reason about entirely by hand.

Take `f` to be the indicator of an interval; by dilation invariance the length does not matter, so I take the
support to be `[-1/4, 1/4]` with height `h`. Its autoconvolution is `(f*f)(t) = ∫ f(x) f(t−x) dx = h^2` times
the length of the overlap between `[-1/4,1/4]` and the shifted copy `[t−1/4, t+1/4]`. For `t ∈ [0, 1/2]` that
overlap runs from `t−1/4` to `1/4`, length `1/2 − t`, so `(f*f)(t) = h^2 (1/2 − |t|)` — a tent of height
`h^2/2` at the origin, vanishing at `t = ±1/2`. The constraint `|t| ≤ 1/2` in the functional is exactly the
support of this tent, so the max is the central peak `h^2/2`. The mass is `∫ f = h/2`, so `(∫ f)^2 = h^2/4`,
and the ratio is `(h^2/2)/(h^2/4) = 2`. The height and the interval length both cancel: the flat indicator
scores exactly `2`.

I redo this on the discrete side, because I want to *trust* the `2N` factor in the evaluator rather than take
it on faith. With `a_n = 1` for `n = 0 … N−1`, the discrete self-convolution `b_k = Σ_n a_n a_{k−n}` counts
index pairs `(n, k−n)` both in `{0, …, N−1}` — the triangular sequence `1, 2, …, N, …, 2, 1`, peaking at
`N`. The mass is `N`, so `R = 2N · N / N^2 = 2`. Discrete and continuous agree, which tells me the `2N`
normalization is precisely the factor that carries the continuum ratio `1/w` (here `w = 1/2`) onto the fixed
interval. The harness arithmetic is sound.

One thing I should not gloss: why is `max_k` over discrete nodes legitimate when the functional asks `max_t`
over the continuum? Because the autoconvolution of a piecewise-constant function is piecewise *linear* —
convolving two step functions with integer breakpoints gives a polyline with kinks only at integer arguments,
and a polyline attains its maximum at a breakpoint. So `max_t (f*f)(t) = max_k b_k` exactly, no discretization
error; that is the reason the whole problem lives on the finite vector `b`.

The next fact governs the entire ladder. Refining the grid does nothing: `R = 2` for *every* `N`. Scaling every
height by a constant `c` scales `b` by `c^2` and `(Σ a)^2` by `c^2`, so `R` is again unchanged. The piece count
and the overall amplitude are both non-levers; only the *shape* of the height profile can move `R`. So when the
harness reports `R = 2` at every piece count I check, coarse and fine alike, that is one fact confirmed
repeatedly, not several data points.

I can sharpen "only shape matters" into a quantitative target with a sum rule that costs nothing:
`Σ_k b_k = Σ_k Σ_n a_n a_{k−n} = (Σ a)^2`. There are `2N − 1` nodes, so their average is `(Σ a)^2 / (2N − 1)`,
and `R = 2N · peak / (Σ a)^2 = [2N/(2N−1)] · (peak / mean)`. The problem is now transparent: up to the mild
prefactor `2N/(2N−1) → 1`, `R` is just the ratio of the tallest autoconvolution node to the average node. For
the flat profile peak `= N`, mean `= N^2/(2N−1) ≈ N/2`, so `peak/mean ≈ 2` — the *worst* way to distribute the
fixed total overlap `(Σ a)^2`, all of it poured into one central spike. Every gain from here must come from
spreading that same total across the nodes so the tallest sits closer to the average.

The same sum rule bounds how far that could go, and exposes how weak the trivial bound is. Since the peak can
never fall below the mean, `R ≥ 2N/(2N−1)`, which approaches `1` as the grid refines — far below the analytic
lower bound `1.28` the literature proves for `C1`. So the averaging argument leaves almost all of the real
constraint on the table: the gap between the trivial `≈1` and the proved `1.28` is entirely the non-negativity
constraint. Were signed heights allowed I could genuinely flatten `a*a` toward its mean and drive `R` toward
`1`; it is `a_n ≥ 0` — every node `b_k` a sum of non-negative products — that forbids the perfect flat top. The
whole difficulty of the descent lives in reconciling "flatten the top" with "stay non-negative," and no
averaging shortcut touches it. What the sum rule *is* good for is calibration: the best published upper bound is
`1.5098`, so a record-grade profile has `peak/mean ≈ 1.51` at large `N`. The enterprise of this ladder — from
the ceiling at `2` down toward `1.51` — is, in one line, cutting the flat profile's `peak/mean ≈ 2` to about
`1.5` while holding the mass fixed, a factor of `4/3` off the worst node.

I do not want to leave "some non-flat shape scores below `2`" on faith when a two-piece computation settles it.
Take `a = [1, 2]`: `b = [1, 4, 4]`, peak `4`, mass `3`, so `R = 2·2·4 / 9 = 16/9 ≈ 1.778` — a single broken
symmetry already clears the ceiling. Through the sum rule it is more telling: the three nodes sum to `9` with
mean `3`, so `peak/mean = 4/3 ≈ 1.333`, a dramatic cut from `2`. The reason `R` only fell to `1.778` and not to
`1.333` is the prefactor: at `N = 2`, `2N/(2N−1) = 4/3` is large and inflates the score. That is a real warning
about scale — at tiny `N` the reported `R` is a poor read on the underlying `peak/mean`; only at
moderate-to-large `N`, where `2N/(2N−1) → 1`, does the score track the shape, and there the shape space is large
enough that I cannot guess the optimum and will have to search it.

Is the flat function actually the right ceiling, or just a convenient one? It is the unique maximally-symmetric
member of the class, parameter-free, provably admissible, and exactly hand-computable — and the symmetry is what
makes it the ceiling rather than an arbitrary point. Its symmetry forces every piece to overlap every other
maximally near zero shift, piling the total overlap into the central node. Any candidate that breaks that
symmetry can only spread the overlap out and lower the peak. So the flat function does the wrong thing on
purpose, cleanly, and certifies `C1 ≤ 2` with a one-line proof. (A random vector would report an accident of a
seed; a smooth bump would already spend the one lever I have — shape — before I understand why it helps.)

There is a subtlety in "spread the overlap out" that makes the descent a genuine optimization rather than a
one-move fix. Lowering the central node is easy — starve the zero-shift self-overlap — but the total overlap
`(Σ a)^2` is conserved, so whatever I take out of the center reappears in the off-center nodes. Overdo it and I
merely trade a tall central node for a tall off-center one; the max is over *all* nodes. The real object is not
"flatten the middle" but "equalize the top of the whole node profile," and pushing one node down tends to push a
neighbor up. That coupling is why a single greedy adjustment cannot get far, and it is on the record now that the
eventual method will have to act on the whole set of near-tallest nodes at once.

I should also pin down where the flat profile sits locally, because that determines what kind of move can leave
it. Perturbing a single height `a_j → 1 + ε`: at `N = 4`, whether I bump an interior or an end height, the
evaluator returns `R − 2 = −2ε^2/16 + O(ε^3)` — first-order term vanishing, second-order negative. The
vanishing gradient is not an accident of `N = 4`: at the flat profile the argmax is the center and its gradient
with respect to each height is the same constant, so the peak term and the mass-normalization term in `∂R`
cancel, and the gradient of `R` is identically zero. So the flat profile is a stationary point of `R`, but of
*maximum* type — a hilltop, with downhill in essentially every direction, only quadratically. A pure
first-order method parked here would never move; anything willing to take a finite step, or to read the negative
curvature, leaves easily. The descent is hard not because flat is a trap but because once off the hilltop the
landscape is non-convex — the `max` over nodes is non-smooth, `a*a` is bilinear — and riddled with genuine local
minima, and the profile that reaches `peak/mean ≈ 1.5` is a specific asymmetric shape I cannot write down. It
has to be *found*, by something that does not simply roll into the first basin.

What I genuinely do not know is which asymmetry wins. The `peak/mean` and continuum arguments both point toward
"heavier near the ends, lighter in the middle," but they do not say whether the optimum is left-right symmetric
with two matching end-humps, lopsided with a single dominant boundary spike, or something combier than either.
The `[1,2]` probe was lopsided and paid; the tent argument is symmetric; these are not the same prediction, and
I should not pretend to resolve them from the ceiling. All I commit to is that *some* structured, non-flat,
end-weighted profile drops `R`, and finding the specific one is the work of the searching rungs.

So I expect the harness to report `R = 2.0` exactly at every piece count, visibly independent of `N`; anything
else would mean my model of the functional is wrong. The limitation this rung exposes sets up the next one: with
no internal degree of freedom and a vanishing gradient, to move at all I have to introduce genuine variation
among the heights and let a search — one willing to take finite, non-improving steps — discover which
asymmetric profile first cuts `peak/mean` off the ceiling. The place to begin is a small piece count. Around
`N = 50` is the sweet spot: the prefactor `2N/(2N−1) = 1.0101` is already within a percent of `1`, so the score
is an honest read on the shape and not the tiny-`N` inflation that held the `[1,2]` probe up at `1.778`, while
`50` heights is short enough — only `99` autoconvolution nodes — that a stochastic search evaluating tens of
thousands of candidates can revisit every coordinate hundreds of times and actually canvas the space. Going
straight to thousands of pieces would push the prefactor closer to `1` but blow up the dimension and leave most
coordinates barely touched. So a first descent at `N ≈ 50` should settle which asymmetry wins, and leave the
question of how low the peak can ultimately be pushed, once the grid is fine, for the rungs after it.
