I want a non-negative step function whose autoconvolution has its peak as *low as possible* relative to the
square of the function's mass — that is the whole content of the ratio I am minimizing. Before I try to be
clever I should fix the ceiling: the simplest legal function, so I know what a value of `R` even feels like
on this problem and what every later rung has to beat by going *below* it. This is the opposite reflex from a
maximization problem — here the baseline is a ceiling I am trying to push down, not a floor I am trying to
climb. The simplest function in the class is a flat one: all heights equal, `a_n = 1` for every piece. That is
the discretized indicator of an interval, and its autoconvolution is the cleanest object I can reason about by
hand. It is worth spending this whole rung on it, because the point is not to read a number off the evaluator
but to understand the functional well enough that every later move is a considered response to something I have
already computed.

So let me reason about the indicator by hand. Take `f` to be the indicator of an interval; by dilation
invariance the length does not matter, so I take the support to be the unit interval, or — to keep the fixed
harness interval in view — the interval `[-1/4, 1/4]` with some height `h`. Its autoconvolution is the classic
triangle, the "tent," supported on an interval of twice the width and rising linearly from `0` to a single peak
at the center before falling back to `0`. The peak of that triangle is the whole story, because the numerator
of `R` is exactly `max_t (f*f)(t)`. Let me actually write `(f*f)(t) = ∫ f(x) f(t−x) dx`; for the indicator of
`[-1/4,1/4]` this is `h^2` times the length of the overlap between `[-1/4,1/4]` and the shifted copy
`[t−1/4, t+1/4]`. For `t ∈ [0, 1/2]` that overlap runs from `t−1/4` up to `1/4`, a length of `1/2 − t`, so
`(f*f)(t) = h^2 (1/2 − |t|)` — a tent of height `h^2/2` at `t = 0`, vanishing at `t = ±1/2`. The constraint
`|t| ≤ 1/2` in the definition of the functional is exactly the support of this tent, so the max over `|t| ≤ 1/2`
is the global peak `h^2/2` at the origin. The mass is `∫ f = h · (1/2)`, so `(∫ f)^2 = h^2/4`, and the
continuous ratio is `peak / (∫f)^2 = (h^2/2) / (h^2/4) = 2`. The height `h` cancels and the interval length
cancels: the flat indicator scores exactly `2` no matter how I scale it. That is the ceiling, derived from the
continuum with nothing swept under a normalization constant.

Now let me redo the same computation on the discrete side, because I want to *trust* the `2N` factor in the
evaluator rather than take it on faith — the two derivations must agree or something is wrong. With `a_n = 1`
for `n = 0 … N−1`, the discrete self-convolution `b_k = (a*a)_k = Σ_n a_n a_{k−n}` counts the number of index
pairs `(n, k−n)` that both fall in `{0, …, N−1}`. That count is `min(k, N−1) − max(0, k−N+1) + 1`, which rises
`1, 2, 3, …` as `k` grows, reaches its maximum at the center `k = N−1`, where it equals `(N−1) − 0 + 1 = N`,
and falls back symmetrically — the triangular sequence `1, 2, …, N, …, 2, 1`. A tiny case makes it concrete:
`N = 3` gives `a = [1,1,1]`, `b = [1, 2, 3, 2, 1]`, peak `3 = N`, mass `Σ a = 3`, and the evaluator returns
`2·3·3 / 3^2 = 2`. In general the peak is `N`, the mass is `N`, so `R = 2N · N / N^2 = 2` exactly. The discrete
`2` and the continuous `2` match, which tells me the `2N` normalization is precisely the factor that carries the
dimensionless continuum ratio `1/w` (here `w = 1/2`, giving `2`) onto the fixed interval. I now trust the
harness arithmetic.

One thing I glossed and should not: why is it legitimate to take `max_k` over the *discrete* nodes when the
functional asks for `max_t` over the continuum? Because the autoconvolution of a piecewise-constant function is
piecewise *linear* — convolving two step functions with breakpoints at the integers produces a function whose
slope changes only at integer arguments, so it is a polyline with kinks at the nodes. A polyline on a bounded
interval attains its maximum at one of its breakpoints, never strictly between two of them, so `max_t (f*f)(t)`
really is `max_k b_k` with no discretization error. This is not an approximation the evaluator makes; it is
exact for step functions, and it is the reason the whole problem can be posed on the finite vector `b`.

The next fact is the one that governs the entire ladder, so I want it in the form of an equation, not a
feeling. Refining the grid does nothing: `R = 2N · N / N^2 = 2` for *every* `N`, and if I scale every height by
a common constant `c`, then `b` scales by `c^2` and `(Σ a)^2` scales by `c^2`, so `R` is unchanged again. The
piece count and the overall amplitude are both non-levers; only the *shape* of the height profile can move `R`.
A flat vector of `10` ones and a flat vector of `30000` ones both have a triangular autoconvolution and both
score `2` exactly. So when the harness later reports `R = 2` at every piece count I check — a coarse grid and a fine one alike — that will
not be several independent data points; it will be one fact confirmed repeatedly, and the invariance is the content.

The continuous picture says the same thing and explains *why* the flat value is exactly `2` rather than some
other constant. The tent `(f*f)(t) = h^2(1/2 − |t|)` is a triangle of base `1` (from `−1/2` to `1/2`) and peak
`h^2/2`. A triangle's mean ordinate over its base is exactly half its peak — its area is `½ · base · peak`, so
`mean = area/base = peak/2`. Hence for *any* triangular autoconvolution the peak is exactly twice the average
value of the profile, `peak/mean = 2` identically, and that `2` is what the ratio reports. So the flat indicator
does not score `2` by coincidence of the interval `[-1/4,1/4]`; it scores `2` because its autoconvolution is a
triangle and a triangle is, by pure geometry, the shape whose peak sits at twice its mean. To score below `2` I
have to make the autoconvolution *not* a triangle — flatter across its top, or split into off-center humps — so
that its peak drops toward its mean. That reframes "heavier near the ends" concretely: pulling density toward
the two endpoints of the support turns the single central pile-up of self-overlap into something with more mass
away from the center, which broadens and lowers the top of the autoconvolution. I cannot compute the optimal
redistribution by hand, but I can see the direction, and it agrees with the discrete `peak/mean` reading.

I can sharpen "only the shape matters" into a quantitative target using a sum rule that costs nothing to
derive. Summing every autoconvolution node, `Σ_k b_k = Σ_k Σ_n a_n a_{k−n} = (Σ_n a_n)(Σ_m a_m) = (Σ a)^2`.
There are `2N − 1` nodes, so their *average* value is `(Σ a)^2 / (2N − 1)`. Rewrite the ratio in terms of that
average: `R = 2N · peak / (Σ a)^2 = 2N · peak / [(2N−1) · mean] = [2N / (2N−1)] · (peak / mean)`. The problem is
now transparent — `R` is, up to the mild prefactor `2N/(2N−1) → 1`, just the ratio of the tallest
autoconvolution node to the average node. For the flat profile the peak is `N` and the mean is
`N^2 / (2N−1) ≈ N/2`, so `peak/mean ≈ 2`; concretely `1.98` at `N = 50` and `1.9983` at `N = 600`. The flat
indicator scores `2` for the most mechanical of reasons: its autoconvolution is a triangle whose single central
node is almost exactly twice the average node. That is the *worst* way to distribute the fixed total overlap
`(Σ a)^2` — pour it into one central spike. Every gain from here has to come from spreading that same total
mass of overlap across the nodes so the tallest is closer to the average, i.e. from pushing `peak/mean` down
from `2` toward `1`.

The same sum rule also tells me, for free, how far this pushing could ever go and how weak the trivial bound is.
Since the peak can never be below the mean, `R ≥ 2N/(2N−1)`, which is `1.0101` at `N = 50` and `1.00083` at
`N = 600` — approaching `1` as the grid refines. This trivial floor is far below the analytic lower bound `1.28`
that the literature proves for `C1`, which means the averaging argument leaves almost all of the real
constraint on the table: the true obstruction to a perfectly flat-topped autoconvolution is subtle, not a
counting bound. The gap between the trivial `≈1` and the proved `1.28` is entirely the non-negativity
constraint at work. If I were allowed signed heights I could genuinely flatten `a*a` toward its mean and drive
`R` toward `1`; it is the requirement `a_n ≥ 0` — every node `b_k` is a sum of *non-negative* products — that
forbids the perfect flat top and holds the achievable peak up above `1.28`. That is worth remembering, because
it says the whole difficulty of the descent lives in reconciling "flatten the top" with "stay non-negative,"
and no amount of clever averaging shortcuts it. What the sum rule *is* good for is calibrating the descent. The
best published upper bound is `1.5098`; translating through the prefactor, a record-grade profile has
`peak/mean ≈ 1.5098 · (2N−1)/(2N) ≈ 1.51` at large `N`. So the whole enterprise of this ladder — from the
ceiling at `2` down toward the published record near `1.51` — is, in one line, the project of taking the flat
profile's `peak/mean ≈ 2` and cutting it to about `1.5` while holding the mass fixed. That is a factor of `4/3`
on the worst node, and it will not come from any single clean formula; it will come from carving a height
profile whose self-overlap refuses to concentrate.

I do not want to leave "some non-flat shape scores below `2`" as an article of faith when I can settle it with
a two-piece hand computation. Take `a = [1, 2]` — the crudest possible asymmetry. Then `b = a*a = [1, 4, 4]`,
so the peak is `4`, the mass is `Σ a = 3`, and `R = 2·2·4 / 3^2 = 16/9 ≈ 1.7778`. A single broken symmetry, on
two pieces, already clears the ceiling by more than a fifth. Reading it through the sum rule is even more telling:
the three nodes sum to `9 = (Σ a)^2` with mean `3`, and the peak `4` gives `peak/mean = 4/3 ≈ 1.333`, a
dramatic cut from the flat `peak/mean ≈ 2`. So the shape lever is not merely real, it is *strong* — even a
two-value profile slashes the central concentration. The reason `R` only fell to `1.778` rather than to `1.333`
is the prefactor: at `N = 2` the factor `2N/(2N−1) = 4/3` is large and inflates the score, masking most of the
shape gain. This is a small warning about scale. At tiny `N` the prefactor dominates and the reported `R` is a
poor read on the underlying `peak/mean`; only at moderate-to-large `N`, where `2N/(2N−1) → 1`, does the score
track the shape faithfully — and there the shape space is large enough that I cannot guess the optimum and will
have to search it. So the honest lesson of the `[1,2]` probe is double: the lever is powerful, and I should
exercise it at a piece count large enough that the prefactor is out of the way but small enough that a search
can still canvas the space.

Before I commit to the flat function as the ceiling I should ask whether some other simple candidate would make
a better reference, because "simplest" is doing real work here. A random height vector would score below `2` but
would not be reproducible or meaningful — it would report an accident of a seed, not a landmark. A smooth bump
(a Gaussian or a raised cosine) also scores below `2`, but it is not exactly hand-computable: its `R` depends on
how I truncate and discretize it, so it would be a soft, arguable number rather than a fixed one, and it would
already be spending the one lever I have — shape — without my understanding *why* it helps. A two-value profile
(some tall pieces, some short) is closer to what the optimum will look like, but choosing the split is already a
one-parameter search, and this rung is supposed to precede all searching. The flat indicator is the unique
candidate that is parameter-free, provably admissible (all heights positive and finite), exactly computable by
hand, and *maximally symmetric* — and that last property is exactly why it is the ceiling rather than an
arbitrary point. Its symmetry is what forces every piece to overlap every other piece maximally near zero
shift, piling the total overlap into the central node. Any candidate that breaks that symmetry can only spread
the overlap out and lower the peak. So the flat function is not just a convenient reference; it is genuinely the
top of the achievable range, and it certifies `C1 ≤ 2` with a one-line proof.

There is a parity here that points the opposite way from the sibling maximization, and naming it keeps me from
importing the wrong intuition. In a maximization of an `L^2`-type ratio the gains come from making the
autoconvolution *more* indicator-like — a flatter, fuller cap. Here the gain comes from making the *peak* low
relative to the squared mass, which for a fixed mass budget means arranging the heights so that no single shift
of `f` against itself lines up too much overlap. Intuitively that wants an *asymmetric* or structured profile:
heavier near the ends and lighter in the middle, or genuinely irregular, so that the central self-overlap — the
thing the flat function maximizes — is deliberately starved. The flat function does the wrong thing on purpose,
and cleanly, which is what makes it the right ceiling.

There is a subtlety in "spread the overlap out" that I should not gloss, because it is what makes the descent a
genuine optimization rather than a one-move fix. Lowering the central node is easy — starve the zero-shift
self-overlap — but the total overlap `(Σ a)^2` is conserved by the sum rule, so whatever I take out of the
center has to go *somewhere*, into the off-center nodes. If I overdo it, I merely trade a tall central node for
a tall off-center node and the max does not fall; the maximum is over *all* the nodes, not just the middle one.
So the real object is not "flatten the middle" but "equalize the top of the whole node profile," and pushing one
node down tends to push a neighbor up. This is a conserved-mass, push-one-up-elsewhere problem — the reason a
single greedy adjustment cannot get far, and the reason I already sense the eventual method will have to act on
the whole set of near-tallest nodes at once rather than chase the current maximum. I cannot solve that here; I
just want it on the record that the ceiling hides a coupled problem.

I should also pin down where the flat profile sits *locally*, because that determines what kind of move can
leave it. Every piece is identical, so there is no internal coordinate to vary and no direction that is
obviously downhill — but "no obvious direction" is not the same as "trapped," and I want to know which it is. Let
me perturb a single height, `a_j → 1 + ε`, and watch `R`. At `N = 4`, whether I bump an interior height or an
end one, the evaluator returns `R − 2 = −2ε^2/16 + O(ε^3)`: the first-order term vanishes and the second-order
term is *negative*. The vanishing of the first-order term is not an accident of `N = 4`; it holds at every `N`,
because at the flat profile the single argmax node is the center and its gradient with respect to each height is
the same constant, so the peak term and the mass-normalization term in the derivative of `R` cancel exactly —
the gradient of `R` at the flat vector is identically zero. So the flat profile is a stationary point of `R` —
which is exactly why there is "no gradient to follow" — but it is a stationary point of *maximum* type: a small
finite perturbation in essentially any direction lowers `R`, only quadratically. That is a precise and useful
thing to know. It means a pure first-order method parked here would never move (its gradient is identically
zero), while any method willing to take a finite step, or to read the negative curvature, will find that the
flat profile is a hilltop with downhill in every direction. The reason the descent is nonetheless hard is not
that flat is a trap; it is that once I step off the hilltop the landscape is non-convex and riddled with genuine
local minima — a consequence of the `max` over nodes being non-smooth and `a*a` being bilinear — and the
profile that actually reaches `peak/mean ≈ 1.5` is a specific asymmetric shape I cannot write down. It has to
be *found*, and found by something that does not simply roll downhill into the first basin it meets.

What I genuinely do *not* know yet is which asymmetry wins. The `peak/mean` and continuum arguments both point
toward "heavier near the ends, lighter in the middle," but they do not tell me whether the optimum is
left-right symmetric with two matching end-humps, or lopsided with a single dominant spike at one boundary, or
something combier and more irregular than either. The `[1,2]` probe was lopsided and paid; the tent argument is
symmetric; these are not the same prediction, and I should not pretend to resolve them from the ceiling. All I
can commit to is that *some* structured, non-flat, end-weighted profile drops `R`, and that finding the specific
one is the work of the searching rungs. Carrying that honest uncertainty forward is better than guessing a shape
now and building the whole ladder on a hunch.

So what do I expect the harness to report, and what would falsify my understanding? I predict `R = 2.0` exactly
at every piece count I check — coarse, moderate, and fine grids alike — with the value visibly independent of `N`,
confirming both the triangle computation and the refinement invariance, and confirming that the evaluator's `2N`
normalization matches my hand arithmetic to full precision. If any of those came back different from `2`, my
model of the functional would be wrong and I would have to stop and rebuild it. As a second calibration I expect
a known optimized construction, scored through this same evaluator, to land near the published `1.5098` — so that the
machine agrees with the literature at both ends of the range, the ceiling I am standing on and the frontier I am
aiming at. The limitation this rung exposes is sharp and it sets up the next one exactly. The flat profile has
`peak/mean` pinned near `2`, no internal degree of freedom, and a vanishing gradient; to move at all I have to
introduce genuine *variation* among the heights and let some search — one that is willing to take finite,
non-improving steps, since the interesting structure lives past a forest of local minima — discover which
asymmetric, peak-suppressing profile first cuts `peak/mean` down off the ceiling. The cleanest place to begin
that search is at a small piece count, where the height vector is short enough that a stochastic local search
can canvas the whole shape space and find a first real drop below `2` before I ever worry about resolution.

That choice of scale is not arbitrary, and it is worth fixing now while the arithmetic is in front of me. I
need the reported `R` to track the underlying `peak/mean` faithfully, which means the prefactor `2N/(2N−1)`
should already be close to `1`: at `N = 50` it is `1.0101`, within a percent of unity, so the score is an honest
read on the shape and not a tiny-`N` inflation like the one that held the `[1,2]` probe up at `1.778`. At the
same time `50` heights is a genuinely short vector — its autoconvolution has only `99` nodes — so a stochastic
search that evaluates tens of thousands of candidates can revisit every coordinate hundreds of times and
actually canvas the space rather than drift through a sliver of it. Going straight to hundreds or thousands of
pieces would put the prefactor even closer to `1` but would blow up the dimension and leave most coordinates
barely touched in any affordable budget. So a piece count around `50` is the sweet spot for a *first* descent:
the prefactor is already negligible and the dimension is small enough to explore thoroughly, which should let
the next rung settle the shape — which asymmetry wins, how sparse the profile wants to be — cheaply, and leave
the separate question of resolution, of how low the peak can ultimately be pushed once the grid is fine, for the
rungs after it.
