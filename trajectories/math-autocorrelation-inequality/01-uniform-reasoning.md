I want a non-negative step function whose autoconvolution is as "spread out yet peaked" as possible, in
the precise sense the ratio measures: lots of `L_2` energy relative to the product of the peak height and
the total mass. Before I try to be clever I fix the floor — the simplest legal function — because I need
to know what a value of `R` even feels like, I need a baseline every later step has to beat, and the
hand-computed case is the cleanest audit of the evaluator. The simplest member of the class is the flat
one: all heights equal, `v_n = 1`. That is the discretized indicator of an interval, its autoconvolution
is the cleanest object I can reason about by hand, and it anchors the scale.

Take `f` to be the indicator of an interval of length `1` (dilation invariance means the length does not
matter). Its autoconvolution is the classic triangle — the "tent" — supported on `[0, 2]`, rising linearly
from `0` to a peak of `1` at the midpoint and falling back. The peak is `1`, so the sup-norm is `1`. The
area is base times height over two, `2·1/2 = 1`, so the `L_1` norm is `1`. And the `L_2` norm squared is
`2·∫_0^1 x^2 dx = 2/3` by symmetry. So `R = (2/3)/(1·1) = 2/3` in the continuum.

The evaluator does not see a continuum, though — it self-convolves a height vector and integrates a
*piecewise-linear* autoconvolution through the node formulas. Take the flat vector of length `N`. Its
self-convolution `c = v*v` counts index pairs summing to `k`, the discrete triangle `c = (1, 2, …, N, …,
2, 1)`, so `L = (0, 1, 2, …, N, …, 2, 1, 0)` on `2N+1` nodes with apex `N`. The three exact norms fall
out arithmetically: `L_inf = N`; the trapezoid `L_1 = ½ Σ (L_j + L_{j+1})` gives `N^2`; and the per-cell
quadratic sum for `L_2^2` gives `2N^3/3`. Together,

`R = (2N^3/3) / (N · N^2) = 2/3,`

for *every* `N`, with no approximation — and no discretization error either: the per-cell formulas
`⅓(L_j^2 + L_j L_{j+1} + L_{j+1}^2)` and `½(L_j + L_{j+1})` are the exact integrals of a squared line and
a line over each cell, and an autoconvolution of a step function is exactly piecewise linear, so the
reported `R` is the genuine continuous value at any `N`. This is the load-bearing fact: piece count on its
own is not a lever. A flat vector of ten ones and one of a thousand ones both score exactly `2/3`; only
the *shape* of the heights moves `R`, and every future gain has to come from bending the autoconvolution
away from the tent, not from adding pieces. It is also what lets me later put thousands of pieces under an
optimizer and trust the number — the grid controls the *space of shapes* I can represent, never the
accuracy with which a given shape is scored.

Now, which way to bend. Writing `g = f*f ≥ 0`, the score is `R = ‖g‖_2^2 / (‖g‖_∞ ‖g‖_1)`, and
`‖g‖_2^2 = ∫ g^2 ≤ ‖g‖_∞ ∫ g` is exactly the Hölder inequality giving `R ≤ 1`, with equality iff
`g` takes only the values `0` and `‖g‖_∞` — iff `g` is a scaled indicator. So `R` literally measures how
close the autoconvolution is to being a two-valued *box*. The tent is the opposite extreme: its graph
passes through every value in `[0,1]`, spending mass and width on the thin near-zero tails that inflate
`‖g‖_1` while contributing almost nothing to `‖g‖_2^2`. And an autoconvolution can never actually *be* a
box: `g = f*f` of a compactly supported non-negative `f` is continuous and tapers to zero at the edges of
its support, so `R = 1` is forever out of reach and the true `C2` is strictly below it — consistent with
the published bounds topping out near `0.96`, not `1`.

To turn "closer to a box is better" into something steerable I compute `R` for a symmetric trapezoid
autoconvolution: linear ramps of width `a` rising to and falling from a flat cap of width `b` at height
`1`. Then `‖g‖_∞ = 1`, `‖g‖_1 = a + b`, and `‖g‖_2^2 = 2a/3 + b`, so

`R = (2a/3 + b) / (a + b).`

At `b = 0` this is the pure tent, `2/3`. At `b = a` it is `5/6 ≈ 0.833`; at `b = 4a`, `≈ 0.933`; as the
cap grows without bound, `R → 1`. Differentiating, `dR/db = (a/3)/(a+b)^2`, which at `b = 0` is
`1/(3a) > 0`: even an infinitesimal flat cap strictly raises `R`, so the tent is not a local maximum over
shapes — there is a genuine ascent direction and it is exactly "flatten the top." The narrower the ramps,
the larger the marginal payoff, so steep sides and a wide cap reinforce each other. The whole climb from
`0.6667` toward the frontier is, in this language, the project of growing a flat-topped plateau on the
autoconvolution while keeping the sides steep and the tails short.

A layer-cake reading gives the single variable I will track from here. For non-negative `g`, with
`μ(t) = |{g > t}|` the super-level width and `T = ‖g‖_∞`, the identities `‖g‖_1 = ∫_0^T μ` and
`‖g‖_2^2 = ∫_0^T 2t\,μ` give

`R = 2⟨t⟩ / T,   ⟨t⟩ = ∫ t μ / ∫ μ,`

so `R` is twice the μ-weighted average level, normalized by the peak — how high up, on average, the
autoconvolution keeps its width. It is `1` when all the width sits at the top (a box, `⟨t⟩ = T/2`); the
tent's linearly-shrinking `μ(t) = 2a(1 − t/T)` gives `⟨t⟩ = T/3` and `R = 2/3`. Every step from here is,
in this one variable, the project of raising `⟨t⟩/T` from the tent's `1/3` toward the box's `1/2`. The
records near `0.96` are shapes whose `μ(t)` stays large until very close to `T` and then falls off fast,
with a thin unavoidable taper at the very top that continuity forces.

That taper quietly tells me about the shape of the eventual `f`. The map `f ↦ f*f` is a *smoothing*
operator: it spreads mass, rounds corners, turns a sharp box into a rounded tent. To make the *output*
sharp-edged and flat-topped, the *input* has to be pre-distorted — concentrated enough to build the
autoconvolution's height fast at the edges of the cap, shaped enough to hold the cap level across its
width. So the winning profile is neither a single interval (which convolves to the over-rounded tent) nor
mass at a few isolated points (which convolves to an empty comb of needles, tiny `μ(t)` everywhere except
the tips). It lives between: something like a concentrated spike flanked by shaped shoulders. No tidy
analytic `f` will do it, and this is the first hint that the winning height profiles will be irregular,
not any closed form.

The last thing to pin down is whether the flat profile offers a shortcut — a gentle slope to walk up from
here. It does not: nudging one interior height upward moves `R` only in the fourth or fifth decimal
(`0.666667 → ~0.66701` for a perturbation of `0.02`), a random or ramp direction barely moves it at all,
nearly symmetric in `±`. The flat point sits in a shallow, almost-flat region dominated by the symmetry
that makes all `N` pieces interchangeable. This reconciles with the trapezoid arithmetic: over the space
of *shapes* there is a strict ascent direction out of the tent (flatten the cap), but over the space of
*height vectors* that ascent is not aligned with any single coordinate — growing a flat cap is a
coordinated reshaping of many heights at once, and no lone nudge realizes more than a sliver of it. So the
flat point is not a local maximum but a featureless plateau to a local probe, its `2/3` locked in by rigid
symmetry, and it is the honest floor: the unique maximally-symmetric, parameter-free, guaranteed-legal
member, whose analytically transparent tent doubles as the cleanest audit of the harness.

So `R = 2/3` exactly, the same for any piece count. Whatever bends the autoconvolution into a flat-capped
plateau will have to be manufactured by a search that makes coordinated moves and crosses the shallow
valley, not discovered by nudging one height. And the trapezoid arithmetic says what that search should
find: because even a modest cap lifts a shape into the high `0.8`s (a cap as wide as one ramp reaches
`5/6`), once *any* plateau grows I expect a large discrete jump off `0.6667`, not a marginal creep. That
jump should start where it is tractable — a small piece count, short enough for a stochastic search to
canvas the shape space and cross the valley, near the `~20`-piece scale already known to clear `0.88`.
