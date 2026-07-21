The previous order returned exactly `2, 4, 8, 16, 32, 64, 128` for `n = 1..7`, `2^n` with no
deviation, matching the optimum only where `2^n` *is* the optimum and falling short after by
`1, 4, 13, 48, 108`. That is the rigid signature of a single mechanism. The walk builds `{0, 1}^n`,
the vectors that never use the symbol `2`, because the low-index points it grabs first close lines
whose third points are exactly the `2`-carrying vectors, sealing them out. The set is *maximal* —
nothing outside can be appended — so its deficit is entirely that one sealed pocket being smaller
than the best sealed configuration in the space. The order is the only lever, and the way this order
fails is by falling into a bad local optimum of the greedy landscape.

Reading the deficit as fractions tells me whether I am chasing a rounding error or a structural gap.
As a share of the optimum, `2^n` captures `80%` at `n = 4`, `71%`, `57%`, `54%` at `n = 5, 6, 7` —
a steadily decaying piece. The increments I would need to recover, `+4, +13, +48, +108`, grow faster
than the floor itself: recovering `+4` to reach `20` at `n = 4` is a `25%` lift, a modest ask;
recovering `+108` to reach `236` at `n = 7` would nearly double the floor, and doubling by lottery
is what a thin tail forbids. So I expect the recoverable fraction of the deficit to *shrink* with `n`
— sampling should reclaim most of a small gap and little of a large one.

Once the failure is "trapped in a bad local optimum," the responses are clear. A smarter but still
fixed order is one blind arrival sequence landing in one maximal cap, so colex or a Gray-code walk
just trades one arbitrary pocket for another. Genuine local search on caps — build a maximal cap,
remove a few points, regrow, hill-climb — can climb past a single greedy pass, but it needs a
neighborhood, an acceptance rule, and a cycling guard, three more levers at once, muddying the
accounting of what the *single* order-lever can buy. So I take the cheapest option first: if a fixed
order is one uncontrolled draw from the space of maximal caps, draw *many* times and keep the best.
The admission rule is identical, so every run still produces a valid cap; I only shuffle the offer
order and take the largest cap over many shuffles. If sampling's ceiling turns out low, that is
itself the clean result that would justify reaching for the heavier lever.

Why should a random order beat lexicographic? Lexicographic grabs a dense cluster of low-index
points first, and the lines they induce blacken a structured swath concentrated right where the walk
is about to step, so the frontier erodes early and locally and the fill suffocates at `2^n`. A
uniformly random order has no such clustering: its early admissions are scattered, the completions
they blacken are scattered too, and the frontier stays alive longer. But scattering is more than a
variance trick. `{0, 1}^n` uses only two symbols, and the strong caps *need* the third — at `n = 3`
the two-symbol corner caps out at `8`, so the `9`-cap must spend a coordinate on a `2`. Any fixed
order that front-loads one corner is structurally shut out of the full-alphabet region where the
large caps live; a random order front-loads no corner, so it can reach the full-alphabet maximal
caps. Scattering changes *which region of cap-space is reachable at all*.

Any single random order is as arbitrary as lexicographic, so the method works by taking the
*maximum* over many independent draws, reaching into the right tail. How far does best-of-`k` reach?
If the sizes are roughly bell-shaped with spread `σ`, the maximum of `k` draws sits about
`σ·√(2 ln k)` above the mean. That `√(ln k)` is the whole story of diminishing returns: going from
`1000` to `5000` restarts raises the expected best by only `σ·(4.13 − 3.72) ≈ 0.41 σ`, less than
half a standard deviation for a fivefold compute increase. So multi-start helps for the first few
thousand restarts and then saturates — a ceiling about the *shape* of the distribution, not compute.

The deeper ceiling also lives in the shape. Best-of-`k` can only reach the top of the support; it
cannot conjure a cap random greedy essentially never produces. Every greedy-maximal cap satisfies
`c(c − 1)/2 ≥ 3^n − c`, flooring the support near `1.73^n`, while the optimum sits near `2.2^n`. At
small `n` the optimum is *inside* the support — the space is tiny, few distinct maximal caps are
greedy-reachable, and an optimal configuration is a plausible random outcome. As `n` grows the large
caps become highly structured algebraic objects a random arrival sequence has vanishing odds of
tracing, so the distribution concentrates below the optimum and its upper tail ends short of it.
When the target is outside the support, no restart budget reaches it. That asymmetry — the tail is
thin *and* it ends in the wrong place — is what caps the method.

I can turn the small-`n` case into an estimate. If a fraction `p` of random orders greedily produce
a `20`-cap at `n = 4`, then `k` restarts hit one with probability `1 − (1 − p)^k`; even `p ≈ 10^{-3}`
is almost certain to surface within `5000` restarts (`1 − e^{-5} ≈ 0.993`). So as long as `20`-caps
occupy even a thin sliver of the reachable distribution, thousands of restarts find one, and the seed
just picks *which* success I report. The reason this fails at `n = 7` is not the restart count but
`p` collapsing to zero once the optimum leaves the reachable band — and no `k` rescues a `p` of zero.

Sizing the compute: one restart scans `3^n` vectors and blocks a line per admitted point against
every prior member, costing about `3^n + |cap|^2·n`. At `n = 4` that is tiny, so I can afford `5000`
restarts nearly free. By `n = 7` the cap is near `140`, each restart costs `~140^2·7 ≈ 1.4×10^5`,
and pushing past `1000` there quadruples wall-clock for the `0.41 σ` the curve buys. So the rational
schedule spends many restarts where they are cheap and dials down as `|cap|^2·n` grows: roughly
`5000` starts at `n = 4, 5`, tapering toward `~1000` by `n = 7`. I fix the seed so each best-of-`k`
is reproducible — though the honest reading is "the top of a `k`-sample at this seed," not "the value
random greedy achieves"; a rerun would wobble by a point or two where the tail is soft, so a
one-point difference between this baseline and some later deterministic order sits inside seed noise.

At `n = 3`, below the scored dimensions, the mechanism is checkable: lexicographic is sealed at `8`,
and a few hundred random restarts reach `9`, the proven optimum, which lexicographic could never
touch — reaching it demands admitting a `2`-vector before the `{0, 1}` cluster condemns it, exactly
the move a scattered order makes. That is the regime where the optimum is inside the support, and it
is why I expect `n = 4` to land *on* `20` rather than merely above `16`.

So the prediction: `n = 4` should land exactly on the optimum `20`; `n = 5, 6, 7` should each clear
their `2^n` floors decisively but land strictly below the optima `45, 112, 236`, with the shortfall
as a fraction of the optimum growing monotonically; and `n = 8`, with caps near five hundred and
each restart costing `~500^2·8`, I expect to leave essentially untouched. If instead `n = 5, 6, 7`
came in at their optima, or `n = 4` fell short of `20`, the escape-but-ceiling picture would be
wrong.

Whatever multi-start recovers, it fixes only the *bias* of a fixed order, not the *blindness*: every
order it tries is still uniform noise, with no preference for a point that blackens few future
completions over one that blackens many. A lottery cannot learn the geometry; it can only sample the
maximal caps random arrival happens to reach, and that set ceilings below the optimum once the
optimum requires real structure. To go further I have to *bias* the order intelligently, scoring each
candidate by a structured priority that reflects the symmetry of `F_3^n` — the move to
priority-function greedy, and the next step.
