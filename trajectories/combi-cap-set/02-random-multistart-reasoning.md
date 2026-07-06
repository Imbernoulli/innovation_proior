The previous rung told me precisely what was wrong, and the feedback confirmed it to the point: the
lexicographic walk returned exactly `2, 4, 8, 16, 32, 64, 128` for `n = 1..7`, `2^n` with no
deviation at any dimension, matching the optimum only where `2^n` *is* the optimum (`n = 1, 2`) and
falling short everywhere after by `1, 4, 13, 48, 108`. That is not a fuzzy tendency I have to
interpret; it is the rigid signature of a single mechanism. The walk builds `{0, 1}^n`, the set of
vectors that never use the symbol `2`, and it does so because the first low-index points it grabs
close lines whose third points are exactly the `2`-carrying vectors, sealing them out one by one.
The set it lands on is *maximal* — I checked at `n = 3` that not one of the nineteen outside vectors
can be appended without closing a line — so lexicographic does not stop short of an improvable
configuration; it runs all the way into a sealed pocket and stays there. Its deficit is entirely the
deficit of that one pocket being smaller than the best sealed configuration in the space. The order
is the only lever, and the specific way this order fails is by falling into a bad local optimum of
the greedy landscape.

It is worth reading the lexicographic deficit as fractions before deciding how much sampling can
recover, because the fraction tells me whether I am chasing a rounding error or a structural gap. As
a share of the optimum, `2^n` captures `16/20 = 80%` at `n = 4`, `32/45 ≈ 71%` at `n = 5`, `64/112 ≈
57%` at `n = 6`, and `128/236 ≈ 54%` at `n = 7` — the captured fraction decays steadily, so the
floor is not a fixed distance below the optimum but an ever-shrinking piece of it. The increments I
would have to recover are `+4, +13, +48, +108`, growing faster than the floor itself. At `n = 4`,
recovering `+4` to reach `20` is a `25%` lift over the floor — a modest ask, well within reach of
sampling. At `n = 7`, recovering `+108` to reach `236` would nearly double the floor, and doubling
by lottery is exactly what a thin tail forbids. So before running anything I already expect the
recoverable fraction of the deficit to shrink with `n`: sampling should reclaim most of a small gap
and little of a large one, which is a sharper prediction than "random will help."

Once I see the failure as "trapped in a bad local optimum," the family of possible responses is
clear, and I want to walk it rather than jump. One option is to make the order *smarter but still
fixed* — but I argued last rung that this is structurally hopeless: every fixed order is one blind
arrival sequence that lands in one maximal cap, with no lever on which one, so trading lexicographic
for colex or a Gray-code walk just trades one arbitrary pocket for another. A second option is
genuine *local search on caps*: build a maximal cap, then try to escape it by removing a few points
and re-growing, hill-climbing across the landscape of maximal caps. That can work, but it is a
heavier machine — it needs a neighborhood definition, an acceptance rule, a way to avoid cycling —
and it abandons the clean "one skeleton, one lever" discipline I want to preserve while I am still
mapping how much the order alone can buy. A third option is the cheapest imaginable: if a single
fixed order is one uncontrolled draw from the space of maximal caps, then draw *many* times and keep
the best. Exhausting all orders is out of the question — there are `(3^n)!` of them, hyperastronomical
even at `n = 4` — but I do not need all of them; I need enough independent draws that the best one is
good. That is randomized greedy with multi-start, and it costs me nothing conceptually: the
admission rule is identical, so every run still produces a valid cap by construction; I only shuffle
the order in which points are offered, and I take the largest cap over many shuffles.

I should say why I am not reaching for the heavier machine yet, since local search on caps is a real
and tempting option. Hill-climbing across maximal caps — remove a handful of points, regrow, accept
if the result is larger — could in principle climb past what any single greedy pass reaches, because
it can partially undo a bad commitment instead of living with it. But it introduces three new choices
I would then have to tune and defend: the neighborhood (how many points to drop, and which), the
acceptance rule (strictly-better only, or occasionally-worse to cross plateaus), and a guard against
cycling. Each of those is its own lever, and adding three levers while I am still measuring what the
*single* order-lever can buy would muddy exactly the accounting I care about. The ladder's value
comes from moving one lever at a time and reading its effect cleanly. Random multi-start keeps the
skeleton and the single lever intact — it only asks the order to be many orders instead of one — so
whatever it buys over lexicographic is attributable to sampling alone, with nothing else changed. If
sampling's ceiling turns out low, that is itself the clean result that would justify reaching for a
different kind of lever, and I would rather earn that conclusion than pre-empt it.

Let me reason about *why* a random order should beat lexicographic before I quantify by how much,
because the mechanism matters for predicting where it will ceiling. Lexicographic is pathological in
one specific way: it grabs a dense cluster of low-index points first, and the lines those clustered
points induce blacken a structured swath of the space concentrated right where the walk is about to
step next, so the admissible frontier erodes early and locally and the fill suffocates at `2^n`. A
uniformly random order has no such clustering. Its early admissions are scattered across `F_3^n`, so
the completions they blacken are scattered too — the blocking is diffuse rather than piled up in one
corner — and the frontier stays alive longer, letting the fill reach a larger sealed configuration
before it suffocates. Intuitively, a random order breaks the accidental alignment between the
enumeration and the blocking structure, so *on average* it should clear the `2^n` floor. But average
is not the operative word, and I need to be careful here: any single random order is just as
arbitrary as lexicographic — it is one draw, and one draw of a maximal cap is a lottery ticket. What
makes the method work is taking the *maximum* over many independent draws, reaching into the right
tail of the cap-size distribution rather than settling for its typical value.

There is a structural version of the same point that makes the escape sharper than "scatter reduces
variance." Lexicographic's `{0, 1}^n` uses only two of the three symbols — it never places a `2` —
and I showed last rung that every `2`-carrying vector is condemned by the `{0, 1}` cluster. But the
strong caps demonstrably *need* the third symbol: at `n = 3` the two-symbol corner caps out at `8`,
so the `9`-cap must spend at least one coordinate on a `2`. The optimal configurations therefore
live outside every two-symbol corner, and any fixed order that front-loads one corner is
structurally shut out of reaching them. A random order front-loads no corner — its early picks draw
all three symbols across scattered coordinates — so the caps it can reach are the full-alphabet
maximal caps, which is exactly where the large ones live. That is why scattering is not merely a
variance trick: it changes *which region of cap-space is reachable at all*, from the two-symbol
corners lexicographic is trapped in to the full-alphabet interior the optima inhabit. Multi-start is
then sampling within that better-reachable region, and the maximum over samples is what pulls the
reported size up its right tail.

How far into that tail does best-of-`k` reach, and what does the tail cost? If the sizes produced by
random greedy are roughly bell-shaped with mean `μ` and spread `σ`, then the maximum of `k`
independent draws sits about `σ·√(2 ln k)` above the mean — the standard extreme-value scaling. That
`√(ln k)` is the whole story of diminishing returns: going from `1000` to `5000` restarts raises the
expected best by only `σ·(√(2 ln 5000) − √(2 ln 1000)) = σ·(4.13 − 3.72) ≈ 0.41 σ`, so a fivefold
increase in compute buys less than half a standard deviation. If `σ` is a few points, thousands of
extra restarts buy roughly one extra point of cap size. This tells me two things at once. First,
multi-start genuinely helps — the first few thousand restarts move me well up the distribution from
the typical draw toward its upper tail. Second, it saturates fast: past a few thousand starts the
`√(ln k)` curve is nearly flat, and no realistic number of restarts will keep climbing. The method
has a built-in ceiling that is not about compute but about the *shape* of the distribution I am
sampling from.

And that shape is where the deeper ceiling lives. Best-of-`k` can only reach the top of the support
of the random-greedy size distribution — it cannot conjure a cap that greedy-with-a-random-order
essentially never produces. At small `n` the true optimum sits inside that support: the space is
tiny, the number of distinct greedy-reachable maximal caps is limited, and an optimal configuration
is a plausible random outcome. At `n = 4` the optimum `20` is only modestly above the `16` floor,
and `20`-caps are not exotic among random orders, so I expect multi-start to reach `20` and match
the proven optimum — the first rung to touch a known-optimal value, bought purely by sampling
orders. As `n` grows, though, the optimum pulls *out* of the support. The large caps are highly
structured algebraic objects, and random greedy has vanishing probability of stumbling into that
structure by chance, so the distribution of random-greedy sizes concentrates well below the optimum
and its upper tail ends short of it. When the optimum is outside the support, no amount of best-of-`k`
reaches it — the `√(ln k)` climb runs along a distribution whose ceiling is itself below the target.
So I expect a gap to the optimum that *grows* with `n`: comfortably clearing the `32` floor into the
high `30s` at `n = 5` but short of `45`; clearing `64` into the mid `70s` at `n = 6` but far short of
`112`; roughly reaching the `140s` at `n = 7` but far short of `236`.

I can place the target inside the size distribution using the bound from the last rung. Every
greedy-maximal cap, whatever the order, satisfies `c(c − 1)/2 ≥ 3^n − c`, so the support of
random-greedy sizes is floored at about `1.73^n`, while the optimum sits near `2.2^n` at the top. A
typical random draw lands low in that band — a scattered order beats the clustered `{0, 1}^n` corner
but is still an unstructured maximal cap — and best-of-`k` pushes the reported size up toward the top
of whatever the distribution actually reaches. The real question is whether the *top of the
greedy-reachable band* meets the optimum. At small `n` it does, because when the space is small
nearly every maximal configuration, the optimal one included, is greedy-reachable. As `n` grows the
optimal caps become special — they carry algebraic regularities a random arrival sequence has
vanishing odds of tracing — so the ceiling of the greedy-reachable band detaches from the true
optimum and sinks below it. Best-of-`k` then climbs a distribution whose own maximum is short of the
target, and no restart budget closes the residual gap. That is a cleaner statement of the ceiling
than "the tail is thin": the tail is thin *and* it ends in the wrong place, and the second half of
that sentence is the one that actually caps the method.

Before I commit the compute I should size it, because the restart budget is a real choice and I want
the numbers trustworthy. One restart scans the `3^n` vectors and, per admitted point, blocks a line
against every prior member, so its cost is on the order of `3^n + |cap|^2·n`. At `n = 4` that is
tiny — `81` plus about `20^2·4 ≈ 1600` — so thousands of restarts are nearly free, and I can afford
`5000`. By `n = 7` the cap is near `140`, so each restart costs about `140^2·7 ≈ 1.4×10^5`, and
`1000` restarts already run to `~1.4×10^8` operations; pushing to `5000` there would quadruple the
wall-clock for the `0.41 σ` the extreme-value curve says it buys. The rational schedule is therefore
to spend many restarts where they are cheap and dial them down as `|cap|^2·n` grows, holding the
total near a fixed budget: roughly `5000` starts at `n = 4, 5`, tapering toward `~1000` by `n = 7`.
I fix the random seed so each reported best-of-`k` is exactly reproducible, and I run every returned
cap through the verifier — and the independent triple scan at small `n` — before believing its size.
The multi-start changes which cap I report, not whether it is valid; validity is still guaranteed by
the admission rule, but I check anyway, because no reported size on this ladder is ever taken on
faith.

One more methodological caution, because the reported number is itself stochastic. Best-of-`k` under
a fixed seed is a single draw of an extreme-value statistic, so a rerun with a different seed would
wobble the reported size — by perhaps a point or two at the dimensions where the tail is soft.
Fixing the seed makes the result reproducible, which is what the harness needs, but it does not make
it canonical: the honest reading of a multi-start number is "the top of a `k`-sample at this seed,"
not "the value random greedy achieves." That matters for how I will compare rungs downstream. A
one-point difference between this baseline and some later deterministic order sits inside the seed
noise of best-of-`k` and should not be over-read; a gap of five or ten is real signal about which
method reaches a genuinely different region of cap-space. I want to hold that distinction in mind so
that when I later set a seedless, structured order against this sampled baseline, I am comparing
against a number whose noise I understand, rather than treating best-of-`5000` as a hard wall.

I can confirm the escape mechanism concretely at `n = 3`, below the scored dimensions but sharp
enough to trust the picture. Lexicographic there is sealed at `8` — the `{0, 1}^3` pocket, provably
unextendable. Running greedy on random orders instead, a few hundred restarts reach `9`, the proven
optimum, which lexicographic could never touch because reaching it demands admitting a `2`-carrying
vector early, before the `{0, 1}` cluster condemns it — precisely the move a scattered random order
makes and a clustered fixed order cannot. So at `n = 3` reordering does exactly what the mechanism
predicts: it steps out of the bad pocket into the best one. That is the small-`n` regime where the
optimum is inside the support, and it is the reason I expect `n = 4` to land on `20` rather than
merely above `16`.

I can turn that expectation into an estimate rather than a hope. If a fraction `p` of random orders
greedily produce a `20`-cap, then `k` independent restarts miss all of them with probability `(1 −
p)^k`, so the chance of hitting at least one is `1 − (1 − p)^k`. Even a fairly rare optimal order —
say `p ≈ 10^{-3}` — is almost certain to surface within `5000` restarts: `1 − (0.999)^{5000} ≈ 1 −
e^{-5} ≈ 0.993`. So as long as `20`-caps occupy even a thin sliver of the greedy-reachable
distribution at `n = 4`, thousands of restarts will find one with near-certainty, and the seed I fix
just picks *which* of the many successes I report. The reason this same argument would fail at `n =
7` is not the restart count but `p` itself: there the fraction of random orders reaching the optimum
is not merely small but effectively zero, because the optimum lies outside the reachable band, and
no `k` rescues a `p` that is zero. That asymmetry — `p` bounded away from zero at `n = 4`, `p`
collapsing to zero by `n = 7` — is the quantitative heart of why the gap grows.

To make the rung falsifiable in the metrics I actually score: if the mechanism is right — scatter
escapes the two-symbol corner, but sampling ceilings at the top of the greedy-reachable band, which
detaches from the optimum as `n` grows — then `n = 4` should land *exactly* on `20`, not merely
above `16`, because the optimum is inside the reachable band there. `n = 5, 6, 7` should each clear
their `2^n` floors of `32, 64, 128` decisively but land strictly below the optima `45, 112, 236`,
with the shortfall as a fraction of the optimum growing monotonically from `n = 5` to `n = 7`. And
`n = 8` I expect to leave essentially untouched by this method: at `3^8 = 6561` vectors with caps
near five hundred, each restart costs on the order of `500^2·8 ≈ 2×10^6`, and the extreme-value curve
says the tail there is both thin and low, so a restart budget spent on it would buy little I could
trust. If instead `n = 5, 6, 7` came in *at* their optima, or `n = 4` fell short of `20`, the
"escape-the-corner-but-ceiling-below-structure" picture would be wrong and I would have to rethink
what sampling reaches. I do not expect that; I expect the growing shortfall, and it is the growing
shortfall that hands the next rung its job.

If the numbers come out as I expect — a clean win over lexicographic at every `n`, the exact optimum
at `n = 4`, and a gap that widens after — the lesson I am setting up is exactly the growing gap.
Random multi-start fixes the *bias* of a fixed order: it no longer commits to one accidental arrival
sequence. But it does nothing about the *blindness*. Every order it tries is still uniform noise,
with no preference for points that sit in symmetric or structured positions, no preference for a
point that blackens few future completions over one that blackens many. It is a lottery over orders,
and a lottery cannot learn the geometry — it can only sample the maximal caps that random arrival
happens to reach, and that set ceilings below the optimum once the optimum requires real structure.
To go further I cannot simply buy more tickets; the `√(ln k)` tail is too thin and its support ends
too low. I have to *bias* the order intelligently — score each candidate by some structured priority
that reflects the symmetry of `F_3^n` and feeds the same greedy admission rule a smarter sequence
than noise. That is the move from "random order" to "priority-function greedy," and it is the next
rung.
