I want a large cap set in `F_3^n`, and the first thing to settle is the simplest constructive idea
that is guaranteed to produce a *valid* cap, so I have a floor to stand on before I get clever. The
defining constraint is local in a useful way. A set fails to be a cap exactly when three distinct
points lie on a line, and over `F_3` a line is just `{a, b, −a−b}`: given two distinct points `a`
and `b`, there is exactly one third point `r = (−a−b) mod 3` that completes a forbidden triple,
because `a + b + c ≡ 0` has the unique solution `c = −a − b`. That uniqueness is the whole handle.
The forbidden structure is not a tangled global property I have to check across all triples at once;
it is a collection of "if you have these two, you may not also have that one" constraints, one per
pair — each unordered pair sits in exactly one line, and every point lies in `(3^n − 1)/2` of them.
The density is high, but the shape is simple, and simple-shaped constraints invite an incremental
rule.

The rule is: walk the points of `F_3^n` in some fixed order, and admit a point only if it is not
the completion `r = −a − b` of a line through two already-admitted points `a, b`. The set I end
with is a cap by construction — every admitted point was checked against all predecessors at the
moment it was added, and a violation needs three points, the last of which would have been
rejected. Validity is not earned after the fact; the construction and its proof of correctness are
the same object. The only thing left in play is *which order* I walk, so the entire quality of a
greedy cap is a property of its ordering and nothing else. Maintaining the block set incrementally,
each admitted point marks the completion of its line with every prior member, costs `O(c^2 · n)` to
build a cap of size `c`.

There are algebraic routes to a valid cap that need no search — lift a known small cap by products
and unions across coordinates, so size `s` in `F_3^k` becomes roughly `s^{n/k}` in `F_3^n`, which
is how many classical lower bounds are proved. But those commit me to a fixed recipe rather than
exposing a single tunable lever I can later refine. What I want from a floor is not the largest
guaranteed cap but the most *assumption-free* one, whose only moving part is the thing every later
attempt will attack — and greedy admission is exactly that: one skeleton with the order as its sole
degree of freedom.

Seen this way, greedy admission on a fixed order is an *online* decision problem, and that is the
root of its weakness. Each point arrives, I say yes or no on the spot, and the choice is
irrevocable. The true maximum cap is an *offline* object: finding it needs all `3^n` points on the
table at once, choosing the whole subset jointly. Any predetermined order fixes the arrival
sequence before a single line has been seen, so the size it attains just measures how lucky its
blind sequence happens to be. The leverage cannot come from cleverness *within* a fixed order — the
handicap is baked into committing to the sequence before looking at the geometry. It has to come
from elsewhere: seeing more of the space before each commitment, or trying enough different
sequences that one lands lucky.

The most obvious order, requiring no thought, is lexicographic: list the `3^n` vectors as if
counting in base three, `00…0, 00…1, 00…2, 00…10, …`, and greedily take each if it keeps the set
cap-valid. It is deterministic, parameter-free, seedless, trivially correct — and I suspect it is
rigid enough that I can predict its output exactly.

Trace `n = 2`. Counting order is `00, 01, 02, 10, 11, 12, 20, 21, 22`. Take `00`. Take `01`; it
blocks the third point `−(00)−(01) = 02`, so `02` is skipped. Take `10`; it blocks `20` and `22`.
Take `11`; it blocks `21` and `12`, and the line through `00` and `11` closes at `22`, already
blocked. Now `12, 20, 21, 22` are all blocked, and the walk ends with `{00, 01, 10, 11}`, size
`4 = 2^2` — precisely the vectors with no entry equal to `2`. The `2` was never admissible at any
coordinate once the two smaller symbols were present.

That is not a coincidence of `n = 2`; the lexicographic greedy cap is exactly `{0, 1}^n`, the `2^n`
vectors with no coordinate equal to `2`, for every `n`, and the proof tells me why the order is
weak. First, `{0, 1}^n` is a cap: if distinct `a, b, c` in `{0, 1}^n` summed to `0 mod 3`, then in
each coordinate three values from `{0, 1}` summing to `0` or `3` force all-zero or all-one, so
`a = b = c`, contradicting distinctness. Second, no `{0, 1}`-vector is ever blocked: the third
point of a line through distinct `a, b ∈ {0, 1}^n` is `r = −a − b`, and in any coordinate where
they differ `r_i = −(0 + 1) = 2`; since `a ≠ b` they differ somewhere, so `r` carries a `2` and
lies outside `{0, 1}^n`. Third, every vector *with* a `2` is rejected: if `v` has `2`s exactly on a
nonempty set `S`, build `a, b` agreeing with `v` off `S` and taking `(0, 1)` on `S`; then
`a, b ∈ {0, 1}^n` are distinct, `−a − b = v`, and both precede `v` in counting order (their first
difference from `v` is at the first coordinate of `S`, where they hold `0` or `1` against `v`'s
`2`), so both are admitted before the walk reaches `v`, and `v` is blocked. So the greedy cap is
`{0, 1}^n` on the nose, size exactly `2^n` — sizes `2, 4, 8, 16, 32, 64, 128` for `n = 1..7`,
settled by the argument, not by faith in the harness.

The proof also exposes the *character* of the failure. `{0, 1}^n` is not just small; it is a
*maximal* cap — every vector outside carries a `2`, and every such vector completes a line through
two of its members, so nothing can be appended. Greedy lexicographic does not stop short of a
locally improvable set; it runs all the way into a dead end and stays there. The `n = 3` case
sharpens the sting: `{0, 1}^3` has size `8`, is maximal, yet the true maximum in `F_3^3` is `9`.
The gap is not that greedy quit early — it walked into the *wrong* maximal cap. The one-point-better
cap lives in a different maximal configuration this order can never reach, because reaching it would
require admitting some `2`-vector early, before the `{0, 1}` cluster condemns it, and the counting
order never offers a `2`-vector before all the `{0, 1}`-vectors that condemn it.

There is a floor beneath even this, worth computing because it separates what is structural in the
`2^n` from what a better order could recover. Any greedy cap ends maximal: every non-member is
blocked. A blocked point is the completion of at least one pair in the cap, and each pair completes
exactly one point, so the `C(c, 2)` pair-completions must cover all `3^n − c` non-members:
`c(c − 1)/2 ≥ 3^n − c`, hence `c ≳ √2 · 3^{n/2} ≈ 1.73^n`. No order can leave a greedy-maximal cap
much below `1.73^n`, so lexicographic's `2^n` is not a disaster. But this bounds *every* maximal cap
and says nothing about which one an order lands in — the gap between the worst maximal cap and the
best is exactly the structural question the counting cannot see and a smarter order must answer.

And the pathology is shared by every fixed order: colex, a reversed sweep, a Gray-code walk each run
greedy into one maximal cap with the identical blindness, because no fixed order consults the line
structure while it walks. Which maximal cap it lands in is a matter of how the enumeration happens
to align with the geometry, and no principled reason makes any particular sweep aligned with a good
one. Lexicographic is not uniquely bad; it is *representatively* bad, and its transparency — I can
prove it gives `{0, 1}^n` — is what makes it the honest floor.

Now the magnitude, because it decides what to do next. Against the optima `2, 4, 9, 20, 45, 112,
236`, my `2^n` matches only at `n = 1, 2` and falls short after by `1, 4, 13, 48, 108`. The absolute
gap widens fast because of a growth-rate mismatch: the optima grow by roughly `2.1`–`2.5` per
dimension (`9/4 = 2.25`, `20/9 ≈ 2.22`, `112/45 ≈ 2.49`) while `2^n` grows by exactly `2`, and a
ratio even slightly above `2` compounds. In fraction terms `{0, 1}^n` captures `80%` at `n = 4`,
`71%` at `n = 5`, `57%` at `n = 6`, `54%` at `n = 7` — an exponentially shrinking share, not a
constant factor. The cap-set theorem's ceiling `|cap| ≤ O(2.756^n)` sits far higher — at `n = 8` a
few thousand — so it certifies large caps are not forbidden, not that they are reachable; the best
known *construction* there is `496`, itself only about twice the `256` this floor gives. Every point
of the deficit is attributable to the geometry-blind order — a smarter order buys size, never
correctness.

So the value here is to pin the floor and localize the entire deficit on the one lever the skeleton
exposes and refuses to pull. If a fixed order is the whole weakness, the cheapest attack is to stop
trusting any single order — try many and keep the best cap any produces. That is the first move up.
