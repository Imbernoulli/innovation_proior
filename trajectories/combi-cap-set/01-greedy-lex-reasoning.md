I want a large cap set in `F_3^n`, and the very first thing I should settle is the simplest
constructive idea that is guaranteed to produce a *valid* cap, so I have a floor to stand on
before I get clever. The defining constraint is local in a useful way. A set fails to be a cap
exactly when some three distinct points lie on a line, and over `F_3` a line is just `{a, b,
−a−b}`: given any two distinct points `a` and `b`, there is exactly one third point `r = (−a−b)
mod 3` that would complete a forbidden triple, because `a + b + c ≡ 0` has the unique solution `c
= −a − b`. That uniqueness is the whole handle. It means the forbidden structure is not some
tangled global property I have to check by looking at all triples at once; it is a collection of
"if you have these two, you may not also have that one" constraints, one per pair. And there are a
lot of them but a countable, structured lot: each unordered pair sits in exactly one line, each
line carries three pairs, so the number of lines is `C(3^n, 2)/3 = 3^n(3^n − 1)/6`, and every
single point lies in `(3^n − 1)/2` of them. For `n = 2` that is `12` lines with each of the `9`
points on `4` of them; for `n = 4` it is `1080` lines. The density of constraints is high, but
their shape is simple, and simple-shaped constraints invite an incremental rule.

The rule that the uniqueness hands me is this: walk the points of `F_3^n` in some fixed order, and
admit a point only if it is not the completion `r = −a − b` of any line through two points `a, b`
I have already admitted. Whatever order I use, the set I end with is a cap by construction,
because every admitted point was checked against all its predecessors at the moment it was added,
and a violation needs three points, the last of which would have been rejected. So validity is not
something I earn or verify after the fact — it is free, guaranteed by the admission test itself.
The only thing left in play is *which order* I walk, and how good the cap that order produces
turns out to be. That reframing is the useful one: the entire quality of a greedy cap is a
property of its ordering, and nothing else. Bad caps and good caps out of this skeleton differ
only in the sequence they were offered points.

Before I settle on greedy admission as the floor, I should acknowledge the other guaranteed-valid
constructions, because part of calling something a floor is knowing what it is a floor *relative
to*. There are algebraic routes to a valid cap that need no search at all: take a known small cap
and lift it by products and unions across coordinates, so a cap of size `s` in `F_3^k` becomes a
cap of size roughly `s^{n/k}` in `F_3^n`. Those product constructions are how many classical lower
bounds are proved, and they can beat `2^n` — but they are rigid in a different way, committing me
to a fixed algebraic recipe rather than exposing a single tunable lever I can later refine. What I
want from a floor is not the largest guaranteed cap; it is the most *assumption-free* one, the one
whose only moving part is the thing every later rung will attack. Greedy admission is exactly that:
one skeleton — offer points in some order, admit each if it closes no line, block the completions —
with the order as its sole degree of freedom. Everything above this rung will be the same skeleton
with a better order, so making the floor the skeleton itself, run on the most naive order, is the
choice that keeps the ladder honest, because it holds everything fixed except the one variable
whose effect I am trying to isolate. The cost is modest and worth stating: maintaining the block
set incrementally, each admitted point marks the completion of its line with every prior member, so
building a cap of size `c` costs `O(c^2 · n)` — the very `O(|cap|^2 n)` bookkeeping the harness
verifier uses, run forward as a constructor instead of backward as a check. Because the admission
test already guarantees validity, there is nothing to verify afterward; the construction and its
proof of correctness are the same object.

Seen this way, greedy admission on a fixed order is an *online* decision problem, and that is the
root of its weakness. Each point arrives, I must say yes or no on the spot, and the choice is
irrevocable — I never revisit an admitted point or reconsider a rejected one. The true maximum cap,
by contrast, is an *offline* object: to find it I would need all `3^n` points on the table at once,
choosing the whole subset jointly against all the line constraints together. Any online rule with a
predetermined order is strictly weaker than that offline optimum, because the order fixes the
arrival sequence before a single line has been seen, and greedy's only response to each arrival is
the myopic "does it fit right now." The size a fixed order attains is therefore a measure of how
lucky its blind arrival sequence happens to be, nothing more. That is why I do not expect cleverness
*within* a fixed order to rescue much: the handicap is structural, baked into committing to the
sequence before looking at the geometry. Whatever leverage exists has to come from somewhere else —
either from seeing more of the space before each commitment, or from trying enough different
sequences that one of them lands lucky.

The most obvious order, the one requiring no thought at all, is lexicographic: list the `3^n`
vectors as if counting in base three, `00…0, 00…1, 00…2, 00…10, …`, and greedily take each one if
it keeps the set cap-valid. This is the natural baseline because it is deterministic,
parameter-free, seedless, and trivially correct. Before I run it I want to reason about what it
should give, because I suspect the answer is rigid enough that I can predict it exactly, and a
prediction I can prove is worth far more than a number I merely read off.

Start on the last coordinate, `n = 1`. Lexicographic takes `0` and `1` immediately — two points,
no line yet. The third point `2` would complete the line `{0, 1, 2}`, whose sum is `0 + 1 + 2 = 3
≡ 0`, so it is rejected. The `n = 1` cap is `{0, 1}`, size `2`, which is the optimum there. Now
lift to `n = 2` and actually trace it, because the trace is where the pattern reveals itself.
Counting order is `00, 01, 02, 10, 11, 12, 20, 21, 22`. Take `00`. Take `01`; it blocks the third
point of the line through `00` and `01`, which is `−(00) − (01) = 02`. So `02` is blocked and gets
skipped. Take `10`; it blocks `−(00) − (10) = 20` and `−(01) − (10) = 22`. Take `11`; it blocks
`−(01) − (11) = 21` and `−(10) − (11) = 12` (the line through `00` and `11` closes at `22`, already
blocked). Now `12, 20, 21, 22` are all blocked, and the walk ends with `{00, 01, 10, 11}`, size
`4 = 2^2`. Look at what that set *is*: it is precisely the vectors whose every entry lies in `{0,
1}`. The `2` was never admissible at any coordinate once the two smaller symbols were present.

That is not a coincidence of `n = 2`; I can prove it holds for every `n`, and the proof tells me
exactly why lexicographic greedy is weak. Claim: the lexicographic greedy cap is exactly `{0,
1}^n`, the `2^n` vectors with no coordinate equal to `2`. Three things have to be checked. First,
`{0, 1}^n` really is a cap: if three distinct points `a, b, c` in `{0, 1}^n` had `a + b + c ≡ 0`,
then in each coordinate the three entries lie in `{0, 1}` and sum to `0 mod 3`; the only ways three
values from `{0, 1}` sum to `0` or `3` are all-zero or all-one, so `a_i = b_i = c_i` in every
coordinate, forcing `a = b = c`, contradicting distinctness. So `{0, 1}^n` contains no line.
Second, no `{0, 1}`-vector is ever blocked by the walk: the third point of a line through two
distinct `{0, 1}`-vectors `a, b` is `r = −a − b`, and in any coordinate where `a_i = b_i` (both `0`
or both `1`) we get `r_i = −2a_i ∈ {0, 1}`, while in any coordinate where they differ we get `r_i =
−(0 + 1) = 2`; since `a ≠ b` they differ somewhere, so `r` carries a `2` and lies *outside* `{0,
1}^n`. Hence no line internal to `{0, 1}^n` ever closes on a `{0, 1}`-vector, so every one of them
survives to be admitted. Third, every vector *with* a `2` is rejected: if `v` has `2`s exactly on
a nonempty coordinate set `S`, build `a` and `b` agreeing with `v` off `S` and taking `(a_i, b_i) =
(0, 1)` on `S`; then `a, b ∈ {0, 1}^n`, they are distinct (they differ on `S`), and `−a − b = v`.
Both `a` and `b` are lexicographically smaller than `v` (their first difference from `v` is the
first coordinate of `S`, where they hold `0` or `1` against `v`'s `2`), so both are already admitted
by the time the walk reaches `v`, and `v` is blocked. Every vector with a `2` falls to a line
through two earlier `{0, 1}`-vectors. So the greedy cap is `{0, 1}^n` on the nose, and its size is
exactly `2^n`. I do not even need to run the constructor to know the sizes it will report: `2, 4,
8, 16, 32, 64, 128` for `n = 1..7`. I will run it anyway, to confirm and to read the gap to the
optima precisely — but the sizes are settled by the argument, not by faith in the harness.

To make the condemnation concrete, take `v = (0, 2, 1)` at `n = 3`. Its lone `2` sits in the middle
coordinate, so `S = {1}`, and the recipe builds `a = (0, 0, 1)` and `b = (0, 1, 1)`, agreeing with
`v` off `S` and splitting `{0, 1}` across `S`. Their line closes at `−a − b = (0, −1, −2) = (0, 2,
1) = v`, and both `a` and `b` are `{0, 1}`-vectors that precede `v` in counting order — their first
disagreement with `v` is the middle coordinate, where they hold `0` and `1` against `v`'s `2`. So
by the time the walk reaches `(0, 2, 1)`, both witnesses are already in the cap and the point is
blocked. The identical two-witness construction condemns every `2`-carrying vector, one after
another, which is exactly why the greedy fill can never step out of the `{0, 1}` corner no matter
how far the walk continues.

The proof also exposes the character of the failure, which is the real point of this rung. `{0,
1}^n` is not just small; it is a *maximal* cap in the local sense — I cannot add a single point to
it. Every vector outside it carries a `2`, and I just showed every such vector is the completion of
a line through two of its members, so appending any of them creates a forbidden triple. Greedy
lexicographic therefore does not stop short of a locally improvable set; it runs all the way into a
dead end and stays there. It is a local optimum of the greedy landscape, and a bad one. The `n = 3`
case sharpens the sting: `{0, 1}^3` has size `8`, it is maximal (nothing can be appended), yet the
true maximum in `F_3^3` is `9`. The gap is not that greedy quit early; it is that greedy walked
into the wrong maximal cap entirely. A single point better exists, but it lives in a *different*
maximal configuration that this order can never reach, because reaching it would have required
admitting some vector with a `2` early, before the `{0, 1}` cluster blocked it — and the counting
order never offers a `2`-vector before all the `{0, 1}`-vectors that condemn it.

There is a floor beneath even this floor, worth computing because it separates what is structural
in the `2^n` from what a better order could recover. Any greedy cap, whatever the order, ends
*maximal*: every non-member is blocked, or greedy would have taken it. A blocked point is the
completion of at least one pair already in the cap, and each pair completes exactly one point, so
the `C(c, 2)` pair-completions must cover all `3^n − c` non-members: `c(c − 1)/2 ≥ 3^n − c`, hence
`c ≳ √2 · 3^{n/2} ≈ 1.73^n`. No order — lexicographic, hand-tuned, or random — can leave a
greedy-maximal cap much below `1.73^n`; that much size is forced by the pair-covering arithmetic
alone. Lexicographic's `2^n` sits above this universal minimum (at `n = 3` the inequality already
forces `c ≥ 7`, and `{0, 1}^3` clears it at `8`), which is why the floor is not a disaster. But the
arithmetic is far too crude to separate `2^n` from the optima's `~2.2^n`: it bounds *every* maximal
cap from below and says nothing about which one an order lands in. That gap, between the worst
maximal cap and the best, is precisely the structural question the counting cannot see and a
smarter order must answer. So the deficit is not that greedy stops too early — it stops at a
genuinely maximal set, comfortably above the covering bound — it is that greedy stops at the
*wrong* maximal set.

I can make "the wrong maximal set" concrete rather than leave it a slogan. At `n = 3` the walk
locks onto `{0, 1}^3`, eight points, and a direct scan of the nineteen vectors outside it — every
vector carrying at least one `2` — confirms it is sealed: not one of them can be appended without
closing a line, because each is the completion `−a − b` of some pair inside `{0, 1}^3`. Zero
admissible extensions remain, yet a nine-point cap exists in `F_3^3`. So the eight-point set is not
almost-complete-and-stuck-just-shy-of-the-answer; it is a fully-sealed pocket that happens to be
one point smaller than a *different* sealed configuration the order could never enter. That is the
texture of the deficit at every `n`: greedy runs to a hard wall, and the wall is in the wrong
place. The dynamics that build the wall are worth watching. When the cap holds `k` points,
admitting the next one closes a line with each of the `k` existing members, so it can blacken up to
`k` fresh completions in a single step; the admissible frontier therefore erodes faster and faster
as the cap grows, and the walk suffocates the moment every remaining vector has been blackened.
Lexicographic makes that suffocation early and local by clustering its first admissions at the
low-index corner, so the erosion concentrates exactly where it is about to step next — the reason it
seals at `2^n` rather than groping further into the space. A good order would have to do the
opposite: scatter its early admissions so their blocking is diffuse, keeping the frontier alive
longer and reaching a larger sealed configuration before it suffocates.

Why does lexicographic order commit to this dead end? Because it commits early and locally, with no
notion of the geometry. The first points it grabs are all clustered at the low-index end of the
space — `00…0`, then its immediate counting-successors — and the lines those clustered points induce
are concentrated in exactly the region the walk is about to traverse next, so the blocking is dense
and self-reinforcing right where greedy is working. There is no reason the base-three counting order
should be aligned with the line structure of `F_3^n`; it is an artifact of how I happen to enumerate
tuples, nothing more. The rule cannot look ahead to see that admitting a particular `{0, 1}`-vector
now will block a `2`-vector that a stronger cap would have wanted; it cannot reconsider a past
admission; it cannot prefer a point that closes few future lines over one that closes many. It takes
whatever the arbitrary order hands it next and lives with the consequences. A cap that reaches the
optimum has to *place* its points so that the lines they generate fall outside the cap as
efficiently as possible, spreading the blocking thin and packing the space tight — and a fixed
lexicographic walk has no mechanism to do any of that.

It is tempting to think the fix is just a *better* fixed order — that counting is the villain and
some other deterministic sweep would do better. But the pathology is not special to lexicographic;
it is shared by every single fixed order. Colexicographic, a reversed sweep, a Gray-code walk —
each is a fixed sequence, each runs greedy into exactly one maximal cap, and each has the identical
blindness, because no fixed order consults the line structure while it walks. Which maximal cap it
lands in is a matter of luck in how the enumeration happens to align with the geometry: a different
order lands in a different maximal cap, possibly a little larger, possibly smaller, and there is no
principled reason any particular fixed sweep is aligned with a *good* one. Lexicographic is not
uniquely bad; it is representatively bad, and its transparency — I can prove it gives `{0, 1}^n` —
is exactly what makes it the honest floor. The lesson generalizes past it: a single deterministic
order, however chosen, is one uncontrolled draw from the space of maximal caps, with no lever on
which draw I get. That already reframes the next move — if any one order is a single uncontrolled
draw, the cheapest possible improvement is to take many draws and keep the best, to stop trusting
any single order at all.

Now let me size the deficit against the yardsticks I have, because the magnitude matters for
choosing what to do next. The known optima through `n = 7` are `2, 4, 9, 20, 45, 112, 236`. My
predicted `2^n` matches only at `n = 1, 2`, where `2^n` *is* the optimum, and falls progressively
short after: `8` vs `9` (a deficit of `1`), `16` vs `20` (`4`), `32` vs `45` (`13`), `64` vs `112`
(`48`), `128` vs `236` (`108`). The absolute gap widens fast, and the reason is a growth-rate
mismatch I can read off directly. The optima grow by roughly a factor of `2.1` to `2.5` per
dimension (`9/4 = 2.25`, `20/9 ≈ 2.22`, `45/20 = 2.25`, `112/45 ≈ 2.49`, `236/112 ≈ 2.11`), while
`2^n` grows by exactly `2`. A ratio that is even slightly above `2` compounds into an
ever-widening multiplicative gap, and the cap-set theorem's ceiling of `O(2.756^n)` sits higher
still. In density terms `{0, 1}^n` occupies a `(2/3)^n` fraction of the space, decaying faster than
the true caps, whose density decays only like `(2.5/3)^n` or slower — so the `2^n` floor is not a
constant factor below the optima but an exponentially shrinking fraction of them, capturing `80%`
of the optimum at `n = 4`, `71%` at `n = 5`, `57%` at `n = 6`, and only `54%` at `n = 7`. The
weakness compounds with dimension, which is exactly the regime I care about.

It helps to hold the two ceilings apart. The cap-set theorem proves only an *upper* bound, `|cap|
≤ O(2.756^n)` — at `n = 8` that is on the order of a few thousand, wildly above anything anyone
constructs — so it certifies that large caps are not forbidden, not that they are reachable. The
*achieved* frontier is far lower: the best known construction at `n = 8` is `496`, itself only
about twice the `256` my floor would give there. My `2^n` is thus squeezed between a provable
ceiling it cannot come near and an achieved frontier it falls short of by a compounding factor.
Reading the deficit as `optimum − 2^n` at each `n` — `1, 4, 13, 48, 108` for `n = 3..7` — even the
*increments* grow, and every one of them is a point that only a better order can buy back. Nothing
about validity is at stake in any of this; a smarter order buys size, never correctness.

So the value of this rung is to pin the floor and to localize the entire deficit precisely on the
one lever it exposes and refuses to pull: the ordering. Validity is free and will never be in
question here; the construction is correct by design, and I have proved what it builds. Every point
of the gap to the optimum is therefore attributable to nothing but the geometry-blind fixed order —
a rigid walk that marches into the maximal-but-suboptimal `{0, 1}^n` and cannot escape. If a fixed
order is the whole weakness, the cheapest possible way to attack it is to stop trusting any single
order: try *many* orders and keep the best cap any of them produces. That is the first move up from
the floor, and it is where I go next.
