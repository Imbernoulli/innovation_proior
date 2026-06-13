# Context

## Research question

Is every infinite set the same size? Concretely: can the positive integers
1, 2, 3, … be put into one-to-one correspondence with the positive real numbers
(or with the reals in an interval [a, b])? The only honest meaning of "same size"
for infinite collections is the existence of a **one-to-one correspondence**
(a bijection) between them — the criterion that already governs finite sets and
that does not secretly presuppose a count. By that criterion, several infinite sets
that "look" much bigger than the integers turn out to be enumerable: the positive
rationals can be arranged in a single list, and so can indexed families of the form
(a_{n₁,n₂,…,n_ν}). The pressing question, posed by Cantor to Dedekind on 29 November
1873, is whether there exists **any** infinite set that *cannot* be listed at all.

This matters on two fronts. If every infinite set were enumerable, the notion of
"size" would collapse — there would be exactly one infinity, and the word
"uncountable" would name nothing. And there is a concrete payoff lurking: Liouville
(1851) had shown by explicit construction that transcendental numbers exist. If one
could show the real algebraic numbers are enumerable while the reals are not, the
existence of transcendentals would fall out instantly from a counting mismatch,
with none of Liouville's delicate approximation estimates. A solution must therefore
deliver a *criterion-respecting* proof — built from bijections and lists, not from
appeals to intuition about "how many" there are.

## Background

The load-bearing concept is the **one-to-one correspondence** as the test for equal
power (Cantor's term is *Mächtigkeit*). Two sets have the same power when their
elements can be paired off exactly; an infinite set is *enumerable* (countable) when
it has the same power as the positive integers, i.e. can be written as a single
sequence x₁, x₂, x₃, … in which every element occurs.

The prevailing evidence in 1873 pointed *toward* universal enumerability, which is
what makes the question live and the eventual answer surprising. The positive
rationals — dense, seemingly "thicker" than the integers — are enumerable. Dedekind
had sent Cantor a proof that the **algebraic numbers** (roots of integer
polynomials), denser still and containing all the rationals plus √2, the golden
ratio, and so on, are *also* enumerable. So density is no obstacle to listability,
and one might reasonably conjecture that every reasonable infinite set of numbers can
be listed. Establishing the *opposite* for the continuum would be the first
demonstration that infinite sets come in genuinely different sizes.

Two further pieces of mathematical furniture are in the room. First, **completeness
of the real line**: a bounded monotone sequence of reals converges to a limit. In
Dedekind's hands this is his "principle of continuity," equivalent to the least-upper-
bound property and flowing from his construction of the reals by cuts. Any argument
that pins down a real number as a *limit* of an approximating process will rest on
this. Second, the surrounding climate: Kronecker, an editor at Crelle's Journal,
distrusted completed infinite sets and arguments resting on Dedekind's construction
of ℝ; Weierstrass was sympathetic to the countability result but uneasy about the
claim that two infinite sets could differ so radically that one is listable and the
other not. These attitudes shape *how* a result about the infinite has to be stated
and defended, not just whether it is true.

There is also an established existence result to react against: **Liouville (1851)**
exhibited transcendental numbers explicitly, as reals approximable by rationals to
unusually high order (e.g. Σ 10^{−k!}). His method is constructive but laborious and
specific. A counting-based proof of the existence of transcendentals would be a
qualitatively different, far cheaper route — provided the requisite counting facts
(algebraics enumerable, reals not) can be had.

## Baselines

**One-to-one correspondence / countability (Cantor's own prior framing, 1873).**
The criterion for equal size is a bijection; "countable" means same power as the
integers. Cantor had already placed the rationals and indexed families
(a_{n₁,…,n_ν}) in lists. Core idea: bypass any notion of "how many" and pair elements
directly. Gap it leaves open: it gives a *positive* technique — to show a set is
countable, exhibit the list — but no technique at all for the *negative* direction,
showing that *no* list can exist. Every set anyone had examined turned out countable,
so the machinery had only ever been used to confirm enumerability.

**Dedekind's countability of the algebraic numbers (1873).** Order integer
polynomials by a magnitude that has only finitely many polynomials below each
threshold, then list their roots; every algebraic number appears. Core idea: a
"height" that slices an infinite set into finite layers, listed layer by layer. This
is the constructive template for *listing* a rich set of numbers. Gap: it sharpens the
worry rather than resolving it — if even the algebraic numbers list, perhaps the reals
do too; it offers no leverage on whether some set escapes listing.

**Liouville's transcendentals (1851).** Numbers approximable by rationals beyond the
rate any algebraic number permits are transcendental; explicit such numbers exist.
Core idea: a quantitative approximation barrier separates algebraic from
transcendental. Gap: it constructs *particular* transcendentals with effort and gives
no global statement about the *size* of the set of reals versus the algebraics; it
cannot tell you whether "most" reals are transcendental, only that some are.

**Nested-interval / monotone-limit reasoning (standard analysis, via Dedekind's
continuity).** A decreasing sequence of closed intervals with endpoints converging
pins down a point. Core idea: trap a number between an increasing lower sequence and a
decreasing upper sequence and read off the common limit. This is the standard way to
*produce* a real number with prescribed avoidance properties. Gap as a size tool: it
is thoroughly wedded to the real line — it uses order, intervals, and the completeness
of ℝ — so even when it succeeds it says nothing about sets that are not collections of
real numbers, and it smuggles a deep analytic property (completeness) into what ought
to be a bare counting question.

## Evaluation settings

The objects of study are: the set ℕ = {1, 2, 3, …}; the rationals; the real algebraic
numbers; the reals in a fixed interval [a, b] (and the continuum [0, 1]). The yardstick
is purely logical: a claim of equal size is
discharged by exhibiting a bijection, a claim of "not the same size" by exhibiting,
for *every* candidate correspondence, a witness it misses. There are no datasets,
metrics, or measurements — the standard of success is a proof valid for an *arbitrary*
purported enumeration, and (for the existence-of-transcendentals corollary)
agreement with Liouville's already-known conclusion that transcendentals exist in
every interval.

## Code framework

The "code" here is the formal scaffold: the definitions and facts already available,
the open listing questions, and the empty slots where a counterexample-producing
construction would have to go.

```text
# Definitions:
#   X countable  :iff  exists bijection X <-> N  :iff  X listable as x_1, x_2, x_3, ...
#   same power(X, Y)   :iff  exists bijection X <-> Y
#   f : X -> Y surjective  :iff  every y in Y is f(x) for some x in X

# Established facts:
#   rationals countable;  (Dedekind) real algebraic numbers countable;
#   (completeness of R) bounded monotone real sequence converges.

# Interval listing question:
proposition INTERVAL_LISTING_QUESTION:
    "decide whether the reals in [a, b] can be written as a list x_1, x_2, x_3, ..."
    # TODO: prove or refute

# Missing-point slot:
def missing_point(x_1, x_2, x_3, ..., interval I):
    # given ANY purported enumeration of the objects of I,
    # return an object of I provably equal to none of the x_n
    # TODO: construct it
    pass
```
