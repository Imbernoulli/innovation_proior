Random multi-start did what I predicted and the feedback drew the ceiling cleanly. It cleared the
lexicographic floor at every dimension — `20, 39, 77, 147` against the floor's `16, 32, 64, 128`, a
lift of `+4, +7, +13, +19` — and it reached the proven optimum `20` at `n = 4`, the first rung to
touch a known-optimal value, exactly the "optimum is inside the reachable band at small `n`" story.
But the gap to the optimum grows on schedule: as a fraction of the optimum the sampled best captures
`20/20 = 100%` at `n = 4`, `39/45 ≈ 87%` at `n = 5`, `77/112 ≈ 69%` at `n = 6`, and `147/236 ≈ 62%`
at `n = 7`. The captured fraction slides with dimension, which is the signature of a method whose
reach detaches from the target as the target becomes structured. Sampling removed the *bias* of a
fixed order but not its *blindness*: every order it tried was uniform noise, ceilinged at the top of
the greedy-reachable band, and that band's ceiling sinks below the true optimum once the optimum
demands algebraic regularity a random arrival sequence essentially never traces. The tail is thin,
and worse, it ends in the wrong place.

So the move is to stop sampling and start *biasing*. Instead of drawing orders at random, I score
every vector by a deterministic function of its coordinates and feed the same greedy admission rule
the vectors in highest-priority-first order. The admission rule is unchanged, so validity stays free;
only the ordering function changes. What is different in kind from the earlier rungs is the bet I am
making. Lexicographic was one blind fixed order; random multi-start was many blind orders; this is
*one order again* — a single deterministic sequence, seedless, like lexicographic — but a
geometry-aware one. The whole wager is that a single well-chosen structured order can beat the best
of thousands of blind ones, because it points greedy at the points that pack well rather than
letting it stumble. That is a real gamble, and I want to be honest that its outcome is not obvious:
one structured order is still one order, and if my structure is the wrong structure, one shot loses
to best-of-thousands.

The design question is which structure of `F_3^n` to reward, and the first instinct is a trap worth
walking into so I can kill it precisely. The line condition `a + b + c ≡ 0` is a statement about
*coordinate sums mod 3*, so the tempting move is to fix the total sum: restrict, or heavily prefer,
vectors whose coordinate sum lies in one residue class mod 3, hoping that pins down a clean
sublattice. But the total sum is a *linear* functional, and that is exactly what dooms it. A
sum-fixed set is an affine coset, and if `a` and `b` both have sum `≡ r`, their line's third point
`−a − b` has sum `−2r ≡ r mod 3` as well — so the coset is *closed* under line completion. I can
check it at `n = 4`: the twenty-seven vectors with coordinate sum `≡ 0` form a coset in which every
single pair's third point stays inside, so it is not merely non-cap, it is saturated with lines,
about the worst possible thing to build a cap from. The linear invariant is the one structure I must
*not* fix. That failure is instructive: it tells me to reach for a *non-linear* feature of the
coordinates instead, one that is not preserved by the line operation and so does not automatically
gather lines.

The natural non-linear feature is the Hamming weight — the number of nonzero coordinates — taken mod
3. It is genuinely non-linear (the weight of `−a − b` bears no additive relation to the weights of
`a` and `b`), so a weight-mod-3 layer is not a coset, and it carries no automatic guarantee of being
line-saturated the way the sum-coset was. That is the whole reason to prefer it over the sum: it is
at least *not obviously wrong*. But I should not overreach and *restrict* the greedy to a single
weight layer, because a layer holds on the order of `3^{n-1}` vectors — far more than any cap — so it
is still full of lines and cannot be a cap itself. What I can do is *softly bias* toward a coherent
weight layer rather than a mix, on the heuristic that a cap whose points share a weight profile might
have more regular line structure than one that scatters across weights. Concretely I add
`0.5·(3 − w mod 3)`: weight `≡ 0 mod 3` earns `+1.5`, weight `≡ 1` earns `+1.0`, weight `≡ 2` earns
`+0.5`, so greedy is nudged toward the weight-`≡0` layer first, then `≡1`, then `≡2`. I want to be
candid that "coherent weight layer packs better" is a guess dressed as a heuristic, not a theorem — I
have a reason it is not *obviously* wrong, not a reason it is right.

It helps to size the layer I am biasing toward, to see whether the bias is even dimensionally sane.
At `n = 4` the weight distribution across weights `0..4` is `1, 8, 24, 32, 16` (a weight-`w` vector
picks `w` coordinates and fills each with a `1` or a `2`, so `C(4, w)·2^w`). The weight-`≡0` layer,
weights `0` and `3`, therefore holds `1 + 32 = 33` vectors; the weight-`≡1` layer, weights `1` and
`4`, holds `8 + 16 = 24`; the weight-`≡2` layer, weight `2` alone, holds `24`. The layer I front-load
has `33` candidates, comfortably more than the `20` an optimal cap needs, so preferring it is not
dimensionally absurd — there is room for a large cap inside a layer's worth of vectors, *if* that
layer's line structure cooperated. But "if it cooperated" is carrying the whole argument: I have no
proof the weight-`≡0` layer is line-sparse, only that it is not a coset and is big enough to hold the
answer. That is the recurring shape of every choice on this rung — dimensionally plausible,
structurally unproven.

The second feature is reflection symmetry, and it comes from a different kind of evidence: the known
large caps are highly structured objects, not random sets, and among the structures they exhibit is
symmetry under coordinate reflection. So I reward a vector when its reflected coordinate pairs agree
— a bonus of `1.0` for each `i` with `el[i] == el[n−1−i]` — on the idea that a cap assembled from
reflection-symmetric vectors inherits a symmetry that regularizes its lines. The same caution
applies: the set of exact palindromes is a *subspace* of dimension `⌈n/2⌉`, hence line-saturated, so
I bias toward reflection matches rather than restricting to palindromes. And here I notice something
about the specific reward I am writing that is worth flagging as a symptom of hand-design. The loop I
reach for runs `i` over `range(1, n//2)`, which at `n = 4` checks only the pair `(1, 2)`, at `n = 5`
only `(1, 3)`, at `n = 6` the pairs `(1, 4)` and `(2, 3)`, at `n = 7` the pairs `(1, 5)` and
`(2, 4)`. In every case it *skips the outermost pair* `(0, n−1)` and, at odd `n`, the self-paired
middle. So the reflection reward I am building is not even the full reflection symmetry — it is a
partial, interior-only version of it, an artifact of where the loop bounds happen to fall. That is
exactly the kind of off-by-one arbitrariness a hand-written score carries and a derived one would
not, and I leave it in deliberately, because this rung is partly a demonstration of what hand-design
looks like from the inside: reasonable-sounding features, imperfectly encoded, with no principle
telling me the encoding is the right one.

The interior-only reward is also *faint* at exactly the dimensions where I most need help. At `n = 4`
and `n = 5` the loop checks a single pair, so a vector earns at most `+1.0` of reflection bonus — one
unit against a base of `4` or `5` and a weight bonus of up to `1.5`, a barely-audible nudge. Only at
`n = 6, 7` does it check two pairs and the reflection signal reach `+2.0`. So the symmetry feature I
am most confident about on general grounds is the quietest part of the score where the caps are
smallest, and it strengthens only as `n` grows. I have no way to know whether that schedule is right;
it is simply what the loop bounds happen to produce. If reflection symmetry genuinely matters, I am
under-rewarding it at small `n`; if it does not, I am injecting noise at large `n`. Either direction,
the score has no dial that a principle set — only bounds I chose by hand and cannot defend against the
alternatives.

Assembling the pieces, the priority is `n` plus the reflection-match bonus plus the weight-layer
bonus plus a `0.01·Σ el` tie-break, and the tie-break earns its place. The reflection and weight
terms take only a handful of distinct values, so vast numbers of vectors share a score; without a
tie-break the argmax would break ties by array position, silently smuggling the lexicographic
pathology back in. The `0.01·Σ el` term is small enough not to reorder vectors of different
structural score but large enough to totalize the order deterministically within a tie, nudging
gently toward higher-symbol vectors. The skeleton is then the same greedy machine as before: score
all `3^n` vectors once, repeatedly take the highest-priority vector still in play, set the closing
point of every line it forms to `−∞`, set its own priority to `−∞`, and add it to the cap — greedy on
a geometry-aligned order instead of a blind one.

I can verify that the order really is geometry-flavored rather than accidental by computing a few
priorities at `n = 4`. The empty vector `(0,0,0,0)` scores `6.5` — weight `0` (`+1.5`), middle pair
matching (`+1.0`), no sum. The all-ones `(1,1,1,1)` scores only `6.04` — weight `4 ≡ 1` earns just
`+1.0`. A weight-`3` vector like `(1,1,1,0)` scores `6.53`, and `(2,2,2,0)` scores `6.56`, edging
above the empty vector because the sum tie-break rewards its `2`s. Indeed the top of the ranking at
`n = 4` is populated by weight-`3` (weight `≡ 0 mod 3`) vectors with the interior pair matching and
high symbol sum — `(0,2,2,2)` and `(2,2,2,0)` at `6.56`, `(0,2,2,1)` and `(1,2,2,0)` at `6.55` —
while the weight-`2` vectors like `(1,0,0,1)` sit down at `5.52`. So the priority does what I built
it to do: it front-loads the weight-`≡0` layer, favors interior reflection matches, and orders by
symbol sum inside ties. Whether *that* ordering is aligned with where the optimal `20`-cap actually
wants its points is the question I cannot answer from the score alone — the cap sizes will tell.

Stepping back, the three rungs are now building structurally distinct caps, and naming the
difference sharpens what is being tested. Lexicographic built a two-symbol *corner*, `{0, 1}^n`.
Random multi-start built an unstructured *scatter*, the luckiest of many diffuse maximal caps. This
structured order builds something concentrated in a *weight layer*: by front-loading weight-`≡0`
vectors it fills from the weights-`0`-and-`3` shell before it reaches the rest, so the cap it grows
is biased toward a coherent Hamming-weight profile. Three rungs, three geometries — corner, scatter,
layer — all fed through the identical greedy skeleton, differing only in the order the skeleton is
handed. The empirical question this rung poses is narrow and clean: is a weight-layer-biased order a
better bet than a scatter? If the layer bias captures something real about how the optimal caps are
organized, it should beat the sampled scatter; if the optimal caps are not organized by weight layer
at all, the bias is merely a differently-shaped arbitrariness and will land in the same band or
below. I do not know which, and that not-knowing is exactly the point I am trying to make concrete.

One more reading of the sampled baseline before I set my order against it, because the shape of
random's win tells me what I am up against. Its lift over lexicographic grows in absolute points —
`+4, +7, +13, +19` — yet its captured fraction of the optimum *falls* — `100%, 87%, 69%, 62%`. Both
hold at once: random reclaims more raw points at high `n` even while losing relative ground, because
the optimum is racing away faster than sampling can chase. So the baseline I must beat is strongest,
in fraction-of-optimum terms, exactly at low `n`, where it already sits *on* the optimum at `n = 4`
and near it at `n = 5`. That is the region where a structured order has the least room to win — there
is almost no gap left for structure to close that sampling has not already closed. Structure's best
chance to *out*-do sampling is at higher `n`, where random's fraction has slipped toward `62%` and a
well-aimed order might reach caps the scatter never traces. If my features are right, higher `n` is
where I would expect to see it; if they are wrong, higher `n` is also where a misaligned bias does
the most damage, since there the greedy fill commits many points under a score I cannot vouch for.
The dimensions cut both ways, and I hold no confident direction — only the clean test.

That uncertainty is the honest center of this rung. The symmetry rewards give greedy a *reason* to
prefer one point over another that is aligned with the geometry instead of with the counting order,
so this should beat lexicographic decisively at every `n` — a structured single order is far better
than a blind single order. But whether it beats the *maximum over thousands of random orders* is
genuinely unclear, because best-of-thousands is a strong baseline and I am putting one hand-tuned
order against it. If my guessed features — interior reflection matches, a weight-`≡0` layer — happen
to align with the optimal cap's structure, I win; if they are even slightly off, a single structured
order can easily land in the same band as, or below, the sampled best. So my honest expectation is:
comfortably above the lexicographic floor everywhere, sitting in the same general neighborhood as
random multi-start, and quite possibly *below* it at several `n`. At `n = 4` I would hope to clear
`16` but I would not bet on matching the `20` that pure sampling found; at the larger `n` I expect to
land near the random best, not dramatically past it, and I will not be surprised if some dimension
barely improves on the floor at all.

There is a handicap this rung carries that has nothing to do with whether my features are good: it is
a single deterministic order, so it gets exactly one attempt at the space. Random multi-start reached
`20` at `n = 4` by taking five thousand attempts and keeping the luckiest; a structured order gets
the one cap its one ordering produces, with no second draw to rescue a near-miss. So even if my
priority aims greedy at a genuinely better region than a typical random order does, it can still be
out-scored by best-of-thousands, because the maximum of many mediocre-but-varied draws can beat a
single good-but-fixed one. This is the same tension the first rung exposed — one deterministic order
is one uncontrolled draw from the space of maximal caps — only now the draw is *aimed* rather than
blind. Better aim raises the mean of where I land; it does not hand me the max over many landings.
Against a baseline that is explicitly a max-over-many, one aimed shot is not guaranteed to win, and I
should not pretend otherwise.

And I have no compass even for repairing the score's visible flaws. The reflection loop skips the
outer pair — should I fix it to `range(0, n//2)` and reward the full symmetry? Perhaps. But that is
merely another guess: I have no evidence the outer pair matters more or less than the interior ones,
no evidence that `1.0` is the right reflection bonus rather than `0.5` or `2.0`, no evidence that the
favored layer should be weight-`≡0` rather than `≡1`, no evidence that weight and reflection should
*add* rather than interact. Every constant in this function is a knob I set by taste, and there are
more knobs than I have principles to turn them. That is the deepest reason to doubt hand-design: not
that any single choice is demonstrably wrong, but that the choices are many and ungrounded, and the
odds that a hand-picked point in that many-knobbed space happens to be the *right* one are poor. A
process that could turn those knobs against the actual evaluator, thousands of times over, keeping
only what measurably builds larger caps, would not share this handicap — and that is the shape of the
move this rung is quietly arguing for.

If it comes out that way, it is not a disappointment — it is the crucial result of the ladder, and I
can state the falsifiable version now. If the mechanism I believe is right — that the skeleton is
correct but a *hand-guessed* priority is not the right function — then the structured order should
beat `2^n` where my features happen to help but land at or below the random best of `20, 39, 77, 147`
across `n = 4..7`, rather than pushing toward the optima `20, 45, 112, 236`. If instead this
hand-designed score sailed past random multi-start and closed on the optima, I would be wrong that
hand-design plateaus, and the whole motivation for the final rung would evaporate. I do not expect
that. I expect a structured order that is clearly better than blind counting yet cannot reliably
out-do brute sampling, which would show that *having* a structured priority is not enough: the
priority has to be the *right* one, finely shaped to the dimension, and a human guessing at which
symmetries to reward — reflection pairs, a weight layer, imperfectly encoded — cannot dependably beat
a lottery over orders. The greedy-priority skeleton is clearly the correct machinery; it is exactly
what the strong constructions use. But the function plugged into it is everything, and the space of
useful priorities is large and non-obvious, full of conditional, dimension-specific structure that no
one would write down from symmetry-first reasoning. What reaches the records is not a cleverer human
guess but a priority *discovered by search* over the function space, tuned to the specific `n`,
encoding regularities a person would not think to hand-code. Establishing that the skeleton is right
and the hand-designed priority is not is exactly the job of this rung — and it is what would motivate
handing the priority itself over to search, which is where the ladder goes next.

One caution I will respect for the numbers: because the structured priority has heavy ties and the
tie-break is what totalizes the order, I fix that tie-break deterministically and run the verifier on
every returned cap — and the independent triple scan at small `n` — before trusting any size. The
priority changes which order greedy walks, never whether the output is a cap; but I check anyway,
because a size I have not verified is a size I do not believe.
