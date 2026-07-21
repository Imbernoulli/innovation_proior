Random multi-start did what I predicted and drew the ceiling cleanly. It cleared the lexicographic
floor at every dimension — `20, 39, 77, 147` against `16, 32, 64, 128`, a lift of `+4, +7, +13, +19`
— and reached the proven optimum `20` at `n = 4`, the first time I touched a known-optimal value.
But the gap grows on schedule: as a fraction of the optimum the sampled best captures `100%` at
`n = 4`, `87%`, `69%`, `62%` at `n = 5, 6, 7`. The captured fraction slides with dimension — the
signature of a method whose reach detaches from the target as the target becomes structured.
Sampling removed the *bias* of a fixed order but not its *blindness*: every order it tried was
uniform noise, ceilinged at the top of the greedy-reachable band, and that band sinks below the
optimum once the optimum demands algebraic regularity a random arrival sequence never traces.

So the move is to stop sampling and start *biasing*: score every vector by a deterministic function
of its coordinates and feed the same greedy admission rule the vectors highest-priority-first. The
admission rule is unchanged, so validity stays free; only the ordering function changes. What is
different in kind is the bet — this is *one order again*, seedless like lexicographic, but a
geometry-aware one. The wager is that a single well-chosen structured order can beat the best of
thousands of blind ones, because it points greedy at the points that pack well rather than letting
it stumble. That is a real gamble: if my structure is the wrong structure, one shot loses to
best-of-thousands.

Which structure of `F_3^n` to reward? The first instinct is a trap worth walking into to kill it.
The line condition `a + b + c ≡ 0` is about *coordinate sums mod 3*, so the tempting move is to fix
the total sum — prefer vectors whose coordinate sum lies in one residue class. But the total sum is
a *linear* functional, and that dooms it. A sum-fixed set is an affine coset, and if `a, b` both
have sum `≡ r`, their third point `−a − b` has sum `−2r ≡ r` too — the coset is *closed* under line
completion. At `n = 4` the twenty-seven vectors with sum `≡ 0` form a coset in which every pair's
third point stays inside: not merely non-cap, but saturated with lines, about the worst thing to
build a cap from. The linear invariant is the one structure I must *not* fix; the failure tells me
to reach for a *non-linear* feature the line operation does not preserve.

The natural non-linear feature is the Hamming weight mod 3. The weight of `−a − b` bears no additive
relation to the weights of `a` and `b`, so a weight-mod-3 layer is not a coset and carries no
automatic guarantee of being line-saturated — its only virtue is that it is *not obviously wrong*. I
cannot *restrict* greedy to a single weight layer, since a layer holds on the order of `3^{n-1}`
vectors, still full of lines. I can only *softly bias* toward a coherent layer, on the heuristic
that a cap whose points share a weight profile might have more regular line structure. So I add
`0.5·(3 − w mod 3)`: weight `≡ 0` earns `+1.5`, `≡ 1` earns `+1.0`, `≡ 2` earns `+0.5`, nudging
greedy toward the weight-`≡0` layer first. "Coherent weight layer packs better" is a guess dressed as
a heuristic — a reason it is not *obviously* wrong, not a reason it is right. At `n = 4` the weight
counts across `0..4` are `1, 8, 24, 32, 16`, so the weight-`≡0` layer (weights `0, 3`) holds `33`
candidates, comfortably more than the `20` an optimal cap needs — room for a large cap inside a layer
*if* its line structure cooperated, and "if it cooperated" is carrying the whole argument.

The second feature is reflection symmetry, from different evidence: the known large caps are highly
structured objects, and among their regularities is symmetry under coordinate reflection. So I reward
`+1.0` for each `i` with `el[i] == el[n−1−i]`, on the idea that a cap assembled from
reflection-symmetric vectors inherits a symmetry that regularizes its lines. Exact palindromes form
a subspace, hence line-saturated, so I bias toward matches rather than restrict to palindromes. And
the specific reward I write is worth flagging as a symptom of hand-design: the loop runs `i` over
`range(1, n//2)`, which at `n = 4` checks only the pair `(1, 2)`, at `n = 6` the pairs `(1, 4)` and
`(2, 3)` — in every case *skipping the outermost pair* `(0, n−1)` and, at odd `n`, the self-paired
middle. So it is not the full reflection symmetry but a partial, interior-only version, an artifact
of where the loop bounds fall — the off-by-one arbitrariness a hand-written score carries and a
derived one would not. I leave it in deliberately, as an honest picture of what hand-design looks
like from the inside: reasonable features, imperfectly encoded, with no principle telling me the
encoding is right. Worse, the interior-only reward is faintest where I most need help: at `n = 4, 5`
it checks a single pair, at most `+1.0` against a base of `4` or `5`, a barely-audible nudge; only at
`n = 6, 7` does it reach `+2.0`. The symmetry I am most confident about is the quietest part of the
score where the caps are smallest, on a schedule no principle set.

Assembling: the priority is `n` plus the reflection bonus plus the weight-layer bonus plus a
`0.01·Σ el` tie-break. The tie-break earns its place — the reflection and weight terms take few
distinct values, so vast numbers of vectors share a score, and without a tie-break the argmax would
break ties by array position, smuggling the lexicographic pathology back in. The `0.01·Σ el` term is
small enough not to reorder vectors of different structural score but large enough to totalize the
order within a tie. The skeleton is the same greedy machine: score all `3^n` vectors once, repeatedly
take the highest-priority vector still in play, set the closing point of every line it forms and its
own priority to `−∞`, add it to the cap. A few priorities at `n = 4` confirm the order is
geometry-flavored: the empty vector scores `6.5`, the all-ones `6.04`, and the top of the ranking is
populated by weight-`3` (weight `≡ 0`) vectors with the interior pair matching and high symbol sum —
`(2,2,2,0)` at `6.56` — while weight-`2` vectors like `(1,0,0,1)` sit down at `5.52`. So the order
front-loads the weight-`≡0` layer as built. Whether *that* ordering is where the optimal `20`-cap
wants its points, the cap sizes will tell. Three orders now build three geometries through one
skeleton — a two-symbol *corner*, an unstructured *scatter*, and now a weight *layer*.

The honest center is that I do not know if the bet pays. The symmetry rewards give greedy a *reason*
to prefer one point over another aligned with the geometry, so this should beat lexicographic
decisively at every `n`. But whether it beats the *maximum over thousands of random orders* is
genuinely unclear: best-of-thousands is a strong baseline, and I am putting one hand-tuned order
against it. There is a handicap independent of feature quality: a deterministic order gets exactly
one attempt, while multi-start reached `20` by keeping the luckiest of five thousand. Better aim
raises the *mean* of where I land; it does not hand me the *max* over many landings. So my honest
expectation is comfortably above the lexicographic floor everywhere, sitting near random multi-start,
and quite possibly *below* it at several `n` — at `n = 4` I would hope to clear `16` but would not
bet on matching `20`.

And I have no compass even for repairing the score's visible flaws. Should I fix the loop to reward
the outer pair? Perhaps — but that is merely another guess. I have no evidence the outer pair matters
more than the interior ones, that `1.0` is the right reflection bonus rather than `0.5` or `2.0`,
that the favored layer should be weight-`≡0` rather than `≡1`, or that weight and reflection should
*add* rather than interact. Every constant is a knob I set by taste, and there are more knobs than I
have principles to turn them — the deepest reason to doubt hand-design: not that any single choice is
demonstrably wrong, but that the choices are many and ungrounded, and the odds that a hand-picked
point in that space is the right one are poor.

So the prediction: the structured order beats `2^n` where my features happen to help but lands at or
below the random best of `20, 39, 77, 147` across `n = 4..7`, rather than pushing toward the optima.
If instead this score sailed past random multi-start and closed on the optima, I would be wrong that
hand-design plateaus. I expect the plateau: the greedy-priority *skeleton* is exactly what the strong
constructions use, but the function plugged into it is everything, and the space of useful priorities
is full of conditional, dimension-specific structure no one writes down from symmetry-first
reasoning. What reaches the records is a priority *discovered by search* over the function space,
tuned to the specific `n` — where I go next.
