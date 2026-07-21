The structured priority came in exactly where I feared. It scored `18, 36, 64, 138` at `n = 4..7` —
beating the lexicographic floor at `n = 4, 5, 7` by `+2, +4, +10` and merely tying it at `n = 6` —
but it landed *below* best-of-thousands random multi-start at every dimension: `18 < 20`, `36 < 39`,
`64 < 77`, `138 < 147`. My most informed guess, a score built from the two symmetries I was most
confident about, lost to a blind lottery over orders everywhere. That loss is decisive: it rules out
the skeleton as the bottleneck, since the same greedy machine that reaches `147` under a random order
reaches only `138` under my structured one. The machine is fine; the ordering function I fed it is
worse than noise-plus-a-maximum. Two dimensions sharpen it. At `n = 6` the structured priority scored
`64`, dead level with the floor — the weight-and-reflection bias produced *nothing* over the counting
corner, as clean a demonstration as I could want that a plausible structure can be inert. At `n = 7`
it posted its largest margin, `+10`, its best relative showing at the highest dimension — consistent
with structure mattering more as `n` grows — yet even there it trailed the random best by `9`. No
re-tuning of these features closes that gap, because the gap is in the *form* of the function, not
the magnitude of its bonuses. The choice of priority moves the number (`147 > 138 > 128`); nothing
about the skeleton does. And I now suspect the function space is full of conditional,
dimension-specific structure that symmetry-first reasoning would never produce.

So the move is to stop guessing the function and hand it to *search*: something that turns the
priority's knobs against the actual evaluator, thousands of times over, keeping only what measurably
builds larger caps. An evolutionary program search does precisely that — a loop pairing a pretrained
language model with the cap-set evaluator, mutating the *body* of the `priority` function, scoring
each candidate by the size of the cap the greedy skeleton builds with it, selecting survivors over
millions of samples. The skeleton stays fixed; only the function evolves. The plan is to take the
function such a search actually discovered (FunSearch) and run it verbatim through the same skeleton,
and see whether a *searched* function reaches what no blind lottery and no hand guess could.

The discovered function's shape is the sharpest evidence for why hand-design failed. It branches
first on the number of zeros. Full-weight vectors — no zeros — get a large *additive* boost `+n²`,
and then reflection-pair matches `el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]` each *multiply*
the score by `1.5` rather than adding. Vectors with zeros instead get their first two reflection
matches *penalized* by `×0.5`, and then the function walks the coordinates applying a *position- and
count-dependent* multiplicative factor to each zero: the first zero scales by `n·0.5`, the last by
`0.5`, an interior zero at ordinal position `in_el` by `n·0.5^{in_el}`, while every nonzero
coordinate adds `1`; then two further matches multiply by `1.5` again. This is nothing like the clean
symmetric *sum* I wrote — it is deeply non-linear, branch-heavy, multiplicative, and sensitive to
both the count and the ordinal positions of the zeros. No one reasoning from "reward symmetry, prefer
a weight layer" writes `n·0.5^{in_el}` as a per-zero multiplier keyed on ordinal position; that term
is not a symmetry, it is a fitted knob found by selecting whatever made the greedy fill reach a large
cap.

Evaluating it at `n = 8` on two extremes shows how violent the non-linearity is. The all-`2`s vector
is full weight, so `score = 8 + 64 = 72`, the three interior matches send it `72 → 108 → 162 → 243`,
the loop adds `1` per coordinate to `251`, and the two final matches lift it to `564.75`. The
single-`1` vector `(1,0,…,0)` has seven zeros: the else-branch penalties knock `8` to `2`, the
per-zero multipliers leave it at `6`, and the final matches only reach `13.5`. A factor of forty
between two vectors differing by one coordinate's weight. The function enormously front-loads
full-weight, reflection-palindromic vectors and buries everything with many zeros.

Two contrasts with my function explain the form gap, both things I would not have crossed on my own.
First, my features were all *invariants* — Hamming weight, reflection matches — quantities with clear
symmetry meaning, the only kind a human naturally rewards. The `n·0.5^{in_el}` factor is keyed on the
ordinal position of the k-th zero counting from the left: not an invariant, no symmetry
interpretation, a bare positional quantity a symmetry-first designer would never reward because it
looks meaningless. The search rewards it anyway, for the only reason it has — it correlates with
building a large cap. Second, my score was a *sum*, so features could not interact: a reflection
match added `+1` whether the vector was heavy or light. This score is a *product*: a reflection match
multiplies by `1.5`, worth a large amount on a full-weight vector already boosted into the hundreds
and almost nothing on a light one. That interaction — reflection mattering *only in combination with*
full weight — is inexpressible in my additive form, and plausibly where the discriminating power
lives.

Its constants also say *which* dimension it is for. It was discovered by a search running specifically
at `n = 8`, and every magic number — the `×1.5` factors, the `n·0.5^{in_el}` weights, the very
presence of an `el[3]==el[-3]` term reaching three coordinates deep — is a value the search settled
on because it made the `n = 8` fill large. A function that precise about eight-dimensional structure
carries no guarantee elsewhere; it is not even well-defined for `n < 4`, since it indexes `el[1..3]`
and their mirror partners. So I expect it *mediocre* below `n = 8`, plausibly falling back toward the
`2^n` neighborhood, with its entire value concentrated at `n = 8` where it does the one thing it was
selected to do — build `512`, improving the previous best construction of `496`. The scored metrics
should come back `16, 32, 64, 128` at `n = 4..7` and `512` at `n = 8`; the strong check is set-level,
whether the `n = 8` output coincides point-for-point with the recorded `512`-cap rather than merely
matching its size.

Running it confirms the mediocrity below `n = 8`, and the way it is mediocre closes a loop with the
lexicographic floor. At `n = 4, 5, 6, 7` it returns exactly `16, 32, 64, 128`, and every admitted
vector is full-weight: the function rebuilds `{1, 2}^n`, the *other* two-symbol corner. Lexicographic
built `{0, 1}^n`; this builds `{1, 2}^n`; both are `2^n`, the same floor from opposite corners. And I
can see why it collapses. At `n = 4, 5, 6` the full-weight minimum score (`51, 35, 69`) sits above
the zero-containing maximum (`15.75, 24.38, 50.62`), so every full-weight vector outranks every
vector with a zero; greedy lays the entire `{1, 2}^n` backbone first, and since `{1, 2}^n` is a cap
its members never block one another, after which every remaining vector — all carrying a zero, hence
the third point of some line through a `{1, 2}`-pair — is blocked. At `n = 7` the raw orders overlap
(top zero score `107` exceeds worst full-weight `63`), but the highest full-weight palindromes are
admitted first and blacken the high-scoring zero-vectors before greedy descends to them; the backbone
completes anyway. Then at `n = 8` something tips, and that is the whole payoff. Only `128` of the
`256` full-weight vectors are admitted — the high-symmetry palindromic half the `×1.5` factors push
to the top — and after them `384` zero-containing vectors survive and enter, for `128 + 384 = 512`.
It is *that specific symmetric core*, rather than the full backbone, that leaves the line structure
open enough for `384` zero-carrying vectors to be added without closing a line. Why `n = 8` is where
the tuned constants tip the ordering from "complete the backbone, seal everything" into "take a
symmetric half, then admit a large family of zero-vectors" I cannot derive from the constants — that
phase change is exactly what the search discovered and what I could not have reasoned my way to. I
can observe it and explain its mechanics after the fact; I could not have written the function.

The `n = 8` cap has size `512`, passes the incremental verifier, and — the stronger check — coincides
as a set with the explicit `512`-cap recorded in the FunSearch repository (`cap_set/n8_size512.txt`),
all `512` points matching, so the run reconstructs the exact object, not merely its size. The
companion discovered function for `n = 9` reaches the known-best `1082` through the identical
skeleton.

Placing `512` against the two ceilings the earlier orders mapped is what makes it more than a bigger
number. Random multi-start could only reach the top of the greedy-reachable band, and a `512`-cap is
emphatically outside it — no random arrival sequence traces the `128`-vector symmetric core and its
`384` companions by chance. And it is outside the reach of hand-symmetry, as the structured
priority's plateau already showed. So `512` lives in the narrow region reachable by the greedy
skeleton but *only* under a priority precise enough to select that exact core — a region neither
blind sampling nor principled hand-design can enter. Search threads it because search is the one
procedure that finds a function by its *results* rather than its *rationale*, and this cap demanded a
function with the right results and no human-legible rationale.

And here I honestly stop. `512` is the current construction frontier; I am *reproducing* it, not
beating it. Everything I have run has been a single constructor — one greedy build, or a few thousand
for multi-start, each `O(|cap|^2·n)`. The search that produced this function evaluated on the order
of millions of candidate priorities, each a language-model generation fed through a full greedy build
— a budget several orders of magnitude beyond anything here, and a fundamentally different object: a
program that writes and tests programs. Producing a *better* output would mean re-running that outer
loop, which is not a single deterministic constructor and is out of scope. The arc is clean: the
skeleton was right from the start, and the whole difficulty was always the function plugged into it —
the part that search, not derivation, had to supply.
