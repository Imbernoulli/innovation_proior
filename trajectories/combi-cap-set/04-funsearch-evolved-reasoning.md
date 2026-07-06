The structured priority came in exactly where I feared and the feedback settled the architecture.
It scored `18, 36, 64, 138` at `n = 4..7` — beating the lexicographic floor at `n = 4, 5, 7` by
`+2, +4, +10` and merely tying it at `n = 6` — but it landed *below* best-of-thousands random
multi-start at every dimension: `18 < 20`, `36 < 39`, `64 < 77`, `138 < 147`. My most informed
guess, a hand-designed score built from the two symmetries I was most confident about, lost to a
blind lottery over orders at every `n`. That is the decisive result, and it is decisive precisely
because it is a loss. It rules out the possibility that the skeleton is the bottleneck: the same
greedy machine that reaches `147` under a random order reaches only `138` under my structured one,
so the machine is fine and the ordering function I fed it is worse than noise-plus-a-maximum. The
weight-layer bias, dimensionally plausible and structurally unproven, turned out to be the wrong
structure — or right structure, wrong encoding, which for these purposes is the same thing.

The ordering of the three hand-or-blind priorities is itself the lesson in miniature: `147` (random)
`> 138` (structured) `> 128` (lexicographic). My structured order beat blind counting — a
geometry-aware order really does help over a geometry-blind one — but it could not reach the blind
lottery's maximum. And two dimensions sharpen the failure. At `n = 6` the structured priority scored
`64`, dead level with the lexicographic floor: the weight-and-reflection bias produced *nothing at
all* over the counting corner there, as clean a demonstration as I could want that a plausible
structure can simply be inert. At `n = 7`, by contrast, the structured order posted its largest
margin over the floor, `+10`, its best relative showing at the highest dimension — consistent with my
guess last rung that structure's chance to matter grows with `n` — and yet even there it trailed the
random best by `9`. So the pattern is unambiguous: hand-structure is real but weak, helpful over
nothing and useless against a maximum-over-many. No re-tuning of these same features closes that gap,
because the gap is not in the magnitude of the bonuses but in the *form* of the function.

Let me name what the whole ladder has now established, because it points at the only move left. Every
rung has been the identical greedy skeleton — score the vectors, add the highest-scoring valid one,
blacken the closing point of every line it forms, repeat — differing only in the priority function
plugged in. Lexicographic is the priority "negative index of the vector," and it builds the
two-symbol corner `{0, 1}^n`. Random multi-start is the priority "a fresh random score each restart,
keep the best," and it builds an unstructured scatter. My structured attempt is the priority
"reflection matches plus a weight layer," and it builds a weight-biased set. Three priorities, three
geometries, and all three plateau below the optima — with the ordering among them, `147 > 138 > 128`,
showing that the *choice of priority* is what moves the number, not anything about the skeleton. The
skeleton is not the limitation; the priority function is. And the last rung taught me why a human
cannot reliably supply it: the function has many constants and I have few principles to set them, so
the odds that a hand-picked point in that many-knobbed space is the right one are poor. The space of
useful priorities is large, non-obvious, and — I now suspect — full of conditional, dimension-specific
structure that symmetry-first reasoning would never produce.

So the move is to stop guessing the function and hand it to *search*. This is exactly the process I
was circling at the end of the last rung: something that can turn the priority's knobs against the
actual evaluator, thousands of times over, keeping only what measurably builds larger caps. An
evolutionary program search does precisely that — a loop pairing a pretrained language model with the
cap-set evaluator, mutating the *body* of the `priority` function, scoring each candidate by the size
of the cap the greedy skeleton builds with it, and selecting the survivors over millions of samples.
The skeleton stays fixed; only the function evolves, under selection pressure toward larger caps. The
endpoint of this ladder is to take the priority function that such a search actually discovered
(FunSearch) and run it, verbatim, through the same skeleton every earlier rung used — and to see
whether a *searched* function reaches what no blind lottery and no hand guess could.

Before running it I want to read the discovered function, because its shape is the sharpest possible
evidence for why hand-design failed. It branches first on the number of zeros in the vector,
`el_count = el.count(0)`, and treats the two branches as almost different functions. Full-weight
vectors — no zeros — get a large *additive* boost, `score += n²`, and then reflection-pair matches
`el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]` each *multiply* the score by `1.5` rather than
adding to it. Vectors that do have zeros instead get their first two reflection matches *penalized*
by `×0.5`, and then the function walks the coordinates applying a *position- and count-dependent*
multiplicative factor to each zero: the first zero scales by `n·0.5`, the last zero by `0.5`, and an
interior zero at ordinal position `in_el` by `n·0.5^{in_el}`, while every nonzero coordinate adds
`1`. Then, on top of all of that, two further reflection matches `el[1]==el[-1]` and `el[2]==el[-2]`
multiply by `1.5` again. This is nothing like the clean symmetric *sum* I wrote — it is deeply
non-linear, branch-heavy, multiplicative, and sensitive to both the count and the ordinal positions
of the zeros. No one reasoning from "reward symmetry, prefer a weight layer" writes `n·0.5^{in_el}`
as a per-zero multiplier keyed on ordinal position; that term is not a symmetry, it is a fitted
knob. Its structure was *found* by selecting whatever made the greedy fill reach a large cap, not
derived from any principle about `F_3^n`.

I can see how violently non-linear it is by evaluating it at `n = 8` on two extremes. Take the
all-`2`s vector `(2,2,2,2,2,2,2,2)`: full weight, so `score = 8 + 64 = 72`, then the three interior
reflection matches send it `72 → 108 → 162 → 243`, the loop adds `1` per coordinate to `251`, and the
two final matches send it `251 → 376.5 → 564.75`. Now take the single-`1` vector `(1,0,0,0,0,0,0,0)`:
seven zeros, so the else-branch penalties knock `8` down to `2`, and the per-zero multipliers
`×4, ×4, ×2, ×1, ×0.5, ×0.25, ×0.5` — interleaved with `+1` for the lone nonzero — leave it at `6`,
which the two final matches lift only to `13.5`. A factor of forty between two vectors that differ by
a single coordinate's worth of weight. The function is not gently preferring symmetry; it is
enormously front-loading full-weight, reflection-palindromic vectors and burying everything with many
zeros. Whatever cap it builds will be laid down starting from the `{1, 2}^n` region, in order of
palindromic symmetry.

Two structural contrasts between my function and this one explain the form gap, and both are things I
would not have crossed on my own. The first is that my features were all *invariants* — Hamming
weight, reflection matches — quantities with clear symmetry meaning, the only kind a human naturally
rewards. The discovered function rewards `n·0.5^{in_el}`, a factor keyed on the *ordinal position of
the k-th zero, counting from the left*. That is not an invariant of anything; it has no symmetry
interpretation; it is a bare positional quantity a symmetry-first designer would never reward because
it looks meaningless. The search rewards it anyway, for the only reason the search has: it correlates
with building a large cap. A searched function is free to exploit non-invariant, order-dependent
structure that a derived one forbids itself as unprincipled. The second contrast is additive versus
multiplicative. My score was a *sum* of bonuses, so its features could not interact — a reflection
match added `+1` whether the vector was heavy or light. This score is a *product*: a reflection match
multiplies by `1.5`, worth a large absolute amount on a full-weight vector already boosted into the
hundreds and almost nothing on a light one. That interaction — reflection mattering *only in
combination with* full weight — is expressible multiplicatively and inexpressible in my additive
form, and it is plausibly where the discriminating power lives. Between non-invariant features and
multiplicative interaction, the discovered function sits in a region of function-space my hand-design
could not even address, let alone tune.

The crucial question is *which* dimension this function is for, and its constants answer it. It was
discovered by a search running specifically at `n = 8`, and every magic number in it — the `×1.5`
reflection factors, the `n·0.5^{in_el}` zero weights, the very presence of an `el[3]==el[-3]` term
that reaches three coordinates deep from each end — is a value the search settled on because it made
the `n = 8` greedy fill large. A function that precise about eight-dimensional structure carries no
guarantee anywhere else. So I should expect it to be *mediocre* below `n = 8`: run outside the regime
it was fitted to, it has no reason to pack `F_3^4` or `F_3^7` well, and I would not be surprised to
see it fall back toward the trivial `2^n` neighborhood, possibly at or below even my structured
priority. Its entire value should be concentrated at `n = 8`, where it does the one thing it was
selected to do — build a cap of `512`, the size that no random sampling and no hand-designed symmetry
on this ladder reached, improving the previous best construction of `496`.

There is even a hint of the specialization in the function's *domain*. It indexes `el[1]`, `el[2]`,
`el[3]` and their mirror partners `el[-1]`, `el[-2]`, `el[-3]`, so it is not well-defined for `n < 4`
— reaching three coordinates in from each end presumes at least a handful of coordinates to reach
into. A priority written from a general principle would degrade gracefully toward small `n`; this one
simply does not apply there, because it was never meant to. Its `el[3]==el[-3]` term in particular —
a third reflection pair that only becomes a distinct constraint once `n` is large enough for the two
ends not to overlap — is a fingerprint of a function shaped around the geometry of a *large* space,
`n = 8` above all, and carried elsewhere only as an afterthought.

Stated as a falsifiable prediction before I run it: if the dimension-specialization read is right,
the scored metrics should come back `16, 32, 64, 128` at `n = 4..7` — back at the `2^n` floor, at or
below both my structured priority and the random best — and `512` at `n = 8`. The strong falsifier is
the set-level check: if the `n = 8` output failed to coincide point-for-point with the recorded
`512`-cap, I would not have reproduced the discovery at all, only stumbled onto a same-sized cap, and
the claim to have run the genuine discovered function would collapse. And if the function were
secretly general, it would beat `2^n` at the smaller `n` too — which I do not expect, since a function
this specific to eight-dimensional structure has no reason to help elsewhere.

Running it through the skeleton confirms the mediocrity below `n = 8`, and the way it is mediocre is
worth seeing, because it closes a loop with the first rung. At `n = 4, 5, 6, 7` the construction
returns exactly `16, 32, 64, 128` — `2^n` again — and inspecting the caps, every admitted vector is
full-weight: the function simply rebuilds `{1, 2}^n`, the *other* two-symbol corner. Lexicographic
built `{0, 1}^n`; the discovered function, front-loading no-zero vectors, builds `{1, 2}^n`; both are
size `2^n`, the same floor reached from opposite corners of the cube. And I can see exactly why it
collapses to the corner at these dimensions. Comparing the highest-scoring zero-containing vector to
the lowest-scoring full-weight vector, at `n = 4, 5, 6` the full-weight minimum (`51, 35, 69`) sits
above the zero-containing maximum (`15.75, 24.38, 50.62`), so every full-weight vector outranks every
vector with a zero. Greedy therefore lays the entire `{1, 2}^n` backbone first — and `{1, 2}^n` is a
cap, so its members never block one another — after which every remaining vector, all of which carry
a zero, is blocked, because each zero-containing vector is the third point of some line through a
`{1, 2}`-pair. The backbone completes, seals off the rest, and the cap is exactly `2^n`.

At `n = 7` the raw ordering already overlaps — the top zero-containing score `107` exceeds the worst
full-weight score `63` — yet the run still returns `128` with no zero-vectors, because the highest
full-weight palindromes are admitted first and, as they accumulate, they blacken the high-scoring
zero-vectors before greedy ever descends to them; the backbone completes anyway. Then at `n = 8`
something tips, and this is the whole payoff. Running it, only `128` of the `256` full-weight vectors
are admitted — the high-symmetry, reflection-palindromic half that the `×1.5` factors push to the top
(the all-`2`s vector and its kin at `564.75`) — and after them, starting at admission position `128`,
`384` zero-containing vectors survive and enter, for a total of `128 + 384 = 512`. The reflection
multipliers select a symmetric `128`-vector core out of `{1, 2}^8`, and it is *that specific core*,
rather than the full backbone, that leaves the line structure open enough for `384` zero-carrying
vectors to be added without closing a line. Why `n = 8` is the dimension where the tuned constants
tip the ordering from "complete the backbone, seal everything" into "take a symmetric half, then
admit a large family of zero-vectors" I cannot derive from the constants themselves — that phase
change is exactly what the search discovered and what I could not have reasoned my way to. I can
observe it, verify it, and explain its mechanics after the fact; I could not have written the
function that produces it.

I verify genuineness rather than assert it, with two checks. First, the `n = 8` cap has size `512`
and passes the incremental cap verifier — a real cap, not a claimed number. Second, and stronger, the
set of `512` points the greedy fill produces coincides, as a set, with the explicit `512`-cap
recorded in the FunSearch repository (`cap_set/n8_size512.txt`): all `512` points match, so the run
reconstructs the exact object the search found, not merely its size. The companion discovered
function for `n = 9` reaches the known-best `1082` in nine dimensions through the identical skeleton,
which I note as corroboration that the mechanism generalizes — the same "search the function, keep the
skeleton" move produces the record at a second dimension.

The `n = 8` number has a suggestive arithmetic worth pausing on. The trivial corner `{1, 2}^8` holds
`256` points; the record cap holds `512`, exactly double it, assembled from a `128`-vector symmetric
slice of that corner plus `384 = 3·128` zero-carrying vectors, the whole `512` running four times
that retained `128`-vector core. I will not over-read the numerology, but it underscores that `512` is not an incremental nudge above the
floor: it *doubles* the corner that both lexicographic and this very function collapse to at smaller
`n`. And it is still far from any proven ceiling — the cap-set theorem's `O(2.756^8) ≈ 3329` sits
about six times higher, and the true maximum at `n = 8` is open — so `512` is a *construction*
frontier, the largest cap anyone builds, not a theorem about the largest that exists. Reproducing it
means matching the best known construction, which is the honest most I can claim from it.

It is worth placing `512` against the two ceilings the earlier rungs mapped, because that is what
makes it more than a bigger number. Random multi-start could only reach the top of the
greedy-reachable band, and that band's ceiling detached from the optimum as `n` grew; a `512`-cap at
`n = 8` is emphatically outside the band — no random arrival sequence traces the `128`-vector
symmetric core and its `384` companions by chance, which is why sampling would never find it however
many restarts I spent. And it is outside the reach of hand-symmetry, as my structured priority's
plateau below the lottery already showed. So `512` lives in the narrow region that is reachable by
the greedy skeleton but *only* under a priority precise enough to select that exact symmetric core —
a region neither blind sampling nor principled hand-design can enter. The searched function threads
it because search is the one procedure that can find a function by its *results* rather than its
*rationale*, and this cap demanded a function with the right results and no human-legible rationale.

It is worth being concrete about why going past that is a different computation and not just more of
the same. Everything on this ladder has been a single constructor: one greedy build, or at most a few
thousand of them for multi-start, each costing `O(|cap|^2·n)`. The search that produced this function
evaluated on the order of *millions* of candidate priority functions, every one a language-model
generation fed through a full greedy build and verification — a budget several orders of magnitude
beyond the entire ladder, and a fundamentally different object: a program that writes and tests
programs, rather than a program that builds a cap. Reproducing its output with one constructor is
cheap; producing a *better* output would require re-running that outer evolutionary loop, which is
out of scope for a constructor and is exactly why there is no honest rung above `512` here.

And here the ladder honestly ends. `512` at `n = 8` is the current frontier for this construction
set; I am *reproducing* it, not beating it, and there is no rung above this that I should claim.
Going further would not mean writing a cleverer constructor — it would mean running the evolutionary
program search itself, millions of language-model samples each scored by the evaluator, to discover a
*new* priority function, which is a different kind of computation entirely, not a single
deterministic constructor of the sort every rung on this ladder has been. So the arc is complete and
it says one thing cleanly: a fixed blind order gives the `2^n` corner; many random orders give the
best of a thin, low-ceilinged lottery, optimal only at `n = 4`; a hand-designed structured priority
lands in the same band and no higher, below even the lottery; and only a priority *discovered by
search* over the function space reaches the record `512`. The skeleton was right from the first rung.
The whole difficulty was always the function plugged into it — and the function was the part that
search, not derivation, had to supply.
