The last rung settled the architecture and exposed the real bottleneck. The greedy-priority
skeleton is the right machinery — score every vector, add the highest-priority valid one, block the
closed lines, repeat — and the earlier rungs are all special cases of it: lexicographic is the
priority "index of the vector," random multi-start is "a random score, best of many," my structured
attempt is "reflection matches plus a weight layer." Every one of them plateaus below the optima,
and the structured priority, my best hand-designed guess, actually fell *below* best-of-thousands
random restarts. That ordering is the thing I keep turning over: I reasoned from symmetry, wrote
down a clean reflection-aware score, and it lost to noise. The skeleton is not the limitation; the
function fed into it is, and the function I can write by hand is apparently the wrong function. If
the useful priorities are not the symmetric, human-legible ones, then they live somewhere I cannot
reach by reasoning from `F_3^n`'s structure — which means the honest move is to stop writing the
priority and let a search write it instead. An evolutionary loop that mutates the body of `priority`,
runs each candidate through the evaluator, and keeps the ones that build larger caps is exactly the
shape of computation that could find a function I would never propose. So the endpoint I want to
examine is whatever priority such a search actually lands on, run through the unchanged skeleton.

A function found that way will not look like mine, and it is worth pinning down *how* it differs,
because the difference is the whole point. My structured attempt rewarded reflection matches and a
weight layer, but it applied them *uniformly* — a flat bonus, the same regardless of how many zeros
a vector has. The discovered function is conditional in a way I would not have written. It branches
first on the number of zeros in the vector. Vectors with no zeros get a large additive boost
(`score += n^2`) and reflection bonuses that *multiply* the score; vectors with some zeros walk
their coordinates and take a *position- and count-dependent multiplicative* adjustment at each zero
— the first zero, the last zero, and the interior zeros scaled differently by factors involving `n`
and the zero's ordinal position (`n·0.5^{in_el}`) — and then stack reflection-pair matches
(`el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]`) as `×1.5` multiplicative factors on top. It is
branch-heavy, non-linear, and position-sensitive: nothing like the clean symmetric sum I wrote, and
nothing I would reach for, because its structure was selected for the score, not derived.

That immediately raises the question of *which* dimension it is for. The constants — the `×1.5`
factors, the `el[3]==el[-3]` term, the `0.5^{in_el}` decay — read like they were tuned at one
particular `n`, and if a search produced them by hammering on the `n = 8` evaluator, they encode the
structure of an eight-dimensional cap and nothing else. So before I trust any headline number I
should look at what this function actually does, and the cleanest place to start is small `n`, where
I can brute-check everything. My first guess at the mechanism is simple enough to state: the zero-free
vectors get the `+ n^2` boost, so they should sit far above everything with a zero, the greedy will
admit all `2^n` of them first, and — since the all-nonzero vectors `{1,2}^n` are themselves a cap (in
any coordinate, three values from `{1,2}` sum to `0 mod 3` only when all three are equal, which forces
the three points to coincide, so no line closes) — that cap is already complete and blocks every
remaining vector. If that picture is right, the function just rebuilds the trivial `2^n` cube at small
`n`, and its whole value is concentrated at the dimension it was tuned for.

Let me actually check the priority separation rather than assume it. Computing
`min` over zero-free vectors and `max` over vectors-with-a-zero at each `n`:

```
n=4: min(zero-free)=51.000  max(has-zero)=15.750   separated=True
n=5: min(zero-free)=35.000  max(has-zero)=24.375   separated=True
n=6: min(zero-free)=69.000  max(has-zero)=50.625   separated=True
n=7: min(zero-free)=63.000  max(has-zero)=107.188  separated=False
n=8: min(zero-free)=80.000  max(has-zero)=192.000  separated=False
```

So my clean hypothesis is only true through `n = 6`. At `n = 7` and `n = 8` it breaks: the
`0.5^{in_el}` decay is bounded, but a vector with several zeros can collect enough `×n` and `×1.5`
factors that some zeroed vectors outrank some zero-free ones. The greedy does *not* simply pour in all
the zero-free vectors first. So I cannot reason my way to the small-`n` sizes from the separation
argument — the ordering interleaves, and I have to run the construction and look.

Running the discovered priority through the skeleton and brute-verifying the result (`n ≤ 6`
cross-checked by the independent triple scan), with sizes and the membership breakdown:

```
n=4: size=16   valid=True   == {1,2}^4 cube? True    zero-free in cap=16    zeroed=0
n=5: size=32   valid=True   == {1,2}^5 cube? True    zero-free in cap=32    zeroed=0
n=6: size=64   valid=True   == {1,2}^6 cube? True    zero-free in cap=64    zeroed=0
n=7: size=128  valid=True   == {1,2}^7 cube? True    zero-free in cap=128   zeroed=0
n=8: size=512  valid=True   == {1,2}^8 cube? False   zero-free in cap=128   zeroed=384
```

This is sharper than the separation argument and partly against it. At `n = 7` the priorities
interleave — some zeroed vectors outscore some zero-free ones — yet the cap that comes out is still
*exactly* the `{1,2}^7` cube, all `128` zero-free vectors and not one zeroed vector. The interleaving
doesn't matter there: whichever zeroed vector the greedy reaches early gets blocked out as the cube
fills, so the fill collapses back to `2^n = 128`. The function carries no advantage at `n = 7`; it is
the trivial cap dressed up. The same collapse happens at `n = 4, 5, 6` for the cleaner reason that
the orderings there really are separated. So across `n = 4..7` the discovered function builds `16, 32,
64, 128` — the floor — and at those dimensions it is no better than the symmetric guess, plausibly
worse than random multi-start, exactly because it is the wrong tool for those spaces.

`n = 8` is where it does something the smaller dimensions cannot. The cap has `512` points and is a
verified valid cap, but the composition is the real news: only `128` of those points are zero-free,
and `384` of them have zeros. The function is not rebuilding the cube and stopping; it is admitting
the cube-like core and then *weaving in* `384` zeroed vectors that the small-`n` fills could never
keep. That is `4 ×` the trivial floor, and it is precisely the part my hand-designed priority could
not produce, because keeping those `384` zeroed vectors valid against each other and against the core
requires the exact position-and-count-dependent weighting the search found and I did not. So the
headline is `512` at `n = 8`, and now I can see *why* it is `512` and not `128`: the discovered
weights make a large family of zeroed vectors mutually compatible, where every order I would have
written tears that family apart.

I should be careful about one claim I cannot fully settle from here. Reaching `512` and verifying it
is a valid cap is solid — I computed it. Whether the `512` points I built coincide *as a set* with a
specific `512`-cap recorded elsewhere is a separate assertion, and I have no reference file in hand to
diff against. The membership profile I do have (`128` zero-free + `384` zeroed, a valid cap of size
`512`) is consistent with the known record-size construction at `n = 8`, and since the priority is
deterministic the object is reproducible; but I would want to actually diff the point sets before
claiming they are the *same* object rather than merely the same size. For now the defensible statement
is the computed one: this priority, through this skeleton, deterministically builds a verified valid
`512`-cap in `F_3^8`, four times the trivial floor and matching the best known size at that dimension,
improving on the previous best construction of `496`.

And here the ladder honestly ends. `512` is the size this construction reaches; reproducing it is the
payoff, not beating it. There is no rung above this that I should claim from a single constructor,
because going further would mean running the evolutionary search itself — millions of evaluated
candidates — to discover a *new* priority, which is a different computation entirely. I would expect
the same mechanism to extend (a companion function evolved at `n = 9` should reach the known-best
`1082` by the same weave-in-the-zeroed-vectors effect), but that is corroboration I would have to run
to confirm, not something I can assert from what I checked here. The ladder, then, is: a blind order
gives the `2^n` cube; many random orders give the best of a thin lottery; a hand-designed symmetric
priority lands in the same band and no higher — and across `n = 4..7` the discovered function lands
there too, because it is tuned away from those dimensions. Only at `n = 8`, where it was evolved, does
the priority do what derivation could not: keep `384` zeroed vectors alive on top of the cube to reach
`512`. The skeleton was right all along; the discovery was the function, and the function was the part
that search, not reasoning, had to supply.
