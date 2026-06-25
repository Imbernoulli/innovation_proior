I have a principled starting matrix — the symmetric Jacobsthal design `Q + I` for the prime `29`,
which gives multiplier `49`. Before I do anything clever I want to know exactly how stuck I am, so
I measure all `841` single-entry flips against the seed's exact determinant. None of them raises
`|det|`: zero of `841`. So `Q + I` is a strict local maximum under the one move I have — flip one
entry's sign — and any greedy hill-climb from here terminates on its first step, at `49`.

But the same sweep tells me something the bare "local maximum" label hides, and it matters for what
comes next. Of the `841` flips, `435` leave `|det|` *exactly* unchanged, and of the rest the
steepest descent is mild: the worst single flip only drops the multiplier from `49` to `42`, a
change in `log|det|` of about `−0.15`, with the median worsening flip far smaller than that. So this
is not a sharp spike I am perched on; it is a broad flat plateau, ringed by gentle downhill slopes,
with no uphill anywhere on it. That changes my read of the difficulty. The problem is not that the
walls out of the basin are tall — they are shallow. The problem is purely that *no* local move
points up, so greed has nothing to follow. To get anywhere I have to be willing to walk downhill on
purpose — accept moves that make `|det|` temporarily worse — and the plateau measurement says the
downhill steps I would have to swallow are small, which is encouraging: a search that tolerates
modest worsening should drift off this plateau without needing to pay much.

The tool built for exactly this is simulated annealing. Keep the single-entry flip as the move, but
change the acceptance rule. Propose flipping a random entry; if it improves `|det|`, take it; if it
worsens `|det|`, take it anyway with a probability that depends on how much worse and on a
temperature I cool over time. Early, when the temperature is high, downhill moves are accepted often
and the search wanders freely; late, when the temperature is low, only improving moves survive and
the search settles into whatever basin it has reached. The bet is that with enough wandering it
finds a basin deeper than `49` before it freezes.

Two design decisions need care, and both come from the geometry of this particular objective.
First, what quantity should the acceptance rule compare? The raw determinant is astronomically
large — multiplier times `2^28 · 7^12`, and at `m = 49` that exact integer is
`182{,}059{,}119{,}829{,}942{,}534{,}144`, twenty-one digits — and its *changes* under a single
flip are enormous in absolute terms while modest in relative terms. If I anneal on the raw
difference `|det'| − |det|`, the temperature would have to span twenty orders of magnitude. The
natural scale is multiplicative: a flip multiplies the determinant by some ratio, and what I care
about is whether that ratio is near `1`, above it, or below it. So I should anneal on `log|det|`. Let
me put numbers on the step sizes that implies. A flip from `m = 49` to `m = 48` is
`log 48 − log 49 = −0.0206`; the worst flip off the seed plateau, `49 → 42`, is `log 42 − log 49 =
−0.154`; and a flip that doubled the determinant would be `log 2 = +0.693`. So on the log scale the
steps I will actually see span roughly `O(0.01)` to `O(1)`, and a temperature of order `0.01–0.1`
sits right inside that band — the schedule becomes tunable, which it was not on the raw scale.

I can sanity-check that a temperature in that band does the two things I need it to. With a warm
`T = 0.06`, the Metropolis probability of accepting the worst seed-flip (`Δ = −0.154`) is
`exp(−0.154 / 0.06) = 0.077`, and a mild `−0.02` flip is accepted with probability
`exp(−0.02 / 0.06) = 0.72` — so at the warm start the search readily takes the small downhill steps
that the plateau measurement said were all I needed, while still rarely taking the steepest ones.
Drop the temperature to a floor of `2 × 10⁻⁴` and the same mild `−0.02` flip is accepted with
probability `exp(−0.02 / 2e-4) = 4 × 10⁻⁴⁴` — effectively zero, i.e. the rule has become pure greedy
hill-climbing. That is the behaviour I want from the two ends of the schedule, and the arithmetic
confirms a single geometric cooling between those two temperatures spans it.

Second, how do I evaluate a candidate flip? The honest, exact thing is the Bareiss integer
determinant, but that is `O(n³)` big-integer arithmetic per candidate and I will propose tens of
thousands of candidates. I do not need exactness *during* the search — I only need a faithful
ranking of which configurations have larger `|det|`, and I will compute the final answer's
determinant exactly. So inside the loop I use floating-point `log|det|` via a sign-and-logdet
routine (`slogdet`): one LU factorization, `O(n³)` floats, fast and accurate enough to rank
configurations at order `29`. It is not clever, but it is correct, and at `n = 29` a float
factorization is cheap — tens of thousands of evaluated flips per second.

One thing I should check before trusting the seed choice: is starting from `Q + I` actually worth
it, or could I just start from a random sign matrix and let annealing do everything? I sample a
couple thousand random `±1` matrices and measure their multipliers. The median is about `0.0002`,
the mean about `0.0004`, and not one of the two thousand reaches even `m = 1`, let alone `49`. So a
random start sits roughly `250{,}000×` below the Jacobsthal seed in `|det|`; annealing from there
would spend its whole budget clawing up out of the random-sign swamp before it even reached the
structured region the seed begins in. Seeding at `Q + I` means the search spends its flips improving
an already-good configuration instead of first manufacturing one.

So the method assembles: seed at the Jacobsthal `Q + I`; set a warm starting `log`-temperature;
propose random single-entry flips; accept on the Metropolis rule in `log|det|`; cool geometrically
toward a small floor; keep the best matrix ever seen; at the end recompute its determinant exactly
with Bareiss and report the multiplier. Picking the schedule numbers from the band above — start
`T = 0.06`, decay geometrically to floor `2e-4` over `40k` flips — I run it once from a fixed seed
and read the exact result. The best matrix has `|det| = 556{,}857{,}955{,}547{,}919{,}417{,}344`,
multiplier `149.87`, score `0.438`. That is a `3.1×` increase in `|det|` over the Jacobsthal
baseline and `+0.295` in score, and it confirms the central claim concretely: accepting downhill
moves did carry the search off a plateau that greed could not leave, from `m = 49` to `m ≈ 150`,
roughly three-quarters of the way from the baseline toward the LLM-evolution frontier near `197`.

What I should be honest about is the ceiling of *this* move set. A single-entry flip is a local
move; annealing with local moves explores one connected basin structure and tends to plateau once
the temperature is low and the remaining improvements require coordinated multi-entry changes that a
single flip cannot make in one step. The run reaching `~150` and not the record `320` is consistent
with that: the limit is not the idea but the budget, because each flip costs a full `slogdet` and I
can only afford tens of thousands of them from one seed. That is the natural opening for what comes
next — if the only thing holding me back is the per-flip cost, then making each flip dramatically
cheaper lets me run far longer, restart from many seeds, and push the multiplier up the part of the
curve this method can only start.
