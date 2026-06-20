Best-Fit left me with a sharp constraint, and I want to take it seriously because it tells me exactly
what *cannot* work. Best-Fit scores each bin as a function of that bin's own leftover slack `r − s`,
and I confirmed the brutal consequence: any rule that is a monotone reshaping of a single bin's slack
makes the identical choices. Squaring the slack, exponentiating it, adding a tiny tie-break for how
full the bin already is — I tried these and the bin counts did not move one unit. So I am not going to
beat Best-Fit by being cleverer about a single hole. Whatever I do has to make a bin's score depend on
the *other* bins — on where this bin sits in the landscape of all the open bins — so that the same
leftover slack can be ranked differently depending on context. That is the only door left open.

Let me think about what context could possibly matter. The harness hands me the remaining capacities
of all the valid bins at once, as an array, in a stable positional order. Best-Fit throws away
everything about that array except each entry's own value. What is the cheapest, most natural piece of
*relational* information I am currently ignoring? The most salient landmark in the array is the
**emptiest** valid bin — the one with the largest remaining capacity. A still-full, freshly-opened bin
is the extreme case of this: it sits at the top, at capacity. Every other bin's remaining capacity can
be read as a *distance below* that emptiest bin: `(r − r_max)`, a non-positive number that is `0` for
the emptiest bin and grows more negative the more a bin has already been filled. That distance is a
genuinely relational quantity — it depends on the whole array, not just one entry — and it has an
intuitive meaning: how much *more committed* this bin already is than the freshest one. A bin far below
the emptiest is a bin I have already invested in.

Why would I want to prefer the bins that are far below the emptiest? Because those are the bins I am
trying to *finish*. Best-Fit's residual waste was the long tail of bins stuck a few units short of
full. If I can bias the policy toward pouring items into the already-committed bins — driving them the
rest of the way to full — rather than touching the freshest ones, I consolidate the committed bins
faster and keep the fresh capacity in reserve. So my relational signal should *reward* a large
distance below the emptiest. The square of that distance, `(r − r_max)²`, is a clean way to write it:
it is `0` at the emptiest bin and grows quickly for the committed ones, so it sharply separates "fresh"
from "committed."

But raw distance is not the whole story, because the item size has to enter. The same `(r − r_max)`
means something very different for a tiny item than for a large one: a large item poured into a
committed bin makes a big difference to closing it, while a tiny item barely advances it. So I want to
weight the distance by how *consequential* the item is relative to the bin, and the simplest
dimensionally-natural way is to divide by the item size: `(r − r_max)² / s`. Now small items, which
are the ones Best-Fit is happy to scatter, get a *gentler* relational pull, and the policy's relational
preference kicks in most strongly exactly where it should. I will still respect feasibility — among
bins that actually fit the item I do not want to send it somewhere that overflows — so I keep the sign
convention that valid bins (those with `r ≥ s`) are the ones in play and I negate the score on them so
that, after the relational term, a tighter genuinely-fitting bin is still preferred. This is fiddly
and I am not fully certain of the sign bookkeeping yet; I will lean on the measured bin count to tell
me whether the construction actually consolidates or accidentally scatters.

Here is the piece I almost missed, and it turns out to be the whole engine. A score that is *per-bin*,
even a relational one like `(r − r_max)² / s`, still gets fed to `argmax` one entry at a time, and I
worried it would collapse back into a Best-Fit-like ordering. What rescues it is making the score of a
bin depend on its **neighbour** in the array: replace each bin's score by how much it *exceeds the bin
before it*, i.e. difference the score array along its order, `score[i] ← score[i] − score[i−1]`. This
is a strange-looking operation and I want to be honest that I am reaching for it somewhat empirically,
but its meaning is defensible: it scores a bin not by its absolute relational value but by the *jump*
in relational value from its predecessor, so a bin only wins if it is a local improvement over its
neighbour. This couples the bins together — no bin's score is decided in isolation anymore, which is
exactly the property I argued was necessary — and it breaks the monotone-collapse that pinned every
single-bin rule to Best-Fit. I cannot fully predict its effect by hand; the differencing makes the
selected bin depend on the array's local structure in a way that I will have to *measure* rather than
derive. My honest expectation is that it could go either way, and the only way to know is to run it.

So the rule I will test is: among valid bins, score each one `(r − r_max)² / s`, flip the sign on the
fitting bins so feasibility is respected, then difference the array so each bin is judged against its
neighbour. It is a single hand-built scoring function, but unlike Best-Fit it reads the whole array —
the emptiest-bin landmark, the item-relative weighting, and the neighbour differencing all make a
bin's fate depend on its context. If my reasoning about consolidation is right, this should pull items
into the already-committed bins and finish them off faster, cutting into Best-Fit's long tail of
nearly-full-but-not-quite bins. If the differencing misbehaves, the bin count will tell me immediately.
What this rung deliberately leaves on the table is the *form* of the relational score: I chose
`(r − r_max)² / s` and a plain neighbour difference by hand, from intuition, and I have no reason to
believe these specific powers and this specific coupling are optimal — searching over that functional
form, rather than guessing it, is what the final rung is for.
