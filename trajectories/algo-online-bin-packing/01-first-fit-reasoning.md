I have a stream of items that I must place one at a time, each one for good before I see the next,
and I want to end up with as few bins as possible. The first thing I notice is how little freedom the
online setting leaves me. Offline, with the whole multiset in hand, I could sort the items and pack
cleverly; here I am blind to the future, so every placement is a bet against items I have not seen.
That immediately tells me the interesting decision is not *whether* to open bins but *which already-
open bin to drop the current item into*, because every time I am forced to open a fresh bin I have
implicitly admitted that the open bins could not absorb this item, and open bins with leftover room
are the only thing standing between me and the trivial policy of one bin per item.

So let me start from the most conservative, most obviously-correct rule I can state, the one whose
behaviour I can fully predict and which gives me a floor to measure everything else against. The rule
is: keep the open bins in a fixed order, and put each item into the **first** bin in that order that
still has room for it; only if none do, open a new bin at the end. First-Fit. Its whole appeal is
that it never does anything clever and never does anything stupid — it greedily reuses capacity in a
deterministic sweep, so it can never do worse than one-bin-per-item, and it will reuse a bin whenever
reuse is possible at all. It is the natural baseline because any heuristic worth proposing has to beat
the policy that just says "reuse the earliest bin you can."

I want to be honest about *why* First-Fit will not be very good, because that diagnosis is what the
next rung has to attack. The problem is the word "first." When several open bins could all fit the
item, First-Fit takes the earliest one purely by accident of bin-creation order, with no regard for
how much room it leaves behind. Picture a bin with `90` units of slack and a bin with `12` units of
slack, and an item of size `10`. First-Fit, if the roomy bin happens to come earlier in the order,
drops the `10` into the bin that had `90` left, leaving `80` — and it has just spent a big, valuable
pocket of capacity on a small item that the tight bin could have swallowed almost perfectly, closing
that tight bin off near-full. The tight bins are exactly the ones I most want to finish off, because a
bin filled to `99` out of `100` is a bin I will essentially never need to open another for; a bin left
at `80` is a half-used resource I now have to hope future items fit into. First-Fit systematically
spreads items across the early bins and leaves a long tail of partially-filled bins that never get
topped off. That is wasted capacity, and wasted capacity is extra bins.

Now I have to fit this rule into the harness, which is a little awkward, because the harness does not
hand me "the open bins in creation order with their fill levels." It hands me, for the current item,
only the sub-array of remaining capacities of the bins that *can* fit the item — the valid bins — and
asks me to score them, and it places the item in the highest-scoring one, breaking ties toward the
lowest index. The bins array it maintains is in a fixed positional order (index `0` is the first bin
ever, and so on), and crucially that order is stable across the whole run, so "first bin that fits"
is exactly "the valid bin with the smallest index." Since the harness already breaks `argmax` ties by
the lowest index, all I have to do to realise First-Fit is hand back a score that is *constant*
across the valid bins — every valid bin tied — and let the tie-break pick the earliest one. Or,
equivalently and more transparently, hand back a score that strictly decreases with the position in
the valid sub-array, so the earliest valid bin is unambiguously the maximum. Either way the content
of the rule is "ignore how good the fit is; take the earliest bin that works." I will write it as a
decreasing function of index so the intent is unmistakable and does not lean on the tie-break
convention.

There is one thing I want to make sure I am not getting wrong. The harness pre-allocates a huge bins
array, all sitting at full capacity, so "open a new bin" is not a special branch I have to code — if
no already-used bin fits the item, the only valid bins left are still-full ones, and First-Fit will
take the earliest still-full bin, which *is* opening a new bin. Good: the new-bin behaviour falls out
for free, and I do not have to distinguish "reuse" from "open" in the score at all. The rule is
genuinely just "earliest valid bin."

So I expect this to land a few percent above the L1 lower bound. The lower bound assumes every bin is
packed to the brim with zero waste, which is unreachable online, and First-Fit's careless spreading
of items across roomy bins should leave it visibly short of even a *careful* greedy. I do not expect
it to be catastrophic — greedy reuse is most of the battle, and First-Fit does reuse — but I expect a
clear margin of waste that a fit-aware rule could claw back. The specific weakness I have named, that
First-Fit ignores the slack a placement leaves and so squanders roomy bins on small items and fails
to finish off tight bins, is the exact thing the next rung must fix: instead of taking the *earliest*
valid bin, take the one whose *fit is tightest*, so that every placement wastes as little capacity as
possible and tight bins get closed off. That is the move from First-Fit to Best-Fit, and it is where I
go next.
