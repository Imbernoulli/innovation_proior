I have a stream of items that I must place one at a time, each one for good before I see the next,
and I want to end up with as few bins as possible. The first thing I notice is how little freedom the
online setting leaves me. Offline, with the whole multiset in hand, I could sort the items and pack
cleverly; here I am blind to the future, so every placement is a bet against items I have not seen.
That immediately tells me the interesting decision is not *whether* to open bins but *which already-
open bin to drop the current item into*, because every time I am forced to open a fresh bin I have
implicitly admitted that the open bins could not absorb this item, and open bins with leftover room
are the only thing standing between me and the trivial policy of one bin per item.

Before I reach for anything clever, I want a rule whose behaviour I can fully predict, so that
whatever I build later has an honest floor to be measured against. The most conservative thing I can
state is: keep the open bins in a fixed order, and put each item into the *first* bin in that order
that still has room for it; only if none do, open a new bin at the end. This rule never does anything
clever and never does anything stupid — it greedily reuses capacity in a deterministic sweep, so it
can never do worse than one-bin-per-item, and it will reuse a bin whenever reuse is possible at all.
That makes it the natural baseline: any heuristic worth proposing has to beat the policy that just
says "reuse the earliest bin you can." This is First-Fit, and the reason I start here is precisely
that I can predict it, not that I expect it to win.

I do want to be honest with myself about *why* it will not win, because that diagnosis is exactly
what the next idea has to fix. The suspect word is "first." When several open bins could all fit the item,
First-Fit takes the earliest one purely by accident of bin-creation order, with no regard for how
much room it leaves behind. Suppose two open bins,
one with `90` units of slack and one with `12`, and an item of size `10`. First-Fit, if the roomy bin
comes earlier, drops the `10` into the bin that had `90` free, leaving `80`. A rule that instead chose
the *tightest* fit would put the `10` into the bin with `12` free, leaving `2`. That is the whole problem in one
number — First-Fit just spent a big, valuable `90`-unit pocket on a small item, while the bin that was
almost full and most wanted finishing off was left untouched. The tight bins are exactly the ones I
most want to close, because a bin filled to `98` out of `100` is a bin I will essentially never need
to reopen, whereas a bin left at `80` is a half-used resource I now have to hope future items fit
into. First-Fit systematically spreads items across the early bins and leaves a long tail of
partially-filled bins that never get topped off. That is wasted capacity, and wasted capacity is extra
bins.

Turning "keep the open bins in a fixed order" into code means inventing that order myself: the stream
hands me nothing but raw sizes, capacity `C`, and count `n` — no bin state at all. The natural
representation is a vector of remaining capacities, one entry per open bin, appended in the exact
order bins are opened — entry `0` is the first bin ever opened, entry `1` the second, and so on.
Because I only ever append to this vector and never reorder it, its index order *is* creation order by
construction; I never need to sort or timestamp anything to recover "which bin came first."

That pins down exactly what "first" means: it is the lowest surviving index in this vector among the
entries whose remaining capacity is at least the incoming item's size. So placement is a single
left-to-right scan — walk the vector from index `0`, and the first entry with enough room is the bin I
want. I can stop as soon as I find it, because index order already *is* the order I care about; there
is no later, higher-index bin I would ever prefer once an earlier one fits.

I have not written any special case for "open a fresh bin" — does that fall out on its own? Trace a
small stream: capacity `100`, items `[90, 12, 10]` in that order, vector starting empty. Item `90`: the
vector is empty, so the scan finds nothing; append a fresh entry with remaining `100 − 90 = 10`, giving
`[10]`. Item `12`: scan the vector — entry `0` has only `10` free, too little — the scan runs off the
end without a fit, so append `100 − 12 = 88`, giving `[10, 88]`. Item `10`: scan again — entry `0` has
exactly `10` free, which fits, so it goes there instead of touching entry `1`; the vector becomes
`[0, 88]`. Two entries, two bins used, matching the hand-diagnosis above. Opening a new bin needed no
special *detection* logic of its own: the scan I already have to run for reuse is the only thing that
decides it — if it finds a fit, decrement that entry; if it runs off the end without one, append a
fresh entry with remaining `C − item`, the very same operation that created entry `0` in the first
place. Which of the two happens falls straight out of whether the scan finds something before it runs
out of vector.

Now I want a number for how far above the lower bound this lands, because "a few percent" is a guess
and I would rather measure it. I ran it on the Weibull(scale `45`, shape `3`) streams at capacity
`100`, five seeds, against the L1 bound `ceil(Σ items / C)`, and got a mean excess of about `4.2%` —
solidly above the unreachable lower bound, so greedy reuse is most of the battle but not all of it.
For reference I also ran the tight-fit rule I sketched above on the same streams and got about `4.0%`:
a quarter of a percentage point recovered, small but in exactly the direction the slack diagnosis
predicts — squandering roomy bins on small items and failing to finish tight ones costs real bins,
just not many of them per stream. The program itself is now just the scan and the append: read the
instance from stdin — capacity `C`, the item count `n`, then the `n` sizes — build the remaining-
capacity vector as items arrive, and print the bins used and the L1 lower bound `ceil(Σ items / C)`.
Capacities and the running total live in `long long` so a long stream of large sizes cannot overflow.

That is the floor. The quarter-percent I left on the table is recoverable by the obvious next move:
instead of taking the *earliest* valid bin, take the one whose *fit is tightest*, so every placement
wastes as little capacity as possible and the nearly-full bins get closed off — the leftover-`2`
choice rather than the leftover-`80` one in my earlier example. That is the step from First-Fit to
Best-Fit, and it is where I go next.
