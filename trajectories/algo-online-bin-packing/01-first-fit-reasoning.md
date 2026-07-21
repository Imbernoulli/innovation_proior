I have a stream of items that I must place one at a time, each one for good before I see the next,
and I want to end up with as few bins as possible. The first thing I notice is how little freedom the
online setting leaves me. Offline, with the whole multiset in hand, I could sort the items and pack
cleverly — largest-first, matching big items to big holes and filling the seams with the small ones;
here I am blind to the future, so every placement is a bet against items I have not seen. That
immediately tells me the interesting decision is not *whether* to open bins but *which already-open
bin to drop the current item into*, because every time I am forced to open a fresh bin I have
implicitly admitted that the open bins could not absorb this item, and open bins with leftover room
are the only thing standing between me and the trivial policy of one bin per item. If I never reused
a bin I would use exactly 5000 of them; the entire game is how much of that I can claw back by reuse.

So before I pick any rule I want the two numbers that bound the game, because they tell me how much
room there is to compete over. The yardstick is the L1 lower bound, `LB = ceil(Σ items / C)`: every
bin holds at most `C`, so if the items sum to `S` I need at least `S / C` bins however cleverly I
pack. On the Weibull(45, 3) family the sizes have mean `45 · Γ(4/3) ≈ 40.18` (the rounding to
integers barely moves that, and the clip at 100 is irrelevant since a Weibull(45,3) exceeds 100 with
probability `≈ 1.7·10⁻⁵`), so a 5000-item stream sums to about `200{,}900` and `LB ≈ 2010`. The
OR-style family is uniform integer on `[20, 100]`, mean exactly `60`, so `Σ ≈ 300{,}000` and
`LB = ceil(300000/150) = 2000`. Two things jump out. Both families want on the order of two thousand
bins, so one-bin-per-item (5000) is already more than twice the floor — greedy reuse alone recovers
most of the distance, and any refinement fights over the last stretch. And `100/40.18 ≈ 2.49`,
`150/60 = 2.50`: both families ideally pack about two-and-a-half items per bin, so they are
geometrically similar at the level of "how many items share a bin."

That two-and-a-half tells me *why* even a careful policy sits a few percent above `LB`, and that gap
is what every rung competes to shrink. The ideal packing is a mixture of two-item and three-item
bins. Two average Weibull items sum to roughly `80.4`, leaving about `19.6` of dead space; that bin
only reaches full if a third, small item — one no larger than the leftover — happens to arrive while
the bin is still open and still the chosen destination. Weibull(45,3) puts a size below `20` with
probability `≈ 0.084`, so roughly one item in twelve is a potential "filler." The entire online
difficulty is catching those fillers and steering them into the bins that need them, before the bin
is stranded, without having seen them coming. A policy that lets fillers scatter into roomy bins
leaves a long tail of two-item bins each carrying ~20 units of permanent dead space, and that dead
space is the excess over `LB`. So the game is not about the big items, which mostly force their own
bins; it is about the disposition of the small ones. I keep that lens on as I read every rung.

The two families are not interchangeable through that lens. Weibull has a continuum of sizes running
down toward `1`, so its fillers come in every size and there is almost always *some* item small
enough to slot into whatever gap a bin has left. The OR-style family has *no* items below `20` at
all. A two-item OR bin holds about `120` of its `150`, leaving roughly `30`, which can only be topped
off by a third item in `[20, 30]` — a band that is just `11/81 ≈ 13.6%` of the distribution — and
three OR items average `180 > 150`, so a genuine three-item bin is rare. OR-style packing is a
coarser, chunkier problem; Weibull packing is fine-grained. A fit-aware rule could earn very
different margins on the two, so I read the two metric columns as two separate experiments.

Now I want the most conservative, most obviously-correct rule I can state, to floor everything else.
Keep the open bins in a fixed order, and put each item into the **first** bin in that order that
still has room; only if none do, open a new bin at the end. First-Fit. Its whole appeal is that it
never does anything clever and never anything stupid — it greedily reuses capacity in a deterministic
sweep, so it can never do worse than one-bin-per-item, and it reuses a bin whenever reuse is possible
at all. Any heuristic worth proposing has to beat "reuse the earliest bin you can," and I would
rather learn the value of cleverness against a floor I completely understand.

I should check I am not starting one rung too high. Below First-Fit sits Next-Fit: keep only the
*most recently opened* bin active, place the item there if it fits, otherwise close that bin
permanently and open a fresh one, never looking back. It is cheaper — one bin, not an array — but it
throws away reusable capacity. Capacity `10`, stream `[5, 6, 5]`: First-Fit puts the `5` in bin 0,
opens bin 1 for the `6`, then fits the second `5` back into bin 0, closing it exactly — two bins,
optimal (`LB = 2`). Next-Fit *seals bin 0* the moment the `6` does not fit, so the second `5` cannot
reach the `5` still free there and is forced into a third bin. The ability to reach *back* into an
older open bin is exactly the reuse property that recovers most of the distance to `LB`, and
First-Fit keeps every opened bin available. That makes First-Fit the right floor: the cheapest policy
that keeps full reuse.

Fitting it into the harness takes a little care, since the harness does not hand me "the open bins in
creation order." It hands me, for the current item, the sub-array of remaining capacities of the bins
that *can* fit it, and places the item in the highest-scoring one, breaking `argmax` ties toward the
lowest index. The bins array is in stable positional order (index `0` is the first bin ever opened),
and `get_valid_bin_indices` returns indices ascending, so `bins[valid]` preserves creation order —
"first bin that fits" is exactly "the valid bin at position `0`." A constant score would already pick
position `0` through the tie-break, but I would rather not rest correctness on a convention I have to
remember, so I hand back a strictly decreasing score, `−np.arange(len(bins))`, whose `argmax` is the
unambiguous position `0`.

The new-bin behaviour then falls out for free, with nothing special to code: the harness pre-fills
the array with still-full bins at capacity `C`, so if no already-used bin fits, the earliest valid
bin is the first still-full one — and taking it *is* opening a new bin. Bins fill left to right and
the still-full tail supplies fresh bins automatically; "open a new bin" is not a second rule bolted
on but the same "earliest valid bin" rule reaching into the tail.

I want to be honest about *why* First-Fit will not be very good, because that diagnosis is what the
next rung must attack, and I would rather see it in a traced example. The problem is the word "first."
When several open bins all fit, First-Fit takes the earliest by accident of creation order, with no
regard for the slack it leaves. Capacity `10`, stream `[2, 9, 1, 8]`. The `2` opens bin 0 (rem `8`);
the `9` does not fit, so it opens bin 1 (rem `1`). Now the `1` arrives with both bins valid, and
First-Fit takes the earliest, bin 0, dropping the `1` into the roomy bin and leaving `[7, 1]` — it
spent a big pocket of capacity on a tiny item and *failed to top off* bin 1, which the `1` would have
closed exactly. Then the `8` fits neither (`7 < 8`, `1 < 8`), forcing bin 2: three bins for a stream
whose sum is `20`, `LB = 2`. Had the `1` gone into bin 1, the `8` fits bin 0 and the whole stream
packs into two. Note the end-state `[7, 1, 2]`: the *oldest* bin is the *roomiest*, so "earliest" and
"tightest" genuinely diverge under First-Fit's own dynamics — there is real daylight for the next
rung to win, not a relabelling of the same choice. This is the filler lens made concrete: First-Fit
does not merely fail to steer fillers, it actively mis-routes them into roomy old bins and strands the
near-full bins they were meant to complete. Over 5000 items that is a long tail of partially-filled
bins that never get topped off.

The rule is deterministic given the stream, so across the five seeds the only variation is the item
draws — and when I later compare rungs on the *same* seeds that variability cancels, leaving a clean
difference of rule. Greedy reuse recovers most of the distance from 5000 toward the ~2010/2000 floor;
the hole-management sloppiness I traced is a persistent leak on top, so I expect a low-single-digit
percent margin above `LB` on both families. The exact height I cannot compute by hand — and since
these numbers rest on a simulator I have taken on faith, the cheapest confidence check is that it
reproduces First-Fit's known figures on a reference instance to the digit before I trust my own
seeded streams.

The weakness I traced — First-Fit ignores the slack a placement leaves, squanders roomy bins on small
items, and fails to finish off tight bins — is exactly what the next rung fixes: instead of the
*earliest* valid bin, take the one whose *fit is tightest*, so every placement wastes the least
capacity and tight bins get closed off. That is the move to Best-Fit, and it is where I go next.
