I have a stream of items that I must place one at a time, each one for good before I see the next,
and I want to end up with as few bins as possible. The first thing I notice is how little freedom the
online setting leaves me. Offline, with the whole multiset in hand, I could sort the items and pack
cleverly — largest-first, matching big items to big holes and filling the seams with the small ones;
here I am blind to the future, so every placement is a bet against items I have not seen. That
immediately tells me the interesting decision is not *whether* to open bins but *which already-open
bin to drop the current item into*, because every time I am forced to open a fresh bin I have
implicitly admitted that the open bins could not absorb this item, and open bins with leftover room
are the only thing standing between me and the trivial policy of one bin per item. If I never reused
a bin I would use exactly 5000 of them; the entire game is how much of that I can claw back by reuse,
and how carefully.

So before I pick any rule I want to know the two numbers that bound the game, because they tell me how
much room there even is to compete over. The yardstick the harness reports against is the L1 lower
bound, `LB = ceil(Σ items / C)`. Its logic is airtight: every bin holds at most `C` units, so if the
items sum to `S` I need at least `S / C` bins no matter how cleverly I pack, and since the count is an
integer I round up. No policy, online or offline, can beat `LB`; a heuristic at `0%` excess has packed
with literally zero wasted capacity. Let me actually compute what `LB` should be on the two stream
families, because that turns the abstract "few bins" into a concrete target. The Weibull(45, 3) family
draws sizes with mean `45 · Γ(1 + 1/3) = 45 · Γ(4/3) ≈ 45 · 0.8930 ≈ 40.18`; the rounding to integers
in `[1, 100]` barely moves that, and the clip at 100 is irrelevant since a Weibull(45,3) exceeds 100
with probability `exp(−(100/45)³) ≈ 1.7·10⁻⁵`. So the 5000-item stream sums to about `200{,}900`, and
`LB ≈ ceil(200900 / 100) ≈ 2010` bins. The OR-style family is uniform integer on `[20, 100]`, mean
exactly `60`, so `Σ ≈ 300{,}000` and `LB = ceil(300000 / 150) = 2000` bins. Two things jump out. First,
both families want on the order of two thousand bins, so the trivial one-bin-per-item policy (5000
bins) is already more than twice the floor — greedy reuse alone recovers most of the distance, and
whatever refinement I add is fighting over the last stretch. Second, `100 / 40.18 ≈ 2.49` and
`150 / 60 = 2.50`: both families ideally pack about two-and-a-half items per bin. That near-coincidence
is worth holding onto, because it means the two families are geometrically similar at the level of "how
many items share a bin," and a rule that works on one has a fair chance on the other.

It is worth being precise about *why* that floor sits a few percent above `LB` even for a careful
policy, because that gap is the thing every rung is competing to shrink, and if I understand its shape
I will read the numbers better. With about two-and-a-half items per bin, the ideal packing is a mixture
of two-item and three-item bins. Two average Weibull items sum to roughly `80.4`, leaving about `19.6`
of dead space; that bin only reaches full if a *third*, small item — one no larger than the leftover —
happens to arrive while the bin is still open and still the chosen destination. Weibull(45,3) puts a
size below `20` with probability `1 − exp(−(20/45)³) ≈ 0.084`, so roughly one item in twelve is a
potential "filler" that can top off a two-item bin. The entire online difficulty is catching those
fillers and steering them into the bins that need them, *before* the bin is stranded, without having
seen them coming. A policy that lets fillers scatter into roomy bins leaves a long tail of two-item
bins each carrying ~20 units of permanent dead space, and that dead space is exactly the excess over
`LB`. So the game is not really about the big items — those mostly force their own bins regardless —
it is about the disposition of the small ones. I will keep that lens on as I read every rung.

The two families are not interchangeable through that lens, and the difference is worth pinning down
now so I do not assume a rule that helps one helps the other equally. The Weibull family has a
continuum of sizes running down toward `1`, so its fillers come in every size and there is almost
always *some* item small enough to slot into whatever gap a bin has left. The OR-style family is
uniform on `[20, 100]` — it has *no* items below `20` at all. A two-item OR bin holds about `120` of
its `150`, leaving roughly `30`, which can only be topped off by a third item in `[20, 30]`, a band
that occupies just `11/81 ≈ 13.6%` of the distribution; and three OR items average `180 > 150`, so a
genuine three-item bin is rare. So OR-style packing is a coarser, chunkier problem — fewer, larger,
harder-to-place fillers over a floor of `20` — while Weibull packing is fine-grained. That structural
difference means a fit-aware rule could plausibly earn very different margins on the two families, and I
should read the two metric columns as two separate experiments rather than one. I will not be surprised
if the numbers diverge.

Now I want the most conservative, most obviously-correct rule I can state, the one whose behaviour I
can fully predict and which gives me a floor to measure everything else against. The rule is: keep the
open bins in a fixed order, and put each item into the **first** bin in that order that still has room
for it; only if none do, open a new bin at the end. First-Fit. Its whole appeal is that it never does
anything clever and never does anything stupid — it greedily reuses capacity in a deterministic sweep,
so it can never do worse than one-bin-per-item, and it will reuse a bin whenever reuse is possible at
all. That last property is not nothing: a policy that ever opened a new bin while an old one could
still take the item would be strictly wasteful, and First-Fit is immune to that failure by
construction. It is the natural baseline precisely because any heuristic worth proposing has to beat
the policy that just says "reuse the earliest bin you can," and I would rather learn the value of
cleverness by measuring it against a floor I completely understand than against another guess.

Before I settle on First-Fit as *the* floor, I should look at the even-simpler policies below it, so I
know I am not starting the ladder one rung too high. The trivial one is one-bin-per-item: never reuse,
5000 bins, more than twice the ~2010/2000 ideal. That is a floor beneath the floor, useful only to
confirm that reuse is where all the value is; I will not run it, because I already know its number
exactly. The interesting simpler alternative is Next-Fit: keep only the *most recently opened* bin
active, place the item there if it fits, and otherwise close that bin permanently and open a fresh one,
never looking back. Next-Fit is even cheaper than First-Fit — it holds one bin, not an array — and it
still does reuse, so I should check whether it is a good enough floor to start from. It is not, and I
can show it costs bins that First-Fit saves. Take capacity `10` and the stream `[5, 6, 5]`. First-Fit
puts the `5` in bin 0 (remaining `5`), cannot fit the `6` there so opens bin 1 (remaining `4`), and
then fits the second `5` back into bin 0 (`5 ≥ 5`, closing it exactly) — two bins, and `LB = ceil(16/10)
= 2`, optimal. Next-Fit puts the `5` in bin 0, cannot fit the `6` so *closes bin 0 forever* and opens
bin 1 (remaining `4`), and then the second `5` does not fit the active bin 1 (`4 < 5`) — but bin 0,
which had exactly `5` left, has been sealed off, so Next-Fit is forced to open bin 2. Three bins for a
two-bin stream. The lesson is sharp: Next-Fit's premature bin-closing throws away reusable capacity, and
the ability to reach *back* into an older open bin is precisely the greedy-reuse property that recovers
most of the distance to `LB`. First-Fit keeps every opened bin available and never closes one that
could still be used, so it dominates Next-Fit by construction. That makes First-Fit the right floor: the
cheapest policy that keeps the full reuse property, and the honest bar the fit-aware rules must clear.

Fitting this into the harness is a little awkward, and I want to get the mechanics exactly right,
because the whole point of the baseline is that it is not approximate. The harness does not hand me
"the open bins in creation order with their fill levels." It hands me, for the current item, only the
sub-array of remaining capacities of the bins that *can* fit the item — the valid bins — and asks me
to score them; it then places the item in the highest-scoring one, breaking `argmax` ties toward the
lowest index. The bins array it maintains is in a fixed positional order (index `0` is the first bin
ever opened, and so on), and that order is stable across the whole run. Crucially,
`get_valid_bin_indices` returns those indices in ascending order, and `bins[valid]` therefore preserves
creation order inside the sub-array, so "first bin that fits" is exactly "the valid bin at position `0`
of the sub-array." I have two clean ways to select it. I could hand back a *constant* score across the
valid bins — every valid bin tied — and lean on the tie-break to pick position `0`. Or I could hand
back a score that *strictly decreases* with position, `−[0, 1, 2, …]`, so position `0` is the
unambiguous maximum and I do not lean on the tie-break convention at all. I prefer the second: it makes
the intent unmistakable, and it is robust even if I misremember which way ties break. Concretely
`−np.arange(len(bins))` gives scores `[−0, −1, −2, …]`, whose `argmax` is index `0` — I checked this is
the earliest valid position, and it is. The content of the rule is "ignore how good the fit is; take
the earliest bin that works," and this realises it exactly.

There is one thing I want to make sure I am not getting wrong, because if I am, First-Fit silently
becomes something else: the new-bin behaviour. The harness pre-allocates a huge bins array, all sitting
at full capacity `C`, so "open a new bin" is not a special branch I have to code — if no already-used
bin fits the item, the only valid bins left are still-full ones, and First-Fit will take the earliest
still-full bin, which *is* opening a new bin. Let me verify this actually falls out rather than merely
assert it. Take capacity `10` and the stream `[6, 3, 5, 4, 2, 7]`. The `6` finds no used bin, so the
earliest valid bin is the first still-full one, bin 0, leaving remaining `[4]`. The `3` fits bin 0
(`4 ≥ 3`), leaving `[1]`. The `5` does not fit bin 0 (`1 < 5`), so the earliest valid bin is the next
still-full one, bin 1, leaving `[1, 5]`. The `4` skips bin 0 (`1 < 4`) and fits bin 1 (`5 ≥ 4`), giving
`[1, 1]`. The `2` fits neither used bin (`1 < 2` twice), so it opens bin 2, `[1, 1, 8]`. The `7` fits
only bin 2 (`8 ≥ 7`), giving `[1, 1, 1]`. Three bins. The sum is `27`, so `LB = ceil(27/10) = 3`, and
First-Fit is optimal on this little stream — which is the point: bins fill left to right, the still-full
tail supplies fresh bins automatically, and "open a new bin" needs no special case at all. The rule is
genuinely just "earliest valid bin," and the new-bin policy is not a second rule bolted on but the same
rule reaching into the still-full tail.

I want to be honest about *why* First-Fit will not be very good, because that diagnosis is what the
next rung has to attack, and I would rather see the failure in a concrete traced example than wave at
it. The problem is the word "first." When several open bins could all fit the item, First-Fit takes the
earliest one purely by accident of bin-creation order, with no regard for how much room it leaves
behind. Let me construct the smallest stream I can where this actually costs a bin, so I am not
theorising. Capacity `10`, stream `[2, 9, 1, 8]`. The `2` opens bin 0, remaining `8`. The `9` does not
fit bin 0 (`8 < 9`), so it opens bin 1, remaining `1`. Now the `1` arrives, and both bins are valid:
bin 0 has `8` left, bin 1 has `1` left. First-Fit takes the *earliest*, bin 0, dropping the `1` into the
roomy bin and leaving `[7, 1]`. Watch what that just did: it spent a big, valuable pocket of capacity
on a tiny item, and it *failed to top off* bin 1, which the `1` would have closed to exactly full. Then
the `8` arrives; bin 0 has only `7` left and bin 1 has `1`, so neither fits, and First-Fit is forced to
open bin 2 — three bins for a stream whose sum is `20`, i.e. `LB = 2`. Had the `1` gone into bin 1
instead (the snug fit, closing it exactly), bin 0 would still hold `8`, the `8` would fit it, and the
whole stream packs into two bins. So the tight, nearly-full bins are exactly the ones I most want to
finish off — a bin filled to `10/10` is a bin I will never open another for — and First-Fit's
indifference to fit quality lets it squander a roomy bin on a small item while leaving a tight bin
un-closed. Over one decision that is one wasted bin; over 5000 items it is a long tail of
partially-filled bins that never get topped off, and wasted capacity is extra bins.

This lines up exactly with the filler lens I set up earlier, and the two together sharpen the
diagnosis. I argued the whole online game turns on steering the small filler items into the two-item
bins that need topping off. First-Fit does the opposite of steering: when a filler arrives it drops it
into the *earliest* bin that happens to fit, which — as the `[7, 1, 2]` end-state shows — can easily be
a roomy old bin rather than the near-full bin one slot over that the filler would have closed. So
First-Fit is not neutral about fillers; it actively mis-routes them, spending them where they do the
least good and stranding the bins they were meant to complete. That is the mechanism behind the
few-percent excess made concrete: not random bad luck, but a systematic misallocation of exactly the
items the distribution analysis flagged as pivotal. A rule that instead sent each item to the *tightest*
fitting bin would, almost by definition, send a filler to the bin it completes — which is why the fix
names itself.

I also want to be sure the realisation itself is not resting on a convention I have misremembered. The
harness breaks `argmax` ties toward the lowest index, and `numpy.argmax` returns the first position of
the maximum, so a *constant* score across the valid bins would already pick position `0` — the earliest
valid bin — and realise First-Fit. But I would rather not have the correctness of my baseline hinge on
a tie-break I would have to remember correctly, so I hand back the strictly decreasing `−np.arange`,
whose maximum is unambiguously at position `0` with no ties to break. Both encode the same rule; the
decreasing one just cannot be silently broken by a change in tie-break convention, and its `argmax` is
a genuine strict maximum I have checked lands on the earliest position.

Before I hand that fix forward I should check it is not a distinction without a difference. There is a
tempting worry that First-Fit's own dynamics already make the earliest bin the tightest — bins open
left to right, and the oldest bins have had the most chances to receive items, so perhaps "earliest"
and "tightest" always coincide and switching criteria would buy nothing. If that were true the whole
ladder would be pointless, so I want to see it settled on the trace I already have. After First-Fit
finishes `[2, 9, 1, 8]` the remaining capacities are `[7, 1, 2]`: the *oldest* bin, bin 0, is the
*roomiest* of the three, and the tightest surviving bin is bin 1, opened later. So "earliest" and
"tightest" genuinely diverge under First-Fit's own behaviour — the roomy old bin is exactly the one
First-Fit keeps pouring small items into and never closes off. There is real daylight between the two
criteria, which means the next rung's change of rule has something concrete to win rather than a
relabelling of the same choice. Good; I am not about to propose a distinction without a difference.

The cost of running this is trivial and worth noting so I know the baseline scales: per item the
harness computes `get_valid_bin_indices` in one pass over the pre-allocated array, and my `priority`
allocates an `arange` of length equal to the number of valid bins and takes its `argmax`, all linear in
the number of bins touched. Nothing here grows super-linearly per item; the harness's own `get_valid_bin_indices`
scan over the pre-allocated array is the dominant cost and it is linear in the array length, so a
5000-item run is comfortably fast even though the pre-allocation makes the per-item scan cover all
5000 slots. The policy is fully deterministic given the stream, so the five seeds differ only through
their different item streams, not through any randomness in the rule — which is exactly what I want for
a baseline: any spread I see across the five seeds is a property of the *streams*, the sampling
variability of a 5000-item Weibull or uniform draw, and not noise I injected. That is why running five
seeds is worth the cost: one seed could sit at a lucky or unlucky corner of the sampling distribution,
and the mean over five tells me where First-Fit really lives while the per-seed spread tells me how
much the stream draw alone moves the number. When I later compare rungs on the *same* five seeds, the
stream variability cancels and the difference I read is purely the change of rule.

That determinism lets me make a fairly specific prediction to hold the run to. First-Fit does greedy
reuse, which I argued recovers most of the distance from 5000 bins down toward the ~2010 / 2000 floor,
so I expect it to land only a few percent above `LB` on both families — not catastrophic, but a clear,
visible margin of waste, and I would put money on low single digits rather than a fraction of a
percent, because the hole-management sloppiness I just traced is a persistent leak, not a one-off. In
the two metric columns I care about, that means `mean_bins` a little above my `LB` estimates — on the
order of `2010`-ish for Weibull and `2000`-ish for OR-style, plus the few-percent margin — and
`excess_over_LB` in the low single digits on both families. I also expect the five seeds to cluster
tightly: 5000 items is a large sample, the law of large numbers pins the sum (hence `LB`) close to my
estimates, and the deterministic rule adds no extra spread, so the per-seed excess should sit inside a
band well under a point wide. The exact height of that band is the one thing I cannot compute by hand —
the accumulated waste depends on the fine structure of the streams — so the feedback table is what will
tell me whether "a few percent" means two or five.

There is one methodological caution I want to register before I trust any of these readings. My `LB`
estimates and my "few percent" guess are both derived from the distribution parameters and a couple of
hand-traces; the simulator itself I have taken on faith. First-Fit is a textbook policy with
well-known behaviour on standard instances, so the cheapest way to earn confidence in the harness is to
check that it reproduces First-Fit's *known* numbers on a reference instance before I read my own
seeded streams — if the simulator agrees with the established First-Fit result to the digit, then the
numbers it reports on my Weibull and OR-style streams mean what they say, and if it does not, I have a
harness bug to find before anything else. I will want that calibration alongside the first feedback,
not after I have built three more rungs on top of an unverified simulator.

The specific weakness I have named and traced, that First-Fit ignores the slack a placement leaves and
so squanders roomy bins on small items and fails to finish off tight bins, is the exact thing the next
rung must fix: instead of taking the *earliest* valid bin, take the one whose *fit is tightest*, so
that every placement wastes as little capacity as possible and tight bins get closed off. That is the
move from First-Fit to Best-Fit, and it is where I go next.
