My hand-built relational rule got most of the way on the Weibull streams and only partway on the
OR-style ones, and I am fairly sure I know why: I *guessed* the functional form. I picked the squared
distance below the emptiest bin, divided by the item once, and a single plain neighbour difference,
all from intuition. Each of those choices — the power on the distance, the power on the item, the
number and shape of the extra terms — is a free parameter I set by hand, and there is no reason my
guess sits at the optimum. The honest move now is to stop guessing the form and let a *search* over
the space of priority functions find it. The fitness signal is already in my hands and it is cheap:
run the online simulator on training streams, count bins, lower is better. That is exactly the kind of
objective a program-search loop can climb — an LLM proposes candidate `priority(item, bins)` bodies,
each is scored by the simulator, the best survivors seed the next round of proposals, and the loop
evolves the code. Whatever it converges to is a priority function selected purely for packing tightly,
with no human prior on its shape. So the final rung is not another form I invent; it is the body that
search settles on, and the work for me is to read that body and check that it actually does what the
fitness number says.

The converged body is short, so let me write it out and then *run* it rather than admire it. It scores
each valid bin as a sum of three terms — `(bins − max_bin_cap)² / item`, plus `bins² / item²`, plus
`bins² / item³` — then sets `score[bins > item] = -score[bins > item]`, then does `score[1:] -=
score[:-1]`. The first term is exactly my rung-3 core: squared distance below the emptiest bin, weighted
by the item. The sign flip and the neighbour differencing are the two structural moves I had also
reached for. The two extra terms, `bins² / item²` and `bins² / item³`, are new, and I would not have
guessed them. Before I trust the story I'd tell about them, I want to know mechanically what this code
does to a placement decision, because two of its lines are easy to misread.

Start with the sign flip, since `score[bins > item] = -score[bins > item]` is the line I least trust by
eye. The simulator only calls `priority` on bins where `bins − item ≥ 0`, i.e. `bins ≥ item`. The flip
fires on `bins > item` — strictly greater — so the one case it does *not* flip is `bins == item`, a bin
whose remaining capacity equals the item exactly, a perfect fit with zero leftover. Take `item = 30`
and three valid bins with remaining `[30, 55, 80]`. The raw three-term scores come out to about
`[84.37, 24.31, 7.35]`. The flip leaves the first (it equals the item) and negates the other two:
`[84.37, −24.31, −7.35]`. So a perfect-fit bin keeps a large positive score while every slack bin is
pushed negative. That is a hard preference for closing a bin exactly, and it falls straight out of the
strict inequality — a detail I would have gotten wrong if I had just assumed `≥`.

Now the differencing, `score[1:] -= score[:-1]`, which I had been thinking of loosely as "compare each
bin to its neighbour." Continuing the example, the post-flip vector `[84.37, −24.31, −7.35]` becomes
`[84.37, −108.67, 16.96]` after each entry subtracts the one before it. The argmax is index 0 — the
exact-fit bin — which matches the preference I just described. Good. But this line has a consequence I
had not taken seriously: because each score now depends on the bin *before* it in the array, the result
depends on the **order** the bins are listed in, not just on their multiset of capacities. Let me check
that, because if it is true it changes how I read the whole function. With `item = 30` and four
slack-only bins `[45, 55, 80, 100]` (no exact fit), the argmax picks the bin with remaining `80`. Now I
permute the *same four numbers*: order `[100, 80, 55, 45]` still picks `80`, but order `[80, 45, 100,
55]` picks `100`. Same capacities, different order, different bin. So the differencing is genuinely
order-sensitive — it is not a clean "nearest-neighbour comparison," it is a position-dependent transform
of the score array. That is a real property of the converged code, and I would have mis-described it.
The simulator feeds bins in a fixed array order, so this is well-defined within a run; I just can't tell
the story that the function ranks bins by an order-free criterion, because it doesn't.

That same four-slack-bin example also kills the lazy reading that this is "just Best-Fit with extra
terms." Best-Fit would take the *tightest* slack bin, remaining `45`. This function took `80`. So among
bins that all leave slack, it is not minimising leftover at all; the placement among slack bins is
governed by the differenced three-term blend, and only the exact-fit case behaves like an obvious
greedy rule. Whatever edge it has over Best-Fit is not a tweak of Best-Fit's objective.

Now the two extra terms. Both scale with `bins²`, but `1/item²` and `1/item³` blow up as the item
shrinks and fade as it grows, so for small incoming items these terms dominate and reshape the ranking,
while for large items the distance-below-emptiest term carries the decision. The plausible reading is
an item-size-dependent *blend*: which relational feature governs placement shifts smoothly with how big
the arriving item is — something my single `(...)² / s` term, with one fixed item weighting, could not
express. That would explain why a hand-tuned rule sits well in one regime and slips in another. But
"plausible reading" is not evidence the terms help, so I should measure it rather than assert it. I run
the converged body and the rung-3 core alone on five seeded Weibull(45,3) streams at capacity 100,
5000 items each, and compare mean excess over the L1 lower bound. The core alone gives `0.718%`; the
full three-term body gives `0.668%`. The extra terms do help, by a small but real margin, and on its
home distribution the full function lands within seven tenths of a percent of a bound no online policy
can beat. The improvement is modest because the core already does most of the work — the search did not
find a smarter *idea*, it found a richer parameterisation of the same idea, fitted by selection.

The last thing to settle is what reproducing this shows on *my* seeded streams versus the streams it was
searched on. I run all three policies through the frozen simulator. On Weibull(45,3), `C = 100`:
First-Fit `4.54%`, Best-Fit `4.15%`, the converged function `0.67%`. So it beats Best-Fit by roughly a
factor of six on its home distribution and effectively closes the gap to the lower bound — and `0.67%`
sits right on the `~0.68%` the published Weibull run reports, which is the calibration I wanted. On the
OR-style uniform `[20, 100]` family at `C = 150`: First-Fit `4.65%`, Best-Fit `4.41%`, the converged
function `3.43%`. It still beats Best-Fit there, but it is nowhere near the lower bound the way it is on
Weibull. That is the expected shape of the result, not a failure of it: this is the *Weibull*-searched
body, and the OR-Library family has its own separately-searched heuristic; a function fitted by selection
to one distribution is sharpest exactly on that distribution. I report the OR number honestly at `3.43%`
rather than quote the `3.11%` that belongs to the OR-discovered function, because mixing them would
misstate what I actually ran.

So the final rung is the body program search settles on, transcribed exactly and read against the same
seeded simulator that scored every rung before it. What I bring to it is the mechanical check of what its
two suspect lines really do — the strict-inequality exact-fit preference and the order-dependent
differencing, both of which I had been describing too cleanly — the measured confirmation that its extra
item-power terms earn their place on Weibull, and the honesty to report it on both its home distribution
and an off-distribution one, so each number means what it says.
