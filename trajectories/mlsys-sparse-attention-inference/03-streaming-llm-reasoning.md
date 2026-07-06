BigBird landed exactly where the NIAH-vs-QA split predicted, and the numbers are worth reading closely
because they tell me which of its three ingredients earned its budget and which wasted it. NIAH collapsed
from the oracle's `1.0` to `0.2` — chance-level retrieval — which is the structural cliff I warned about:
a content-blind mask covers a mid-haystack needle only if the needle's block happens to fall in a window,
in the two global blocks, or in the fixed random sample, and on average it does not. The QA tasks held up
better, as distributed evidence should: Qasper F1 `0.0871` against the oracle's `0.1406`, MultiFieldQA-EN
`0.2298` against `0.3447`. Let me turn those into recovery fractions so I know how much is actually left to
chase. Qasper kept `0.0871 / 0.1406 = 0.62`, so about 62% of the oracle's F1 survived; MultiFieldQA kept
`0.2298 / 0.3447 = 0.67`, about two-thirds. That is a usable share but it is *below* what I would hope a
sensible local-plus-anchor pattern gives, and Qasper in particular came in soft. The density readings are
the other tell, and they are alarming: `0.2601` on NIAH, `0.2421` on Qasper, `0.2693` on MultiFieldQA. Two
of those three are *above* `0.25`. The MultiFieldQA run reported `0.2693` against an abort line of
`0.25 + 0.02 = 0.27` — it cleared the harness by `0.0007`, less than a thousandth of density from being
killed and scoring nothing. So BigBird spent its full budget, overshot it on two of three tasks, survived
one of those by a hair, and *still* cratered NIAH and underperformed on Qasper.

That last fact reframes the next move. BigBird did not fail for lack of budget; it failed because of
*where* it spent it. A meaningful share of its `~26%` went to **random** blocks — 23 long-range expander
edges per query block chosen blind to the question. Those edges buy graph connectivity in the abstract,
and the mixing-time argument that justified them is not wrong, but at inference, with a fixed sample and no
query signal, a random far block is overwhelmingly likely to be irrelevant to the current question, so that
budget is mostly spent routing to noise. Worse, spending budget on random blocks *starves the parts that
actually carry signal* — the local context the model leans on most heavily, and the anchor tokens that hold
the attention distribution together. And the density overshoot itself is a liability I now have direct
evidence for: the random sampling plus the global rows pushed the realized density right up against the
abort line, so the `0.88` margin I thought was conservative turned out to be thin, and on a longer prompt or
an unluckier layer I am one bad draw from the harness killing the run. The diagnosis is therefore not "make
the pattern adaptive yet" — it is "first, spend a static budget on the parts that are reliably useful, cut
the random gamble that bought nothing on these tasks, and size the window so the density lands *at* budget
instead of over it." Get the floor right before getting clever.

So what is reliably useful in a static pattern? Two things, and they are exactly the two BigBird diluted
with random. The first is the recent local window: the keys immediately preceding the query carry the bulk
of the attention mass in trained models, and for language modeling the next-token prediction leans hardest
on local context. The second is the anchor tokens at the very start. There is a real mechanism here, not
just a heuristic. Softmax forces the attention weights to sum to one — there is no "attend to nothing"
option — so on a query whose context holds nothing it strongly needs, the model must still deposit a full
unit of attention mass somewhere. It learns to dump that surplus on a fixed, always-reachable set of
positions. Under causal masking, the only positions visible to *every* later query are the first few
tokens, so those become the universal dumping ground — attention sinks. They are valuable not for their
content but for the softmax denominator they hold: drop them and every remaining weight is renormalized
into a shape the model never saw, and quality falls off a cliff. So a few sink tokens plus a recent window
is the minimal static pattern that respects how trained attention actually distributes its mass — and it is
precisely what BigBird's random gamble was crowding out.

I want to be precise about why this is the right *order* of operations, because it is tempting to jump
straight to adaptivity now that the static rungs are visibly failing NIAH. The reason to fix the static
floor first is that the sink+window mask and an adaptive mask are not competitors — the adaptive method I
will eventually want still needs a guaranteed local keep and still benefits from anchors, so whatever I
learn here about sizing the local budget carries forward. And there is a falsification value in running
this rung that I do not get by skipping it: if a *correctly sized* sink+window — one that spends its whole
budget on local context and anchors, with nothing wasted on random — still leaves NIAH at chance, then I
will have isolated the NIAH failure to staticness per se rather than to BigBird's particular budget split.
That is a cleaner diagnosis than "BigBird was bad," and it is the diagnosis that justifies the cost and
complexity of going adaptive in the next rung. So this rung is partly a control: it removes the
random-block confound and asks whether the best possible static pattern moves NIAH at all.

Let me also be explicit about what the sink columns do *not* buy, because it bounds my NIAH expectation.
The sinks anchor the softmax denominator and recover the distribution's trained shape, which is why
window-only attention collapses and sink+window does not — but the sink tokens are at the *start* of the
sequence, so they cover the needle only in the degenerate case where the needle was planted in the first
few tokens. For a needle placed uniformly in an 8K haystack that is a chance of about `num_sinks / N ≈
4 / 8192`, vanishingly small. So the sinks fix *stability*, not *retrieval*; they keep the model coherent
under sparsity but they do not route to arbitrary positions. That is the precise sense in which a static
pattern, however well sized, is structurally blind to the needle — and it is why I expect this rung to make
the model behave sensibly (QA recovers) while NIAH stays stuck.

There is a budget split to decide between the two ingredients, and it is a real allocation, not a default.
The sinks and the window both eat into the same `~0.25` per-row budget, and they do different jobs: sinks
buy *stability* (the softmax denominator) and window buys *coverage* (recent context). The marginal value
of a sink falls off fast — a handful restores the trained distribution and further sink columns just repeat
a job already done — while the marginal value of a window token is roughly constant, each one recovering
one more token of the local context the model actually reads. So the rational split is to give the sinks
the *minimum* that stabilizes the distribution and pour everything else into window. Four sinks is that
minimum: with `num_sinks = 4` and `W = 1020`, the sinks are `0.4%` of the row budget and the window is the
other `99.6%`. I could imagine being wrong at the extremes — zero sinks collapses the model (the known
window-only failure), and an absurdly large sink count would waste coverage on four repeated anchors — but
between those the allocation is not delicate, and the derivative argument says put the budget where each
unit still pays, which is the window.

Now I have to be careful that this rung is *this task's* method and not the streaming-cache variant of it,
because the difference is substantial. The streaming-cache version of sink+window is a **KV-cache
eviction** scheme: it keeps a small rolling cache, evicts the middle, and — the signature move — re-indexes
positions *within the cache* so the rotary positions stay contiguous and in-distribution, which requires a
position-shift attention adapter that rotates cached keys by cache-position at each decode step. None of
that exists here. The harness runs `use_cache=False`; every forward is a full parallel pass over the entire
prefix, and the same module replays at every generation step. There is no cache to evict, no eviction-time
position re-indexing — the positions are the model's own RoPE positions, applied by the loop before I see
`q, k, v`. So in this setting the streaming approach is not a cache policy at all; it is a **static
sink+window mask** over the full `(N, N)` causal matrix. The "constant memory, constant latency" story the
cache version sells is irrelevant here — what transfers is only the *attention pattern*: keep the first
`num_sinks` columns and a recent window per query, mask everything else. I have to derive the window size
from the density contract directly, in this static-mask setting, not from a cache budget.

That sizing is where BigBird's overshoot teaches me to be exact rather than conservative-by-fudge-factor.
I want the measured density to land *at* the budget, because every token of window I can afford is local
context recovered, and I do not want to leave budget on the table the way a hand-tuned margin would — but I
also cannot overshoot, or the harness aborts. So I derive the window from the mask-sum over the harness's
exact causal denominator. Take the causal case. Once the query index exceeds `num_sinks + W`, each row
keeps exactly `num_sinks + W` keys — the sinks plus the last-`W` window, with no overlap — so to a first
approximation the total mask sum is `N · (num_sinks + W)` and the density over `N(N+1)/2` is
`2(num_sinks + W)/(N+1)`. Set that equal to the budget `ρ` and solve: `W ≈ round(ρ · (N+1)/2) − num_sinks`,
clamped to at least 1. Plug in `N = 8192`, `ρ = 0.25`, `num_sinks = 4`:
`round(0.25 · 8193 / 2) − 4 = round(1024.125) − 4 = 1024 − 4 = 1020`. So `W = 1020`, and a saturated row
keeps `1024` keys.

Now I do not want to trust that first approximation blindly, because it assumes *every* row keeps the full
`num_sinks + W`, and the early rows — those with index below `W` — keep only their whole causal prefix,
which is fewer. So let me actually sum the realized mask rather than use the flat estimate, since the whole
lesson from BigBird was that the realized count drifts off the algebra. For a row `i ≤ W − 1 = 1019`, the
window `[i−W+1, i]` clips to `[0, i]`, which already contains the sink columns, so the row keeps `i + 1`.
For a row `i ≥ 1023` the window `[i−1019, i]` starts past the sinks, so the row keeps `1020 + 4 = 1024`.
Summing, `Σ_{i=0}^{1019}(i+1) + Σ_{i=1020}^{8191}(1024) = 1020·1021/2 + 7172·1024 = 520{,}710 +
7{,}344{,}128 ≈ 7{,}864{,}838`, over the denominator `N(N+1)/2 = 8192·8193/2 = 33{,}558{,}528`. That is a
density of `≈ 0.234` (the fourth digit is not trustworthy — the early-row boundary is approximated). So the number I should actually see reported is not "just under `0.25`" but
concretely around `0.234`, and it should be nearly the same on all three tasks even though the QA contexts
have different lengths, because `W` is solved per `N` and scales with it, so the density ratio is pinned to
roughly the same value regardless of the exact context length. This is the correction that matters: the
naive sizing that took `avg_row = ρ · (N+1)/2` as a *row count* conflated the row-relative window with the
column-relative density and over-shot — the same overshoot BigBird showed — while deriving `W` from
`mask_sum / denom` and then *checking* the sum lands it comfortably under budget with the early-row deficit
built in. The non-causal branch would be the symmetric `|i−j| ≤ W` with `W ≈ (round(ρ·N) − num_sinks − 1)/2`,
but `is_causal = True` always here, so the causal formula is the live one. I keep `num_sinks = 4` because
trained models with no single consistent start token spread the sink role across several initial positions,
and one or two sinks do not fully restore the distribution.

Let me sanity-check the formula at its extremes before I trust it, because a sizing rule that breaks at the
boundaries is a sizing rule I do not understand. As `ρ → 0` the solve gives `W → 0` and the mask degenerates
to sinks-only, which is the minimal stable pattern — sensible. As `ρ → 1` it gives `W = round(1·(N+1)/2) −
4 = 4093`, a saturated row keeps `4097` keys, and the density `2·4097/8193 ≈ 1.0` — full attention. So the
formula interpolates correctly between the pure-sink floor and dense, which is exactly what a budget-to-
window map should do, and it tells me the intermediate value at `ρ = 0.25` is not a coincidence but a point
on a curve I can read at any budget.

The other thing the derivation buys, beyond landing under budget, is *uniformity*, and I can see its value
directly in BigBird's numbers. BigBird's three densities were `0.2601`, `0.2421`, `0.2693` — a spread of
`0.027` across tasks, because the random sampling interacts with each task's length differently, and that
spread is what put one run a thousandth from aborting. My `W`-solve is derived from `N` alone, so it should
report essentially the same `≈ 0.234` on all three tasks regardless of their context lengths — no task is a
bad draw from the ceiling, because there is no draw. And the reclaim is concrete when I count it in blocks:
BigBird's window was 3 blocks, about 192 tokens, with the bulk of its off-local budget — 23 of its 28 kept
blocks — spent on random. Cutting the random and pouring it into window turns those `~192` local tokens into
`~1020`, about 16 contiguous blocks, so this rung reads roughly five times as much contiguous local context
as BigBird did, bought entirely by refusing the random gamble. That five-fold wider local read is the
mechanism I expect to show up as the QA recovery.

I should name what I am giving up to buy it, because it is exactly the property last rung's graph argument
said was essential. Cutting the random edges throws away the long-range reachability — sink+window is a ring
with a start-anchor, its multi-hop diameter is terrible, a mid-sequence fact cannot propagate to the end in
few hops. The mixing-time reasoning that motivated BigBird's random blocks says that should be costly. But
BigBird's own numbers are the counter-evidence: it *had* that reachability and its NIAH still sat at chance
`0.2` and its Qasper was soft. So the abstract `O(log N)` reachability the expander bought was not cashing
into retrieval on a frozen model at inference — a random long-range edge is a graph edge, but it is not a
*relevant* edge, and the softmax over an irrelevant far block deposits negligible mass there. Dropping
reachability therefore costs me little that BigBird was actually collecting, and it frees the budget for
local context that the model reads heavily. That is the trade this rung makes with its eyes open: sacrifice
a theoretical property that did not pay, to strengthen a concrete one that does. If I am wrong — if the QA
numbers fall because the lost long-range edges were quietly carrying distributed evidence — the F1 will drop
below BigBird's rather than rise, and that would tell me the reachability was doing real work after all.

It is worth being clear-eyed about how little of the document this actually sees, because it bounds the QA
prediction. A window of `W = 1020` is `1024 / 8192 ≈ 12.5%` of the context: for the question tokens at the
very end, the window covers only the last `~1020` tokens of the document — its tail — and everything before
that is invisible except for the four sink columns at the very start. So sink+window is betting that the
evidence a query needs is either in the document tail or reconstructable from the model's parametric
knowledge plus the stabilized local read. That bet pays partially: many LongBench questions are answerable
from the tail or from local context, and the sink-restored distribution keeps the model coherent, which is
why QA recovers a usable share rather than collapsing. But it also caps the recovery — mid-document evidence
is simply gone — so I should not expect sink+window to *reach* the oracle's QA F1, only to claw back what
BigBird wasted on random and turn it into denser local coverage. That is the honest ceiling for this rung on
QA, and it is a different ceiling from NIAH, where the needle is by construction not in the tail.

The implementation is the masked-softmax form, same as the previous rung: build the `(N, N)` boolean
keep-mask — `(i − j) ≥ 0 & (i − j) < W` for the row-relative last-`W` window, OR `j < num_sinks` for the
sink columns, AND the causal lower triangle — then compute `QKᵀ · scale` in float32 for stable
masking/softmax, `masked_fill` the dropped entries with `−∞`, softmax, `nan_to_num` any empty row, multiply
by `V`, cast back. I report `last_density` as the *measured* fraction of the realized mask over `N(N+1)/2`,
not the formula, so the contract gets the true kept fraction — and since I hand-summed that realized count
to `≈ 0.234`, the measured report and my derivation should agree, which is itself a check that the mask I
built is the mask I intended.

Before the predictions, let me be explicit about what the NIAH number will mean either way, because this
rung is designed as a control and I want to know in advance how to read it. BigBird's NIAH sat at `0.2`,
which is not zero — it is the floor a model hits when it answers the retrieval question *without* having
seen the needle, guessing from format and priors, so `0.2` is the "no-needle" baseline for this task, not a
signal that anything was retrieved. Sink+window is a geometrically completely different static mask from
global+window+random: no random edges, a much wider contiguous window, the same start anchors. If, despite
that different geometry, its NIAH lands at the same `0.2`, then two independent static patterns have
converged on the no-needle floor, and the only thing they share is *staticness* — the mask is fixed before
the query is known. That convergence would be the decisive control result: it would rule out "BigBird just
spent its budget badly" and pin the NIAH failure on the property both patterns share, which is exactly the
property the next rung has to break. If instead sink+window's NIAH climbs meaningfully above `0.2`, then
the static-versus-adaptive story is more complicated than I think and I would have to reconsider whether a
cleverer *static* mask could still route. I expect the former, and I am setting up the measurement so that
the result is diagnostic rather than merely a score.

Now the predictions against BigBird's measured numbers, which is the bar this rung must clear. On the QA
tasks I expect to *recover* the budget BigBird wasted on random blocks and turn it into local window, which
is where QA evidence and the language-modeling signal actually live. So I expect Qasper F1 to come *up*
from `0.0871`, toward or past the `0.10` mark — if the five-fold wider local read translates into a
proportional gain in covered evidence I would put it somewhere in the `0.10–0.12` band, still short of the
oracle's `0.1406` because the mid-document evidence is gone, but clearly above BigBird's soft `0.0871`
because a clean sink+window keeps more of the contiguous relevant context than a window diluted with random
gambles. MultiFieldQA I am less sure about: BigBird's
`0.2298` already had a real local window inside it, so replacing random with more window may only hold it in
its band rather than lift it — I expect it to stay near `0.2298`, and I will not be surprised either way by
a small move, because the gain from more window competes against whatever scattered spans the random blocks
happened to catch. On NIAH I am honest that this is still a *static* pattern: the sink columns only cover
the first few tokens and the window only covers the recent tokens, so a needle planted in the middle of an
8K haystack is still outside the mask, and I expect NIAH to stay stuck around BigBird's `0.2` — chance.
Static sink+window does not solve the routing problem; it only spends the static budget more honestly. And
all three densities should sit at `≈ 0.234`, comfortably under `0.25`, where BigBird's two-of-three sat
over — that itself is a falsifiable improvement, and if the density does not drop as I computed then my
mask is not the mask I derived. If NIAH stays at chance while Qasper rises and the densities drop to `0.234`,
that is the clean diagnosis I want: the static floor is now correct, the wasted-random budget is reclaimed,
and the *only* remaining gap to the oracle's NIAH `1.0` is the one thing no fixed mask can buy — a selection
that depends on what the query is asking. That gap is what hands the next rung its job: make the kept blocks
*content-adaptive*. The distilled module — the literal scaffold fill with the density-exact window sizing —
is in the answer.
