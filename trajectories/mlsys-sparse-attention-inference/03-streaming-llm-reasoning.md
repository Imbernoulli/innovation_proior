BigBird landed where the NIAH-vs-QA split predicted, and the numbers tell me which of its three
ingredients earned its budget and which wasted it. NIAH collapsed from the oracle's `1.0` to `0.2` —
chance — the structural cliff I warned about: a content-blind mask covers a mid-haystack needle only if the
needle's block happens to fall in a window, in the two global blocks, or in the fixed random sample, and on
average it does not. The QA tasks held better, as distributed evidence should: Qasper F1 `0.0871` against
`0.1406` is `0.62` of the oracle recovered, MultiFieldQA-EN `0.2298` against `0.3447` about two-thirds —
usable, but *below* what a sensible local-plus-anchor pattern should give, and Qasper in particular came in
soft. The densities are the other tell and they are alarming: `0.2601`, `0.2421`, `0.2693`, two of three
*above* `0.25`, and the MultiFieldQA run cleared the `0.25 + 0.02 = 0.27` abort line by `0.0007` — less
than a thousandth of density from being killed and scoring nothing. So BigBird spent its full budget,
overshot on two of three tasks, survived one by a hair, and *still* cratered NIAH and underperformed on
Qasper.

That reframes the next move: BigBird failed not for lack of budget but because of *where* it spent it. A
meaningful share of its `~26%` went to 23 random expander blocks per query, chosen blind to the question.
Those buy graph connectivity in the abstract, and the mixing-time argument is not wrong, but at inference,
with a fixed sample and no query signal, a random far block is overwhelmingly likely to be irrelevant to
the current question, so that budget mostly routes to noise. Worse, spending it on random blocks *starves*
the parts that carry signal — the local context the model leans on hardest, and the anchor tokens that hold
the distribution together. And the density overshoot is a second liability with direct evidence: the random
sampling plus the global rows pushed the realized density right up against the abort line, so the `0.88`
margin I thought was conservative turned out thin, and on a longer prompt or an unluckier layer I am one
bad draw from the harness killing the run. The diagnosis is therefore not "make the pattern adaptive yet" —
it is spend a static budget on the parts that are reliably useful, cut the random gamble, and size the
window so density lands *at* budget instead of over it. Get the floor right before getting clever.

What is reliably useful in a static pattern is the two things BigBird diluted with random. The first is the
recent local window: the keys immediately preceding the query carry the bulk of the attention mass in
trained models, and next-token prediction leans hardest on local context. The second is the anchor tokens
at the very start, where there is a real mechanism, not just a heuristic. Softmax forces the weights to sum
to one — there is no "attend to nothing" option — so a query whose context holds nothing it strongly needs
must still deposit a full unit of attention mass somewhere, and it learns to dump that surplus on a fixed,
always-reachable set. Under causal masking the only positions visible to *every* later query are the first
few tokens, so those become the universal dumping ground — attention sinks, valuable not for their content
but for the softmax denominator they hold. Drop them and every remaining weight is renormalized into a
shape the model never saw, and quality falls off a cliff. A few sink tokens plus a recent window is the
minimal static pattern that respects how trained attention actually distributes its mass — precisely what
BigBird's random gamble was crowding out.

Fixing the static floor first is the right order because sink+window and an adaptive mask are not
competitors: the adaptive method I will eventually want still needs a guaranteed local keep and still
benefits from anchors, so whatever I learn here about sizing the local budget carries forward. And this
rung is a control. Both static patterns share exactly one property — the mask is fixed before the query is
known — and if a *correctly sized* sink+window, with nothing wasted on random, still leaves NIAH at chance,
then I will have isolated the NIAH failure to staticness per se rather than to BigBird's particular budget
split. That is a cleaner diagnosis than "BigBird was bad," and it is the one that justifies the cost of
going adaptive next. The sinks bound my NIAH expectation directly: they sit at the *start* of the sequence,
so they cover the needle only in the degenerate case where it was planted in the first few tokens — a
chance of `num_sinks / N ≈ 4/8192`, vanishingly small. So the sinks fix *stability*, not *retrieval*: they
keep the model coherent under sparsity but do not route to arbitrary positions.

The budget split between the two ingredients is a real allocation. Sinks buy stability (the softmax
denominator) and window buys coverage (recent context), and their marginal values differ: a handful of
sinks restores the trained distribution and further sink columns just repeat a done job, while each window
token recovers one more token of local context at roughly constant value. So the rational split gives the
sinks the *minimum* that stabilizes and pours everything else into window. Four is that minimum — models
with no single consistent start token spread the sink role across several initial positions, and one or two
do not fully restore the distribution — and with `num_sinks = 4` and `W = 1020` the sinks are `0.4%` of the
row budget and the window the other `99.6%`.

This is *this task's* static mask, not the streaming-cache variant, and the difference is substantial. The
native version is a KV-cache eviction scheme: keep a small rolling cache, evict the middle, and — the
signature move — re-index positions *within the cache* so the rotary positions stay contiguous, which needs
a position-shift adapter that rotates cached keys by cache-position each decode step. None of that exists
here. The harness runs `use_cache=False`; every forward is a full parallel pass over the entire prefix, and
the same module replays at every generation step, so there is no cache to evict and no eviction-time
re-indexing — the positions are the model's own RoPE, applied by the loop before I see `q, k, v`. So the
approach is a static sink+window mask over the full `(N, N)` causal matrix; only the *pattern* transfers,
not the constant-memory story, and I derive the window size from the density contract directly.

BigBird's overshoot teaches me to be exact rather than conservative-by-fudge-factor, because every affordable
window token is local context recovered and I do not want to leave budget on the table the way a hand-tuned
margin would. Once the query index exceeds `num_sinks + W`, each row keeps exactly `num_sinks + W` keys —
the sinks plus the last-`W` window, no overlap — so to first approximation the mask sum is
`N·(num_sinks + W)` and the density over `N(N+1)/2` is `2(num_sinks + W)/(N+1)`; setting that equal to `ρ`
gives `W ≈ round(ρ·(N+1)/2) − num_sinks`. At `N = 8192, ρ = 0.25, num_sinks = 4`:
`round(1024.125) − 4 = 1020`, a saturated row keeping `1024` keys. But the first approximation assumes every
row keeps the full quota, while the early rows (index below `W`) keep only their whole causal prefix, which
is fewer — and BigBird's lesson was that the realized count drifts off the algebra, so I sum the realized
mask. A row `i ≤ 1019` keeps `i + 1`; a row `i ≥ 1023` keeps `1024`. So
`Σ_{i=0}^{1019}(i+1) + 7172·1024 = 520{,}710 + 7{,}344{,}128 ≈ 7{,}864{,}838` over
`8192·8193/2 = 33{,}558{,}528`, a density `≈ 0.234` (the early-row boundary is approximated, so the fourth
digit is not trustworthy). The number I should see reported is concretely around `0.234`, not just "under
`0.25`", and it should be nearly identical on all three tasks even at different context lengths, because `W`
is solved per `N` and the ratio is pinned. That is the correction that matters: the naive sizing that took
`ρ·(N+1)/2` as a row *count* conflated the row-relative window with the column-relative density and
overshot — the same overshoot BigBird showed. As `ρ → 0` the solve gives sinks-only; as `ρ → 1` it gives
`W = 4093` and density `≈ 1.0`, full attention — so it interpolates correctly and `0.25` is a point on that
curve.

That uniformity is worth its own line against BigBird's numbers. BigBird's densities spread `0.027` across
tasks because the random sampling interacts with each length differently, and that spread is what put one
run a thousandth from aborting. My `W`-solve depends on `N` alone, so no task is a bad draw from the ceiling
— there is no draw. And the reclaim is concrete: BigBird's window was 3 blocks (`~192` tokens) with 23 of
its 28 kept blocks on random; pouring that into window turns `~192` local tokens into `~1020`, about 16
contiguous blocks, so this rung reads roughly five times as much contiguous local context, bought entirely
by refusing the random gamble.

I should name what I give up to buy it, because it is exactly the property last rung's graph argument called
essential. Cutting the random edges throws away the long-range reachability: a mid-sequence fact cannot
propagate to the end in few hops. But BigBird's own numbers are the counter-evidence — it *had* that
reachability and its NIAH still sat at chance `0.2` and its Qasper was soft, so the abstract `O(log N)`
reachability the expander bought was not cashing into retrieval on a frozen model, because a random
long-range edge is a graph edge but not a *relevant* edge, and the softmax deposits negligible mass on an
irrelevant far block. Dropping it costs little that BigBird was actually collecting and frees the budget for
local context the model reads heavily. If I am wrong — if the QA numbers fall *below* BigBird's rather than
rise — that would tell me the reachability was doing real work after all. And I should be clear-eyed about
how little of the document this sees: `W = 1020` is `1024/8192 ≈ 12.5%` of the context, covering only the
document tail for the question tokens at the end, everything before invisible except the four sink columns.
So sink+window bets the evidence a query needs is in the tail or reconstructable from the model's parametric
knowledge plus the stabilized local read — a bet that pays partially on QA but also caps the recovery,
because mid-document evidence is simply gone.

The implementation is the masked-softmax form, same as the previous rung: build the `(N, N)` boolean
keep-mask (`0 ≤ i − j < W` for the row-relative last-`W` window, OR `j < num_sinks` for the sink columns,
AND the causal lower triangle), compute `QKᵀ · scale` in float32 for stable masking/softmax, `masked_fill`
the dropped entries with `−∞`, softmax, `nan_to_num` any empty row, multiply by `V`, cast back. I report
`last_density` as the *measured* fraction of the realized mask over `N(N+1)/2`, which should agree with my
hand-summed `≈ 0.234` — itself a check that the mask I built is the mask I intended.

Now the predictions against BigBird's measured numbers. On the QA tasks I expect to reclaim the budget
BigBird wasted on random and turn it into local window, where QA evidence and the language-modeling signal
live, so Qasper F1 should come *up* from `0.0871` toward the oracle — a clean sink+window keeps more
contiguous relevant context than a window diluted with random gambles — though still short of `0.1406`
because the mid-document evidence is gone. MultiFieldQA I am less sure about: its `0.2298` already had a
real local window inside it, so replacing random with more window may only hold it in its band rather than
lift it, since the gain from more window competes against whatever scattered spans the random blocks
happened to catch, and I will not be surprised by a small move in either direction. On NIAH I am honest
that this is still a *static* pattern: the sink columns cover the first few tokens, the window covers the
recent tokens, and a needle in the middle of an 8K haystack is outside both, so I expect NIAH to stay stuck
near BigBird's `0.2`. And all three densities should sit at `≈ 0.234`, comfortably under `0.25` where
BigBird's two-of-three sat over — if the density does not drop as I computed, my mask is not the mask I
derived. If NIAH stays at chance while Qasper rises and the densities drop, that is the clean control result:
two independent static geometries converging on the no-needle floor pins the failure on the one thing they
share, and the *only* remaining gap to the oracle's NIAH `1.0` is what no fixed mask can buy — a selection
that depends on what the query is asking. That gap hands the next rung its job: make the kept blocks
content-adaptive. The distilled module with the density-exact window sizing is in the answer.
