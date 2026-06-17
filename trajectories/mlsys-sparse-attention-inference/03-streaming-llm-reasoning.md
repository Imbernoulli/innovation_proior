BigBird landed exactly where the NIAH-vs-QA split predicted, and the numbers are worth reading closely
because they tell me which of its three ingredients earned its budget and which wasted it. NIAH
collapsed from the oracle's `1.0` to `0.2` — chance-level retrieval — which is the structural cliff I
warned about: a content-blind mask covers a mid-haystack needle only if the needle's block happens to
fall in a window, in the two global blocks, or in the fixed random sample, and on average it does not.
The QA tasks held up better, as distributed evidence should: Qasper F1 `0.0871` against the oracle's
`0.1406`, MultiFieldQA-EN `0.2298` against `0.3447`. So roughly two-thirds of MultiFieldQA's F1
survived, but Qasper kept only about 60% — and Qasper actually came in *below* what I would hope a
sensible local+anchor pattern gives. The density readings are the other tell: `0.2601` on NIAH, `0.2421`
on Qasper, `0.2693` on MultiFieldQA. Two of those three are *above* `0.25` and one is right at the
`0.25 + 0.02` ceiling. So BigBird spent its full budget — and then some — and still cratered NIAH and
underperformed on Qasper.

That last fact reframes the next move. BigBird did not fail for lack of budget; it failed because of
*where* it spent it. A meaningful share of its 26% went to **random** blocks — long-range expander edges
chosen blind to the question. Those edges buy graph connectivity in the abstract, but at inference, with
a fixed sample and no query signal, a random far block is overwhelmingly likely to be irrelevant to the
current question, so that budget is mostly wasted on noise. Worse, spending budget on random blocks
*starves the parts that actually carry signal* — the local context the model leans on most heavily, and
the anchor tokens that hold the attention distribution together. And the density overshoot itself is a
liability: it means the random sampling plus the global rows pushed me right up against the abort line,
so on a longer prompt or a different layer I am one bad draw away from the harness killing the run for
nothing. The diagnosis is therefore not "make the pattern adaptive yet" — it is "first, spend a static
budget on the parts that are reliably useful, and cut the random gamble that bought nothing on these
tasks." Get the floor right before getting clever.

So what is reliably useful in a static pattern? Two things, and they are exactly the two BigBird diluted
with random. The first is the recent local window: the keys immediately preceding the query carry the
bulk of the attention mass in trained models, and for language modeling the next-token prediction leans
hardest on local context. The second is the anchor tokens at the very start. There is a real mechanism
here, not just a heuristic. Softmax forces the attention weights to sum to one — there is no "attend to
nothing" option — so on a query whose context holds nothing it strongly needs, the model must still
deposit a full unit of attention mass somewhere. It learns to dump that surplus on a fixed, always-
reachable set of positions. Under causal masking, the only positions visible to *every* later query are
the first few tokens, so those become the universal dumping ground — attention sinks. They are valuable
not for their content but for the softmax denominator they hold: drop them and every remaining weight is
renormalized into a shape the model never saw, and quality falls off a cliff. So a few sink tokens plus
a recent window is the minimal static pattern that respects how trained attention actually distributes
its mass — and it is precisely what BigBird's random gamble was crowding out.

I want to be precise about why this is the right *order* of operations, because it is tempting to jump
straight to adaptivity now that the static rungs are visibly failing NIAH. The reason to fix the static
floor first is that the sink+window mask and an adaptive mask are not competitors — the adaptive method I
will eventually want still needs a guaranteed local keep and still benefits from anchors, so whatever I
learn here about sizing the local budget carries forward. And there is a falsification value in running
this rung that I do not get by skipping it: if a *correctly sized* sink+window — one that spends its
whole budget on local context and anchors, with nothing wasted on random — still leaves NIAH at chance,
then I will have isolated the NIAH failure to staticness per se rather than to BigBird's particular
budget split. That is a cleaner diagnosis than "BigBird was bad," and it is the diagnosis that justifies
the cost and complexity of going adaptive in the next rung. So this rung is partly a control: it removes
the random-block confound and asks whether the best possible static pattern moves NIAH at all.

Let me also be explicit about what the sink columns do *not* buy, because it bounds my NIAH expectation.
The sinks anchor the softmax denominator and recover the distribution's trained shape, which is why
window-only attention collapses and sink+window does not — but the sink tokens are at the *start* of the
sequence, so they cover the needle only in the degenerate case where the needle was planted in the first
few tokens. For a needle placed uniformly in an 8K haystack that is a vanishingly small chance. So the
sinks fix *stability*, not *retrieval*; they keep the model coherent under sparsity but they do not route
to arbitrary positions. That is the precise sense in which a static pattern, however well sized, is
structurally blind to the needle — and it is why I expect this rung to make the model behave sensibly
(QA recovers) while NIAH stays stuck.

Now I have to be careful that this rung is *this task's* method and not the paper's, because the
difference is substantial. The paper's version of sink+window is a **KV-cache eviction** scheme: it
keeps a small rolling cache, evicts the middle, and — the signature move — re-indexes positions *within
the cache* so the rotary positions stay contiguous and in-distribution, which requires a position-shift
attention adapter that rotates cached keys by cache-position at each decode step. None of that exists
here. The harness runs `use_cache=False`; every forward is a full parallel pass over the entire prefix,
and the same module replays at every generation step. There is no cache to evict, no eviction-time
position re-indexing — the positions are the model's own RoPE positions, applied by the loop before I see
`q, k, v`. So in this setting StreamingLLM is not a cache policy at all; it is a **static sink+window
mask** over the full `(N, N)` causal matrix. The "constant memory, constant latency" story the paper
sells is irrelevant here — what transfers is only the *attention pattern*: keep the first `num_sinks`
columns and a recent window per query, mask everything else. I have to derive the window size from the
density contract directly, in this static-mask setting, not from a cache budget.

That sizing is where BigBird's overshoot teaches me to be exact rather than conservative-by-fudge-factor.
I want the measured density to land *at* the budget, because every token of window I can afford is local
context recovered, and I do not want to leave budget on the table the way a hand-tuned margin would — but
I also cannot overshoot, or the harness aborts. So I derive the window from the mask-sum over the
harness's exact causal denominator. Causal case: once the query index exceeds `num_sinks + W`, each row
keeps exactly `num_sinks + W` keys (sinks plus the last-`W` window, no overlap), so the total mask sum is
about `N · (num_sinks + W)`, and the density over `N(N+1)/2` is `2(num_sinks + W)/(N+1)`. Set that equal
to the budget and solve: `W ≈ round(budget · (N+1)/2) − num_sinks`, clamped to at least 1. This is the
correction that matters — the naive sizing that took `avg_row = budget · (N+1)/2` as a row-count
conflated the row-relative window with the column-relative density and over-shot, which is the same kind
of overshoot BigBird showed; deriving `W` straight from `mask_sum / denom` lands it on budget. The
non-causal branch (symmetric `|i−j| ≤ W`) is `W ≈ (round(budget·N) − num_sinks − 1)/2`, but
`is_causal=True` always here, so the causal formula is the live one. `num_sinks = 4`, because trained
models with no single consistent start token spread the sink role across several initial positions and
one or two sinks do not fully restore the distribution.

The implementation is the masked-softmax form, same as the previous rung: build the `(N, N)` boolean
keep-mask — `(i − j) ≥ 0 & (i − j) < W` for the row-relative last-`W` window, OR `j < num_sinks` for the
sink columns, AND the causal lower triangle — then compute `QKᵀ · scale` in float32 for stable
masking/softmax, `masked_fill` the dropped entries with `−∞`, softmax, `nan_to_num` any empty row,
multiply by `V`, cast back. Report `last_density` as the *measured* fraction of the realized mask over
`N(N+1)/2`, not the formula, so the contract gets the true kept fraction. Because the window size is
solved to hit budget exactly, I expect all three density readings to sit just *under* `0.25` — and that
is itself a falsifiable improvement over BigBird, whose two-of-three densities sat above it.

Now the predictions against BigBird's measured numbers, which is the bar this rung must clear. On the QA
tasks I expect to *recover* the budget BigBird wasted on random blocks and turn it into local window,
which is where QA evidence and the language-modeling signal actually live. So I expect Qasper F1 to come
*up* from `0.0871` — closer to or past the `0.10` mark — and MultiFieldQA F1 to come up from `0.2298`,
because a clean sink+window keeps more of the contiguous relevant context than a window diluted with
random gambles. On NIAH I am honest that this is still a *static* pattern: the sink columns only cover
the first few tokens and the window only covers the recent tokens, so a needle planted in the middle of
an 8K haystack is still outside the mask, and I expect NIAH to stay stuck around BigBird's `0.2` — chance.
Static sink+window does not solve the routing problem; it only spends the static budget more honestly.
If NIAH stays at chance while Qasper and MultiFieldQA rise and all three densities drop to just under
`0.25`, that is the clean diagnosis I want: the static floor is now correct, the wasted-random budget is
reclaimed, and the *only* remaining gap to the oracle's NIAH `1.0` is the one thing no fixed mask can buy
— a selection that depends on what the query is asking. That gap is what hands the next rung its job:
make the kept blocks *content-adaptive*. The distilled module — the literal scaffold fill with the
density-exact window sizing — is in the answer.
