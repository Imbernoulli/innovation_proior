196.80 tokens/s at 862 GB/s. The quantization lever has been pulled twice — bf16 → int8 → int4 — and each
pull bought less: `155.58/104.9 ≈ 1.48×`, then `196.80/155.58 ≈ 1.27×`. Reading 862 back, `862 / 196.80 ≈
4.38 GB` per token, matches the ~4.5 GB int4 model. The diminishing return has a clear cause: as I shrink the
quantized bulk, the un-quantized bf16 tail (~0.5 GB) becomes a larger *fraction* of the token and finer-group
metadata eats the savings. Below 4 bits costs real accuracy, so the representation lever is nearly spent.
More importantly, every rung so far has accepted one assumption without questioning it: **one full forward
pass per committed token**. That's the deepest inefficiency I haven't touched — each step reads ~4.4 GB of
int4 weights to produce *one* token, doing a memory-bandwidth-sized amount of work for a single token.

Here's the asymmetry to exploit, and I'll quantify it because it's the whole basis of the rung. A forward
pass on *one* token and a pass on *several* at once cost almost the same wall-clock, because both are
dominated by reading the weights once. A pass over `k+1` positions reads the same ~4.4 GB as a pass over 1
and does `(k+1)×` the arithmetic — but arithmetic was ~45 µs against a ~6.75 ms weight read, so even at `k=5`
it climbs to ~270 µs, under 5% of the read. A pass over 6 positions costs ~`7.0 ms` instead of ~6.75 — barely
more — yet evaluates the model at *six* positions. The bottleneck is bytes-per-*pass*, not bytes-per-token,
so one pass can verify many positions for the price of one weight read. Autoregressive decoding seems to
forbid this: I can't process token `t+5` until I know `t+1`, which depends on `t`. That serial dependency is
exactly what forces one weight read per token — unless I can *guess* the next several tokens cheaply and check
all the guesses in a single pass.

The quality of the guess determines the payoff. A cheap statistical guesser (n-gram, prompt retrieval) is
cheap but agrees with the big model only on boilerplate, so guesses get rejected and amortization is small.
Extra prediction heads on the model are powerful but change the model and need training, closed to me by the
frozen weights. The option that fits the constraints with the best guess-quality-per-cost is a small, fast
**draft** model proposing the next `k` tokens one at a time — cheap because it's small, and high-agreement
because it's a smaller version of the same kind of model on the same data. I take its `k` guesses, append them
to the current token, and run the big **verifier** *once* over all `k+1` positions in parallel — one weight
read giving the verifier's true next-token distribution at *every* position simultaneously. If the draft
guessed well I commit several tokens for one verifier pass; if badly, I fall back. Either way the expensive
resource — the verifier's weight read — is amortized over multiple tokens.

The thing I absolutely cannot give up is the output distribution. This whole ladder has been "faster, not
dumber," and the quantization rungs paid for bytes with a bounded, validated quality cost; this rung must pay
*nothing*, because I can make it exact. So the acceptance rule has to be exactly distribution-preserving: the
committed stream must be distributed identically to what the verifier alone would produce. That kills the
naive schemes — "accept if the draft's top token matches the verifier's top token" biases the output toward
both modes and changes the effective temperature. I need a rule that provably reproduces `q`, the verifier's
distribution, at every committed position.

The correct rule is rejection sampling against the two distributions. For each drafted position let `p` be
the draft's probability of the token it sampled and `q` the verifier's probability of that same token. Accept
with probability `min(1, q/p)`. If `q ≥ p`, always accept; if `q < p`, accept with probability `q/p`, and on
rejection resample from the residual `normalize(max(q − p, 0))` — exactly the mass the verifier wanted that
the draft under-sampled. I won't take the exactness on faith, so I prove it preserves `q`. Fix a position and
token `x`. The committed token equals `x` two mutually exclusive ways. The accept path contributes
`p(x)·min(1, q(x)/p(x)) = min(p(x), q(x))`. For reject-then-resample: the total rejection probability is `Σ_y
p(y)·(1 − min(1, q(y)/p(y))) = Σ_y max(p(y) − q(y), 0)`, and the residual draws `x` with `max(q(x) − p(x), 0)
/ Σ_z max(q(z) − p(z), 0)`. Since both `p` and `q` sum to 1, `Σ_y max(p(y) − q(y), 0) = Σ_z max(q(z) − p(z),
0)` — the rejection total and the residual normalizer cancel — so the reject-resample contribution is just
`max(q(x) − p(x), 0)`. Add the paths: `min(p(x), q(x)) + max(q(x) − p(x), 0)`, which is `q(x)` whether `q(x)
≥ p(x)` or `q(x) < p(x)`. The committed token is drawn from the verifier's distribution exactly, for any
draft `p`, even a bad one — so the draft's quality affects only *speed* (the accept rate), never
correctness. That's why this rung carries no quality caveat: it needs no accuracy check, exact by
construction.

The mechanics: draft `k` tokens sequentially with the small model, run the verifier once over the `k+1`
positions, sweep the acceptance test position by position, stop at the first rejection; full code in the
answer. Two details earn their place. When *all* `k` drafts are accepted I get a free *bonus* token — the
verifier's pass already produced a distribution at position `k+1` (its prediction after `draft_k`), so I
sample it directly from `target_probs[-1]` and run one draft step to keep the draft's KV-cache in sync. That
gives `k+1` committed tokens from one verifier pass in the best case, one more than I drafted. And I stop at
the *first* rejection rather than salvaging later accepted tokens because once a position is resampled from
the residual, the tokens after it were drafted conditioned on a token that changed, so they're no longer
valid continuations — sweeping to the first rejection and truncating is what keeps the distribution exact.

`speculate_k` trades the gamble, with a clean structure. If the draft agrees with the verifier on any token
with probability `α`, the probability of accepting a *run* of `j` is roughly `α^j` — agreement compounds
multiplicatively — so accepted runs get exponentially rarer as `k` grows. The expected tokens committed per
verifier pass is a truncated geometric, `(1 − α^{k+1})/(1 − α)`, saturating at `1/(1−α)` as `k → ∞` (for
α=0.8, ~5 tokens), so beyond a point I'm drafting tokens that will almost surely be rejected. Put a number on
the amortization so "worth it when the verifier is expensive" isn't hand-waving: at `α = 0.7`, `k = 5`, the
expected commit is `(1 − 0.7^6)/0.3 ≈ 2.94` — one verifier read yields ~2.9 tokens instead of 1. But I paid
~5 draft passes for that, so the *net* speedup is `2.94 / (1 + 5r)` where `r` is the draft-to-verifier cost
ratio. A tiny draft with `r ≈ 0.05` gives `2.94/1.25 ≈ 2.35×`; a 7B drafting for a 7B has `r ≈ 0.5`, so
`2.94/3.5 ≈ 0.84` — *slower* than not speculating. That single calculation is the whole shape of the rung:
speculation wins when `r` is small *and* the verifier read dominates. The extreme case is a 7B-int4 drafting
for a 70B-int4 — small `r`, enormous amortized read. A 70B in int4 is ~35 GB (bf16's ~140 GB would OOM the
80 GB card, but int4 fits), yet each un-speculated token reads all 35 GB, `35/2000 ≈ 17.5 ms`, capping near 57
tok/s ideal and less in practice; amortizing that read over ~3 tokens runs the 70B several-fold above its
un-speculated rate. The 70B is where the expensive resource is most expensive, so it's where amortization
pays most — the canonical pairing to validate, not a 7B verifier with a tiny draft (real but modest).

A precondition for correctness: draft and verifier must share the same tokenizer and vocabulary, since the
rule compares `p(x)` and `q(x)` for the *same* token id over the same 32000-symbol space. A 7B and 70B from
the same family satisfy this, another reason the same-family draft is natural. Both models keep their *own*
KV-caches — the draft advances its cache as it drafts, the verifier fills its cache in the single pass, and on
a rejection at position `j` both roll back to length `input_pos + j` so discarded drafts don't pollute the
next round. Getting this bookkeeping wrong wouldn't break the distribution proof (that's about
probabilities), but it would corrupt the context the next round conditions on, so it's a correctness detail I
keep straight even though it's orthogonal to the acceptance math.

The compile story flexes here. The verifier pass over a variable committed count and the host-side
accept/reject logic don't fit the single static CUDA graph the pure decode step used — speculation
deliberately varies how many positions each step processes. So `model_forward` is compiled separately in
reduce-overhead (the `k+1`-position pass is still worth capturing) and the speculative loop keeps `input_pos`
on the host to branch on accept/reject. That's a small concession of the launch-overhead win in exchange for
the multi-token amortization, worth it exactly when the verifier is expensive relative to the draft. (The
acceptance rule holds for whatever `p` and `q` are, so as long as both models apply the same temperature-0.8,
top-k-200 transform, the committed stream is the verifier's temperature-0.8 distribution; it degrades
gracefully at temperature 0 to exact greedy matching, and top-k gives `q(x) = 0` — certain rejection — for any
token the verifier would never emit.) I note the tree-of-candidates variant — drafting alternative
continuations checked in one wider pass — as the next place to squeeze accept-length, but it needs a custom
tree attention mask and the marginal gain over linear drafting is modest when `α` is high and `k` well-chosen,
so linear drafting is the right altitude for the scaffold's slots.

The prediction. This is the one rung where the headline depends entirely on the *pairing*, because the
speedup is the expected accepted length per verifier pass, set by `α`. For a well-matched pair the
amortization is large; for a poor one the draft passes are wasted and it can even slow down. Against the
196.80-tok/s int4 7B, drafting with a *tiny* model gives a modest combined gain — roughly 1.3× — because a 7B
verifier pass is already cheap (~4.4 GB), so there's less to amortize and the draft passes plus control flow
eat a bigger share. The dramatic wins are where the verifier is expensive: a 70B-int4 verifier drafted by a
7B-int4, where the absolute tok/s is lower (a 70B streaming ~35 GB per pass) but the *speedup factor* from
speculation is largest because the amortized thing is huge. The distribution guarantee is already proven, so
the only open question is the accept rate, not correctness. The change is `speculative_decode` with the
rejection-sampling acceptance rule and the draft/verifier split; full scaffold code in the answer.
