196.80 tokens/s at 862 GB/s. The quantization lever has now been pulled twice — bf16 → int8 → int4 — and
each pull bought less than the last: `155.58/104.9 ≈ 1.48×`, then `196.80/155.58 ≈ 1.27×`. Let me read the
862 back to a model size to confirm the mechanism one more time: `862 / 196.80 ≈ 4.38 GB` per token, which
matches the ~4.5 GB int4 model I estimated. So the story is intact and so is the diminishing return, and
the diminishing return has a clear cause: as I shrink the quantized bulk, the un-quantized bf16 tail
(embeddings, head, norms, ~0.5 GB) becomes a larger *fraction* of the token, and the metadata overhead of
ever-finer groups eats into the byte savings. Extrapolating the width lever: going below 4 bits starts
costing real accuracy (my SNR estimate degrades ~2× per bit and GPTQ can't manufacture resolution that
isn't there), and the metadata fraction climbs. The representation lever is nearly spent. More importantly,
every rung so far has accepted one assumption without questioning it: **one full forward pass per committed
token.** That assumption is the deepest source of the inefficiency, and I haven't touched it. Each decode
step reads ~4.4 GB of int4 weights to produce *one* token. The matmuls are trivial — still ~150× to the
left of the roofline ridge — so almost all the time is the weight read. The GPU is doing a
memory-bandwidth-sized amount of work and getting a single token for it.

Here's the asymmetry I want to exploit, and I'll quantify it because it's the entire basis of the rung. A
forward pass on *one* token and a forward pass on *several* tokens at once cost almost the same wall-clock,
because both are dominated by reading the weights once — the weight read is amortized across however many
positions I feed in. Concretely: a pass over `k+1` positions reads the same ~4.4 GB of weights as a pass
over 1, and does `(k+1)×` the arithmetic — but the arithmetic was ~45 µs against a ~6.75 ms weight read, so
even at `k=5` the arithmetic climbs to ~270 µs, still under 5% of the weight-read time. So a pass over 6
positions costs maybe `6.75 ms + 0.27 ms ≈ 7.0 ms` instead of `6.75 ms` — barely more than a single-token
pass, yet it evaluates the model at *six* positions. The bottleneck is bytes-per-*pass*, not
bytes-per-token, and a single pass can verify many positions for the price of one weight read. The problem
is that autoregressive decoding seems to forbid this: I can't process token `t+5` until I know token `t+1`,
which depends on `t`, and so on. The dependency is serial, and that serial dependency is exactly what's
been forcing one weight read per token.

Unless I can *guess* the next several tokens cheaply and then check all the guesses in a single pass. Let
me walk the design space of "guess," because the quality of the guess is what determines the payoff. One
option is a cheap statistical guesser — an n-gram model, or retrieval from the prompt — appended as
candidate continuations; cheap, but its agreement with the big model is poor except on boilerplate, so the
guesses get rejected and the amortization is small. Another is to add extra prediction heads to the model
that propose several tokens at once from one hidden state; powerful, but it changes the model, needs
training, and I'm confined to the frozen weights and the scaffold's slots. The option that fits the
constraints and has the best guess-quality-per-cost is a small, fast **draft** model that proposes the
next `k` tokens one at a time — cheap because it's small (say the int4 7B drafting for an int4 70B verifier,
or a tiny model drafting for the 7B), and its agreement with the verifier is high because it's a smaller
version of the same kind of model trained on the same kind of data. I take its `k` guesses, append them to
the current token, and run the big **verifier** model *once* over all `k+1` positions in parallel. That one
big-model pass — one weight read — gives me the verifier's true next-token distribution at *every* one of
those positions simultaneously. Now I have the draft's guesses and the verifier's verdict on each, from a
single expensive forward pass. If the draft guessed well, I commit several tokens for the price of one
verifier pass; if it guessed badly, I fall back. Either way the expensive resource — the verifier's weight
read — is amortized over multiple tokens.

The thing I absolutely cannot give up is the output distribution. This whole ladder has been "faster, not
dumber"; a speedup that changes what the model would have said is a different model, not a faster one. The
quantization rungs paid for their bytes with a bounded, validated quality cost; this rung must pay
*nothing*, because I can make it exact. So the acceptance rule has to be *exactly* distribution-preserving:
the stream of committed tokens must be distributed identically to what the verifier alone would have
produced. That's a real constraint, and it kills the naive schemes — "accept if the draft's top token
matches the verifier's top token" would bias the output toward the modes of both distributions and change
the sampling temperature in effect. I need a rule that provably reproduces `q`, the verifier's
distribution, at every committed position.

The correct rule is rejection sampling against the two distributions. For each drafted position let `p` be
the draft model's probability of the token it actually sampled there, and `q` the verifier's probability of
that same token. Accept the draft token with probability `min(1, q/p)`. If `q ≥ p` the verifier likes the
token at least as much as the draft did — always accept. If `q < p` the draft over-favored it — accept with
probability `q/p`, and on rejection, resample from the *residual* distribution `normalize(max(q − p, 0))`,
which is exactly the mass the verifier wanted that the draft under-sampled.

Let me prove it preserves `q`, because the whole rung rests on this being exact and I won't take it on
faith. Fix a position and a token `x`. The committed token equals `x` in two mutually exclusive ways.
Either the draft sampled `x` (probability `p(x)`) and I accepted it (probability `min(1, q(x)/p(x))`); or
the draft sampled something and I rejected it, then resampled `x` from the residual. The accept
contribution is `p(x)·min(1, q(x)/p(x)) = min(p(x), q(x))`. For the reject-then-resample contribution:
the total probability of a rejection is `Σ_y p(y)·(1 − min(1, q(y)/p(y))) = Σ_y (p(y) − min(p(y), q(y))) =
Σ_y max(p(y) − q(y), 0)`, and the residual distribution I draw from is `max(q(x) − p(x), 0) / Σ_z max(q(z)
− p(z), 0)`. Now note `Σ_y max(p(y) − q(y), 0) = Σ_z max(q(z) − p(z), 0)` — the mass `p` has in excess of
`q` exactly equals the mass `q` has in excess of `p`, since both `p` and `q` sum to 1. So the rejection
total and the residual normalizer cancel, and the reject-resample contribution to committing `x` is just
`max(q(x) − p(x), 0)`. Add the two paths: `min(p(x), q(x)) + max(q(x) − p(x), 0)`. If `q(x) ≥ p(x)` this is
`p(x) + (q(x) − p(x)) = q(x)`; if `q(x) < p(x)` it's `q(x) + 0 = q(x)`. Either way it's `q(x)`. The
committed token is drawn from the verifier's distribution exactly — *provably identical in distribution* to
plain verifier decoding, for any draft `p`, even a bad one. So the draft's quality affects only *speed*
(the accept rate), never *correctness*. That's why this rung carries no quality caveat the way int4 did: it
needs no accuracy check, it's exact by construction.

Walking the mechanics: draft `k` tokens sequentially with the small model, run the verifier once over the
`k+1` positions, then sweep the acceptance test position by position and stop at the first rejection:

```python
def speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs):
    # draft k tokens sequentially with the small model
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
    draft_tokens = torch.cat(draft_tokens)
    # one parallel verifier pass over [cur_token, draft_1..draft_k]
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device))
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: verifier prob, p: draft prob, of the drafted tokens
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)         # min(1, q/p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:        # all k accepted -> also take a free bonus token from q
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        model_forward(draft_model, draft_tokens[-1].view(1, -1), orig_input_pos + speculate_k)
        return torch.cat([draft_tokens, last_token])
    else:                                       # first rejection -> resample from the residual
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]; q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])
```

Two details earn their place. When *all* `k` drafts are accepted, I get a free *bonus* token: the
verifier's pass already produced a distribution at position `k+1` (one past the last draft — the verifier
saw `[cur_token, draft_1..draft_k]` and its last output is the prediction *after* `draft_k`), so I sample
it directly from `target_probs[-1]`, and I run one draft-model step to keep the draft's own KV-cache in
sync for the next round. That gives `k+1` tokens committed from one verifier pass in the best case — a
subtle but real bonus, because it means a perfect draft run yields one more token than it drafted. The
second detail is why I stop at the *first* rejection rather than trying to salvage later accepted tokens:
once a position is resampled from the residual, the tokens after it were drafted conditioned on a token
that changed, so they're no longer valid continuations — I must discard them and restart drafting from the
resampled token. Sweeping to the first rejection and truncating is what keeps the distribution exact; a
"gather all accepted positions" shortcut would break the conditioning.

`speculate_k` is the knob that trades off the gamble, and it has a clean structure. If the draft agrees
with the verifier on any given token with probability `α` (the acceptance rate), then the probability of
accepting a *run* of `j` drafts is roughly `α^j` — agreement compounds multiplicatively — so accepted runs
get exponentially rarer as `k` grows. The expected number of tokens committed per verifier pass is a
truncated geometric: `E[accepted] = (1 − α^{k+1})/(1 − α)` in the idealized independent-accept model. That
formula is exactly the shape of the trade: as `k → ∞` the expected commit saturates at `1/(1−α)` (for
α=0.8, about 5 tokens), so beyond a point I'm drafting tokens that will almost surely be rejected and
paying draft passes for nothing. Differentiating the *speedup* — expected tokens per unit cost, where cost
is `k` draft passes plus one verifier pass — gives an interior optimum `k` that depends on `α` and on the
draft/verifier cost ratio; in practice the scripts use `speculate_k ≈ 4-6`, which is right where the
geometric saturates for a well-matched pair without over-drafting.

Let me put a concrete number on the amortization so the "worth it when the verifier is expensive" claim
isn't hand-waving. Suppose the draft agrees with the verifier at `α = 0.7` and I set `k = 5`. The expected
committed tokens per verifier pass is `(1 − 0.7^6)/(1 − 0.7) = (1 − 0.118)/0.3 ≈ 2.94`. So one verifier
weight read yields ~2.9 tokens instead of 1 — nearly a 3× amortization of the expensive read. The catch is
the cost of drafting: I paid ~5 small draft passes to get there, so the *net* speedup is
`2.94 / (1 + 5·r)` where `r` is the draft-to-verifier cost ratio. If the draft is a tiny model with `r ≈
0.05`, the denominator is `1.25` and the net speedup is ~2.35× on the verifier's own rate; if the draft is
itself a 7B (r ≈ 0.5 against a 7B verifier, since they're the same size) the denominator balloons to `3.5`
and the net is `2.94/3.5 ≈ 0.84` — *slower* than not speculating. That single calculation explains the
whole shape of the rung: speculation wins big when `r` is small (cheap draft) *and* the verifier read is
what dominates (expensive verifier). A 7B drafting for a 7B is the worst case; a tiny draft for a 7B is a
modest win; a 7B-int4 drafting for a 70B-int4 is where `r` is small *and* the amortized read is enormous,
so the speedup factor is largest even though the 70B's absolute tok/s is low.

That points at what speculation *uniquely* enables beyond raw speedup: serving a model whose per-token read
is otherwise punishing. A 70B in int4 is ~35 GB of weights — it fits on the 80 GB card (bf16 would be ~140
GB and OOM, but int4 fits), yet each un-speculated token reads all 35 GB, `35/2000 ≈ 17.5 ms`, capping it
near 57 tok/s ideal and far less in practice. Speculation amortizes that 35 GB read over ~3 tokens, so a
70B verifier drafted by a fast 7B runs several-fold above its un-speculated rate. The 70B is the case where
the expensive resource is most expensive, so it's where amortization pays most — which is exactly why the
canonical pairing to validate is a 70B verifier with a 7B draft, not a 7B verifier with a tiny draft (that
one is real but modest).

A precondition I have to respect for any of this to be correct: the draft and verifier must share the same
tokenizer and vocabulary. The acceptance rule compares `p(x)` and `q(x)` for the *same* token id `x`, so
the two models' output distributions have to be over the same 32000-symbol space, or `q/p` is meaningless.
A 7B and a 70B from the same model family satisfy this — same vocab, same tokenization — which is another
reason the same-family draft is the natural choice over an arbitrary small model. And both models keep
their *own* KV-caches: the draft advances its cache as it drafts `k` tokens, the verifier fills its cache
for the `k+1` positions in its single pass, and on a rejection at position `j` both caches must be rolled
back to length `input_pos + j` so the discarded drafts don't pollute the next round. The scaffold's loop
handles the verifier side by writing only the committed tokens into `seq` and advancing `input_pos` by the
accepted count; the draft's extra step in the all-accept branch keeps its cache aligned. Getting this cache
bookkeeping wrong wouldn't break the distribution proof — that's about probabilities — but it would corrupt
the context the next round conditions on, so it's a correctness detail I have to keep straight even though
it's orthogonal to the acceptance math.

The compile story has to flex a little for this rung, and I want to be honest about what I'm giving up. The
verifier pass over a variable number of positions (`k+1`, but the *committed* count varies, so the loop's
per-iteration shape isn't fixed) and the host-side accept/reject logic don't fit the single static CUDA
graph the pure decode step used — the whole point of that graph was one fixed-shape step replayed
identically, and speculation deliberately varies how many positions each step processes. So `model_forward`
is compiled separately in reduce-overhead mode (the verifier's `k+1`-position pass is still worth
capturing) and the speculative loop keeps `input_pos` on the host so it can branch on accept/reject. That's
a small concession of the launch-overhead win — I reintroduce a little host-side control flow — in exchange
for the multi-token amortization, and it's worth it exactly when the verifier is expensive relative to the
draft, because then the amortized weight read dwarfs the reintroduced overhead.

One interaction I should check is the sampler's temperature and top-k, since the acceptance rule is stated
in terms of `p` and `q` and those are the *post*-temperature, post-top-k distributions. The proof above
holds for whatever `p` and `q` actually are, so as long as the draft and verifier both apply the same
temperature-0.8, top-k-200 transform before I read off `p(x)` and `q(x)`, the committed stream is drawn
from the verifier's temperature-0.8 distribution — which is exactly the target. There's a subtlety at the
extremes: at temperature 0 (greedy) both distributions collapse to point masses, `q/p` is 0 or 1, and
acceptance becomes "accept iff the draft's argmax equals the verifier's argmax" — the rule degrades
gracefully to exact greedy matching. And top-k on the verifier means `q(x) = 0` for tokens outside its top
200, so any draft token the verifier would never emit is rejected with certainty (`min(1, 0/p) = 0`), which
is correct. The rule composes cleanly with the sampling settings; I don't need a special case.

I briefly consider the more elaborate variant of drafting a *tree* of candidates rather than a single
linear sequence — proposing several alternative continuations at each step and letting the verifier check
them all in one (wider) pass, which raises the expected accepted length for a given verifier pass. It's a
real lever, but it complicates the verifier pass (a custom attention mask over the tree) and the acceptance
bookkeeping, and the marginal gain over linear drafting is modest when `α` is already high and `k` is
well-chosen. For this rung the linear draft is the right altitude: it captures almost all of the
amortization with a single extra model and a clean, provably-exact acceptance rule, and it fits the
scaffold's slots without a bespoke masked-attention kernel. I note the tree variant as the next place to
look if I later needed to squeeze the accept-length further, but linear drafting is what I'm committing to
here.

The prediction. This is the one rung where the headline number depends entirely on the *pairing*, because
the speedup is the expected accepted length per verifier pass — set by `α`, how often the draft agrees with
the verifier. For a verifier and draft that track closely (high `α`) the amortization is large; for a
poorly matched pair the draft passes are wasted and it can even slow down (if `α` is low, `E[accepted]` is
near 1 and I've paid `k` draft passes for one token). Against the 196.80-tok/s int4 7B, drafting with a
*tiny* model gives a modest combined gain — roughly 1.3×, into the 240s — because a 7B verifier pass is
already cheap (only ~4.4 GB), so there's less to amortize and the draft passes plus the control flow eat a
bigger share. The dramatic wins are where the verifier is *expensive*: a 70B-int4 verifier drafted by a
7B-int4, where one avoided 70B weight read is worth a great deal — there the absolute tok/s is lower (it's
a 70B streaming ~35 GB per pass) but the *speedup factor* from speculation is the largest, because the
thing being amortized is huge. The result I'd validate is the realized tok/s for a concrete pairing on the
benchmark; the distribution guarantee is already proven above, so the only open question is the accept
rate, not correctness. The change is `speculative_decode` with the rejection-sampling acceptance rule and
the draft/verifier split; full scaffold code in the answer.
