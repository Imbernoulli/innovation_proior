196.80 tokens/s at 862 GB/s. The quantization lever has now been pulled twice — bf16 → int8 → int4 — and
each pull bought less than the last (1.48×, then 1.27×), with the achieved bandwidth steadily falling.
I'm running out of bits: going below 4 would start costing real accuracy, and the metadata overhead of
ever-finer groups eats into the byte savings. More importantly, every rung so far has accepted one
assumption without questioning it: **one full forward pass per committed token.** That assumption is the
deepest source of the inefficiency, and I haven't touched it. Each decode step reads ~3.5 GB of int4
weights to produce *one* token. The matmuls are trivial; almost all the time is the weight read. So the
GPU is doing a memory-bandwidth-sized amount of work and getting a single token for it.

Here's the asymmetry I want to exploit. A forward pass on *one* token and a forward pass on *several*
tokens at once cost almost the same wall-clock, because both are dominated by reading the weights once —
the weight read is amortized across however many positions you feed in. Processing 5 tokens in one pass
isn't 5× the time of processing 1; it's barely more, because the bytes streamed are the same and only the
(cheap) arithmetic grows. The bottleneck is bytes-per-*pass*, and a pass can verify many positions for
the price of one weight read. The problem is that autoregressive decoding seems to forbid this: I can't
process token `t+5` until I know token `t+1`, which depends on `t`, and so on. The dependency is serial.

Unless I can *guess* the next several tokens cheaply and then check all the guesses in a single pass. If I
have a small, fast **draft** model that proposes the next `k` tokens one at a time (it's cheap because
it's small — say the int4 7B drafting for an int4 70B verifier, or a tiny model drafting for the 7B), I
can take its `k` guesses, append them to the current token, and run the big **verifier** model *once* over
all `k+1` positions in parallel. That one big-model pass — one weight read — gives me the verifier's true
next-token distribution at *every* one of those positions simultaneously. Now I have the draft's guesses
and the verifier's verdict on each, from a single expensive forward pass. If the draft guessed well, I
commit several tokens for the price of one verifier pass; if it guessed badly, I fall back. Either way the
expensive resource — the verifier's weight read — is amortized over multiple tokens.

The thing I absolutely cannot give up is the output distribution. This whole ladder has been "faster, not
dumber"; a speedup that changes what the model would have said is a different model, not a faster one. So
the acceptance rule has to be *exactly* distribution-preserving: the stream of committed tokens must be
distributed identically to what the verifier alone would have produced. That's a real constraint, and
naive "accept if the draft's top token matches" violates it.

The correct rule is rejection sampling against the two distributions. For each drafted position let `p`
be the draft model's probability of the token it actually sampled there, and `q` the verifier's
probability of that same token. Accept the draft token with probability `min(1, q/p)`. If `q ≥ p` the
verifier likes the token at least as much as the draft did — always accept. If `q < p` the draft
over-favored it — accept with probability `q/p`, and on rejection, resample from the *residual*
distribution `normalize(max(q − p, 0))`, which is exactly the mass the verifier wanted that the draft
under-sampled. This is the standard speculative-sampling correction, and the algebra guarantees the
committed token at each position is drawn from `q`, the verifier's true distribution — so the output is
*provably identical in distribution* to plain verifier decoding. No accuracy check needed; it's exact by
construction, which is why this rung carries no quality caveat the way int4 did.

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
verifier's pass already produced a distribution at position `k+1` (one past the last draft), so I sample
it directly — `k+1` tokens committed from one verifier pass. And `speculate_k` is the knob that trades
off the gamble: larger `k` means more tokens per verifier pass *if* the draft keeps being right, but the
draft's per-token agreement compounds, so accepted runs get exponentially rarer as `k` grows — past a
point I'm drafting tokens that will be rejected and paying draft passes for nothing. The right `k` depends
on how well the draft tracks the verifier (the scripts use `speculate_k` ≈ 4-6).

The compile story has to flex a little for this rung: the verifier pass over a variable number of
positions and the host-side accept/reject loop don't fit the single static CUDA graph the pure decode
step used, so `model_forward` is compiled separately in reduce-overhead mode and the speculative loop
keeps `input_pos` on the host. That's a small concession of the launch-overhead win in exchange for the
multi-token amortization — worth it exactly when the verifier is expensive relative to the draft.

The prediction. This is the one rung where the headline number depends entirely on the *pairing*, because
the speedup is the expected accepted length per verifier pass — set by how often the draft agrees with the
verifier. For a verifier and draft that track closely the amortization is large; for a poorly matched pair
the draft passes are wasted and it can even slow down. Against the 196.80-tok/s int4 7B, drafting with a
*tiny* model gives a modest combined gain (roughly 1.3×, into the 240s) because a 7B verifier pass is
already cheap, so there's less to amortize. The dramatic wins are where the verifier is *expensive*: a
70B-int4 verifier drafted by a 7B-int4, where one avoided 70B pass is worth a lot — there the absolute
tok/s is lower (it's a 70B) but the *speedup factor* from speculation is the largest, because the thing
being amortized is huge. The result I'd validate is the realized tok/s for a concrete pairing on the
benchmark; the distribution guarantee is already proven, so the only open question is the accept rate, not
correctness. The change is `speculative_decode` with the rejection-sampling acceptance rule and the
draft/verifier split; full scaffold code in the answer.
