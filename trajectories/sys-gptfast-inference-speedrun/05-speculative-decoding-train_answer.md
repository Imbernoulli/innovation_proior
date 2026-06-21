int4+GPTQ reached 196.80 tok/s at 862 GB/s, and the quantization lever has now been pulled twice — bf16 → int8 → int4 — each pull buying less than the last ($1.48\times$, then $1.27\times$) as the achieved bandwidth steadily falls. I am running out of bits: below 4 starts costing real accuracy, and ever-finer groups' metadata eats into the byte savings. But the deeper issue is an assumption every rung so far has accepted without questioning: **one full forward pass per committed token.** Each decode step reads ~3.5 GB of int4 weights to produce a single token, with trivial arithmetic; the GPU does a memory-bandwidth-sized amount of work and gets one token for it.

The asymmetry to exploit is that a forward pass over *one* token and a pass over *several* tokens cost almost the same wall-clock, because both are dominated by reading the weights once — the weight read is amortized over however many positions you feed in. Processing 5 tokens in one pass is barely more than processing 1; the bytes streamed are the same and only the cheap arithmetic grows. So a single pass — one weight read — can verify many positions. The obstacle is the serial dependency of autoregression: I cannot compute token $t+5$ until I know $t+1$. *Unless* I guess the next several tokens cheaply and then check all the guesses in one pass. I propose **speculative decoding**: a small fast **draft** model proposes the next $k$ tokens one at a time, and the big **verifier** model runs *once* over all $k+1$ positions in parallel, giving its true next-token distribution at every position simultaneously. One expensive forward pass then yields the verifier's verdict on every draft guess at once.

The thing I cannot give up is the output distribution — this ladder has been "faster, not dumber," and a speedup that changes what the model would have said is a different model. So the acceptance rule must be *exactly* distribution-preserving, and the naive "accept if the draft's top token matches" violates that. The correct rule is rejection sampling against the two distributions. For each drafted position let $p$ be the draft's probability of the token it actually sampled and $q$ the verifier's probability of that same token. Accept the draft token with probability
$$\min\!\left(1,\ \frac{q}{p}\right).$$
If $q \ge p$ the verifier likes the token at least as much as the draft did, so always accept; if $q < p$ the draft over-favored it, so accept with probability $q/p$, and on rejection resample from the *residual* distribution $\operatorname{normalize}(\max(q-p,\,0))$ — exactly the mass the verifier wanted that the draft under-sampled. The algebra guarantees the committed token at each position is drawn from $q$, the verifier's true distribution, so the output stream is *provably identical in distribution* to plain verifier decoding. This is exact by construction, which is why this rung carries no quality caveat the way int4 did — no accuracy check is needed at all.

Two mechanical details earn their place. When *all* $k$ drafts are accepted I get a free *bonus* token: the verifier's pass already produced a distribution at position $k+1$, one past the last draft, so I sample it directly and commit $k+1$ tokens from one verifier pass. And `speculate_k` is the knob that trades off the gamble — larger $k$ means more tokens per pass *if* the draft keeps being right, but the draft's per-token agreement compounds, so fully-accepted runs get exponentially rarer as $k$ grows; past a point I am paying draft passes for tokens that will be rejected. The right $k$ depends on how closely the draft tracks the verifier, around 4–6 in practice. The compile story has to flex: the verifier pass over a variable number of positions and the host-side accept/reject loop do not fit the single static CUDA graph the pure decode step used, so `model_forward` is compiled separately in `reduce-overhead` mode and the speculative loop keeps `input_pos` on the host — a small concession of the launch-overhead win in exchange for the multi-token amortization, worth it exactly when the verifier is expensive relative to the draft.

The headline number depends entirely on the *pairing*, because the speedup is the expected accepted length per verifier pass, set by how often the draft agrees. For a 7B int4 verifier drafted by a tiny model the gain is modest (roughly $1.3\times$, into the 240s) because a 7B pass is already cheap, so there is little to amortize. The dramatic wins are where the verifier is *expensive*: a 70B-int4 verifier drafted by a 7B-int4, where one avoided 70B pass is worth a great deal — the absolute tok/s is lower (it is a 70B) but the *speedup factor* from speculation is the largest, because the amortized resource is huge. The distribution guarantee is already proven, so the only open question is the accept rate, never correctness.

```python
def speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs):
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
    draft_tokens = torch.cat(draft_tokens)
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device))
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob.  q>=p: always accept; q<p: q/p prob to accept
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
    if rejected_locations.shape[0] == 0:               # all accepted -> bonus token from q
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        model_forward(draft_model, draft_tokens[-1].view(1, -1), orig_input_pos + speculate_k)
        return torch.cat([draft_tokens, last_token])
    else:                                              # first rejection -> residual resample
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]; q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

# in main(): model_forward compiled separately (variable-length verifier pass);
#   the speculative loop keeps input_pos on the host.
if is_speculative:
    model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

# in generate(): drive the speculative loop with a draft model
while input_pos < T_new - 1:
    cur_token = next_token.view(())
    next_tokens = speculative_decode(model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs)
    accept_counts[len(next_tokens) - 1] += 1
    num_added = min(T_new - input_pos - 1, len(next_tokens))
    seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
    input_pos = input_pos + num_added
    next_token = next_tokens[-1]
# run: python generate.py --compile --checkpoint_path .../model.pth --draft_checkpoint_path .../draft.pth --speculate_k 5
```
