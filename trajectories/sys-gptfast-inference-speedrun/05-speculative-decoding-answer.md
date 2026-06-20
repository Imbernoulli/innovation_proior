**Problem (from baseline).** int4+GPTQ reached 196.80 tok/s, and the quantization lever is nearly spent
(below 4 bits costs accuracy). Every rung so far assumed one full verifier forward pass — one ~3.5 GB
weight read — per committed token. A pass over *k* positions costs almost the same as a pass over 1 (both
are dominated by the single weight read), so a verifier pass can confirm many tokens for the price of one;
the serial dependency is what blocks it.

**Key idea.** **Speculative decoding.** A small fast **draft** model proposes the next *k* tokens
sequentially; the big **verifier** runs *once* in parallel over all *k+1* positions, giving its true
next-token distribution at each. Commit drafted tokens via exact rejection sampling: accept token *t* with
prob `min(1, q_t/p_t)` (q = verifier prob, p = draft prob); on the first rejection, resample from the
residual `normalize(max(q−p, 0))` and stop. If all *k* accept, also take a free bonus token from the
verifier's (k+1)-th distribution.

**Why it works.** The acceptance rule makes the committed stream **provably identical in distribution** to
plain verifier decoding — exact by construction, so no quality check is needed (unlike the quantization
rungs). The expensive resource (the verifier's weight read) is amortized over the expected accepted run
length per pass, which is set by how often the draft agrees. `speculate_k` trades more-tokens-per-pass
against exponentially rarer full-accept runs. The win is largest when the verifier is *expensive*
relative to the draft (e.g. 70B verifier, 7B draft), where one avoided big-model pass is worth the most.

**Change / code.** `speculative_decode` in `generate.py` (draft loop → one verifier pass → rejection
sampling), wired into `generate` with a draft model.

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
