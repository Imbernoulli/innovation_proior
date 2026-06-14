I have a trained masked diffusion language model and I need to make it generate good text. Let me be precise about what the model actually gives me, because the whole problem lives in that one primitive. I feed it a sequence in which some positions hold real tokens and the rest hold a special `[MASK]` symbol, and in a single forward pass it returns, for every masked position simultaneously, a categorical distribution over the vocabulary — its guess for the clean token there, conditioned on every visible token, left and right. That is all it does. It is a bidirectional fill-in-the-blanks predictor, trained by a masked cross-entropy that is a bound on the data likelihood. So I am not generating left to right; I start from a response that is entirely `[MASK]` and I have to fill it in by repeated calls to this parallel predictor. The question is how.

Let me get the dynamics straight first, because the constraints come from there. The model defines its distribution through a reverse process that undoes a forward masking process. Forward, each token is independently masked with probability `t`, so `q_{t|0}(x_t^i = M) = t` and `q_{t|0}(x_t^i = x_0^i) = 1 - t`; the schedule is linear in `t`, which is the natural choice if I think of information as roughly proportional to the number of tokens, so I should lose it at a constant rate. At `t = 1` everything is masked; at `t = 0` everything is clean. Generation is the reverse: go from `t = 1` down to `t = 0`. The reverse kernel, for a step from time `t` down to `s` with `s < t`, factorizes across positions, and for one position it is: if the position is already unmasked, keep it (probability 1 it stays as is); if it is masked, with probability `s/t` it stays masked, and with probability `(t-s)/t` it gets filled, drawn from the data-prediction distribution `q_{0|t}(· | x_t)`. Written out,

  q_{s|t}(x_s^i | x_t) = 1                                if x_t^i ≠ M, x_s^i = x_t^i,
                       = s/t                              if x_t^i = M, x_s^i = M,
                       = ((t-s)/t) · q_{0|t}(x_s^i | x_t) if x_t^i = M, x_s^i ≠ M,
                       = 0                                otherwise.

Two things I want to lean on hard. First, once a position is unmasked it is frozen — the reverse kernel leaves a visible token unchanged, and the implementation removes already-visible positions from the transfer candidates. That is the carry-over property of absorbing diffusion: the forward chain, once a token hits the absorbing `[MASK]` state, stays there, and the reverse step mirrors this by not overwriting committed tokens. Second, the only thing I actually need from the network is `q_{0|t}`, the clean-token distribution at the masked positions; and for an absorbing process that distribution is time-free — it equals `p_data(x_0^i | x_t^{UM})`, a function of nothing but the currently-unmasked context. So my predictor is literally a masked-token classifier, and I do not even feed it `t`. Good — that means at any step I just call the model on the current partially-masked sequence and read off, per masked position, a distribution over tokens.

Now discretize. I split `[0,1]` into `N` equal steps, so each step goes from `t` to `s = t - 1/N`. At a step the kernel says: of the currently-masked positions, keep a fraction `s/t` masked and fill the fraction `(t-s)/t`. With the linear schedule and equal time steps, the *expected number* of positions that get filled per step is constant across steps — the mask probability drops by the same `1/N` each step, so the same expected count transitions each time. That is a clean budget: if the block has `m` masked positions and I have `N` steps, I fill about `m/N` positions per step. Let me hold that thought; the budget per step is essentially fixed and uniform, and I'll get the exact integer counts from a helper that just spreads `m mod N` extra unmasks over the first few steps.

So the *number* to fill per step is settled by the schedule. The real decisions are: which masked positions do I fill, and what token do I write in each. Let me separate the mathematical kernel from the implemented baseline, because otherwise I will fool myself. The exact kernel would independently leave each currently masked position masked with probability `s/t`; if a position is filled, it would sample the token from `q_{0|t}(· | x_t)`. The released decoder instead first proposes a token everywhere by the same path used for generation — argmax at temperature zero, or a Gumbel-perturbed argmax when temperature is nonzero — and then decides which proposals survive. In random remasking, that survival decision is uniform at random, so the mask-count marginal is right, but the token proposal is not an exact draw from the full reverse kernel when I use greedy mode. Let me run that in my head on a math problem. The model is asked to produce a chain of arithmetic. Step one, everything is masked; the model's per-position distributions are mostly flat, so many proposals are weak guesses. Random selection now commits, say, `m/N` of these guesses — scattered uniformly, some at positions where the model happened to be confident, but many at positions where it was near-uniform and the proposal is essentially noise. Those noisy commitments are frozen by carry-over. The next step conditions on them, and they can poison the context for everything downstream. Wall. Random selection gets the masking rate right but is wasteful: it commits tokens the model is unsure about and freezes the mistakes.

Let me stare at *why* it is wasteful, because the fix should come out of the diagnosis. The model handed me a full distribution at *every* masked position, and the random rule used none of that for position selection. But those distributions are not equal. At some positions the distribution is sharply peaked — the surrounding context already pins the token, the model is nearly certain. At others it is flat — genuinely ambiguous given what is visible. The carry-over freeze means a commitment is irreversible, so the *cost of being wrong* is identical everywhere, but the model-assigned risk varies sharply across positions. If I am forced to commit `m/N` positions this step and freeze them forever, I should commit the ones I am least likely to regret — the peaked ones — and leave the flat ones masked so that next step, after the certain tokens have sharpened the context, they might become certain too.

There is a second, sharper reason hiding in the parallel structure, and it makes the same prescription. The reverse step used for fast decoding factorizes across positions, so when I fill several positions in one step I am treating their proposals as conditionally independent given the current context. That is a tau-leaping approximation to the clean-data conditional I ultimately care about: once I commit position `i`, the right context for deciding position `j` may change. The approximation error grows with how many positions I commit at once and with how uncertain each one is — committing a peaked position barely moves the conditional picture, while committing a flat position is exactly the case where other positions could have changed the answer. So the tokenwise-independence error is *smallest precisely at the high-confidence positions*. Both arguments — irreversibility and independence error — point the same way: commit the positions the model is most sure about, defer the rest.

So I want a per-position *confidence* score and I want to commit the top-budget positions by that score. What is the score? The cleanest scalar is the model's own belief in the token it would write. In the default greedy case I run the predictor, take `x0^i = argmax_v p_θ(v | x_t)` at each masked position, and let the confidence be `c^i = p_θ(x0^i | x_t)`, the probability mass on that chosen token. A peaked distribution gives `c^i` near 1; a flat one gives `c^i` small. There are two equivalent ways to encode the carry-over freeze in this scoring: assign already-unmasked positions confidence 1 so they always survive the keep step, or — what I will actually implement with a transfer-index — score only the currently masked positions in the active block and set every other position's selection score to `-inf`. Then this step's rule is: among the masked positions in the active region, take the `k` with the highest `c^i`, where `k` is the schedule's budget for this step, write their proposed tokens, and leave everyone else masked. Next step, re-run, re-score, commit the next-most-confident `k`. The order that emerges is easy-first: the model fills the slots its context determines, those become context, and ambiguity collapses inward. That is the curriculum the random rule destroyed.

Let me sanity-check the token assignment, because I quietly switched from the exact kernel's token sample to the implementation's proposal. At temperature zero, the proposal is the mode. That is the annealed extreme of sampling. For autoregressive LMs it is well understood that greedy/low-temperature decoding suppresses diversity, which is bad for open-ended text but good when there is a single correct answer, because diversity there just means more ways to be wrong. On math and code I want the mode. So greedy token + commit-most-confident is a coherent package: it is the diffusion analogue of low-temperature autoregressive decoding. I should keep the code's temperature knob so I can re-inject diversity for open-ended generation: perturb the logits with the Gumbel race before the argmax; when temperature is nonzero, the selected token may not be the top-probability token, and its confidence is still the original model probability assigned to that selected token. But the default for accuracy is temperature zero.

Now, is "probability of the chosen token" really the right confidence, or is there something better? The natural competitor is the *margin* — the gap between the top-1 and top-2 predicted probabilities. The intuition for margin is that a high absolute probability can still be a near-tie (top-1 at 0.45, top-2 at 0.44 is "confident" by probability but genuinely ambiguous), whereas a large gap is an unambiguous decision. That is a real point. But the chosen-token probability is the quantity that most directly answers both of my diagnoses: the model-assigned chance of being wrong is `1 - c^i`, and the perturbation from committing this position should be smaller when the chosen token already carries most of the mass. The margin is a second-difference statistic — more sensitive to the exact shape of the runner-up tail and not as directly tied to "probability this commitment is a mistake." So I will take the selected-token probability as the primary signal; margin is a sensible alternative I could swap in, but it is not obviously better and it is fussier. At temperature zero this is exactly `p_θ(argmax)`.

I should double-check this confidence-keeping idea against a place it has already been made to work, so I am not reinventing a broken wheel. The exact move — run a masked parallel predictor, score each masked location by the predicted probability of its sampled token, keep only the most confident, remask and re-predict the rest over `T` iterations — is the iterative-decoding recipe for masked image transformers (Chang et al. 2022): Predict all masked locations in parallel; Sample a token at each and record its predicted probability as a confidence (unmasked positions get confidence `1.0`); compute how many to keep masked from a schedule; keep the most confident, remask the rest. "The model predicts all tokens simultaneously but only keeps the most confident ones." So the structure transfers. But two of their choices are tuned for *image diversity* and I should not import them blindly. They sample the token stochastically with temperature annealing — because in image synthesis they want diverse samples; I want the mode for deterministic tasks, so I take argmax (temperature toward zero) by default. And they use a *cosine* schedule for how many to keep masked — concave, "few confident predictions early, many late." My schedule is dictated by my generative model, not chosen for aesthetics: my forward masking probability is linear in `t`, so the principled per-step budget is the *uniform* one (equal expected count per step), and that is what the helper hands me. So I take their confidence-keeping skeleton but with a likelihood-consistent linear/uniform schedule and a greedy token, which is what my setting demands.

Now the regime question: do I decode the whole response as one parallel block, or carve it into pieces? Pure parallel decoding fills positions anywhere in the response, ordered only by confidence — fully bidirectional, maximally parallel, and best for open-ended continuation where there is no strong left-to-right backbone. But for long structured outputs — a multi-step derivation, a function body — a coarse left-to-right ordering helps: you want the early reasoning settled before the later steps that depend on it. So I allow a block size: split the generation region into consecutive blocks and process them strictly left to right, running the full confidence-keeping diffusion *within* each block before moving on. When the block length equals the generation length there is a single block and this is exactly fully-parallel decoding; when it is smaller it is semi-autoregressive — blocks in order, diffusion inside each. The constraint `gen_length % block_length == 0` keeps the blocks even, and I divide the total step budget evenly across blocks. One thing I must guard: when scoring confidences I have to forbid committing into positions of *future* blocks — set their confidence to `-inf` so the top-`k` never selects them — otherwise the parallel predictor, which sees the whole sequence, would happily fill ahead of the current block and break the left-to-right discipline. Inside the current block, carry-over already protects already-committed tokens (their confidence is effectively `-inf` for selection too, since I only ever score and select *masked* positions).

Let me also be careful with numerics, because the whole strategy is a ranking and a ranking is only as good as the scores. The released code casts to float64 in the optional Gumbel proposal path, because low-precision Gumbel-max can hurt generation quality. The confidence score itself is gathered from `softmax(logits)` in the logits' normal dtype. If I want a stricter local implementation I can compute that softmax in higher precision, but that would be a deliberate deviation from the canonical code, not the published mechanics.

One practical failure mode I should anticipate for an instruction-tuned model. If it was fine-tuned with heavy end-of-sequence padding, the model becomes very confident very early about emitting the end-of-sequence token — so under "commit-most-confident" it can commit too many end tokens and terminate the response almost immediately, collapsing accuracy. The fix is to keep that token from winning the early top-`k` competition — drive its confidence (or its logit) to the floor, setting it to `-inf` so it is never selected, on the tasks where premature termination is a problem. It's an option I want available, off by default for a clean base model.

Let me assemble the per-step loop concretely, filling the one empty slot in the decode harness. The harness already builds `x = [prompt | gen_length masks]`, splits into blocks, gives me the uniform per-step budget `num_xfer` from the schedule helper, and runs one forward pass per step. The slot is: from the logits, decide the commits.

  for each block b, left to right:
    compute the uniform per-step budget num_xfer over this block's masks
    for each step:
      mask_idx = positions still masked, restricted to block b
      if none left: break
      logits = model(x)                      # one forward pass, all positions
      logits_for_choice = add_gumbel_noise(logits, temperature)
      x0 = argmax_v logits_for_choice         # greedy at temperature zero, Gumbel choice otherwise
      p = softmax(logits)                     # confidence distribution from original logits
      conf = p[x0]                            # confidence = prob of the selected token
      for each position not in mask_idx: conf = -inf   # never (re)commit a frozen/out-of-block slot
      k = num_xfer[step]
      select the top-k positions by conf; write x0 there into x   # commit the most confident
      used += 1

That is the entire strategy: selected token, confidence = its model probability, commit the top-`k` most confident masked positions in the active block, repeat. Everything else — the mask layout, the block loop, the uniform budget, the one-forward-pass cost — was already given.

The slot becomes:

```python
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DemaskDecoder:
    """Low-confidence remasking: each step, commit the most-confident masked
    positions (default greedy token, confidence = its model probability); leave
    the rest masked and re-predict next step. Works fully-parallel or blockwise."""

    def __init__(self, mask_id, temperature=0.0):
        self.mask_id = mask_id
        self.temperature = temperature

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        # [prompt | gen_length masks]
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length          # == 1  -> fully parallel
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        used = 0
        for b in range(num_blocks):                      # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            # linear schedule -> uniform per-step unmask budget over this block
            num_xfer = get_num_transfer_tokens((x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m            # masks in the current block only
                if not mask_idx.any():
                    break

                logits = model(x).logits                 # one forward pass: all positions at once
                logits_for_choice = add_gumbel_noise(logits, self.temperature)
                x0 = torch.argmax(logits_for_choice, dim=-1)  # greedy at temperature zero
                p = F.softmax(logits, dim=-1)
                conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    c = conf[j].clone()
                    c[~mask_idx[j]] = -float("inf")      # frozen / out-of-block: never selected
                    _, topk = torch.topk(c, int(num_xfer[j, step].item()))   # most confident k
                    xfer[j, topk] = True
                x = torch.where(xfer, x0, x)             # commit; carry-over leaves the rest
                used += 1
        return x, used
```

Let me retrace the causal chain. The model is a one-shot parallel mask predictor; its distribution is defined by a reverse unmasking process whose kernel, per step, fills a schedule-fixed fraction of the masked positions and (by carry-over) freezes them forever. The exact reverse kernel would randomize both survival and token sampling, but the released decoder proposes tokens greedily by default and uses random remasking only as a position-selection baseline; that baseline gets the mask rate right while wasting the confidence information the predictor already produced. Both the irreversibility of the freeze and the independence-error argument say to commit the positions the model is most certain about, so I score each masked position by the predicted probability of the token selected for that position, take the greedy token at temperature zero, and commit the top-`k` by confidence, where `k` is the uniform per-step budget that the linear schedule dictates. This is the confidence-keeping iterative-decoding skeleton from masked image transformers, retargeted with a likelihood-consistent linear schedule and a greedy default token instead of a cosine schedule and stochastic image sampling. A block size turns it into semi-autoregressive decoding for long structured outputs (with future-block positions masked out of the selection) and degenerates to fully-parallel decoding when the block spans the whole response. Float64 belongs to the optional Gumbel proposal path in the canonical code, and EOS confidence/logit suppression handles the instruction-tuned premature-termination failure mode. The result fills the one empty slot in the decoder: selected token, confidence equal to its model probability, commit the most-confident masked positions each step.
