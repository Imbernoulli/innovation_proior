I have a trained masked diffusion language model and I need to make it generate good text. Let me be precise about what the model actually gives me, because the whole problem lives in that one primitive. I feed it a sequence in which some positions hold real tokens and the rest hold a special `[MASK]` symbol, and in a single forward pass it returns, for every masked position simultaneously, a categorical distribution over the vocabulary — its guess for the clean token there, conditioned on every visible token, left and right. That is all it does. It is a bidirectional fill-in-the-blanks predictor, trained by a masked cross-entropy that is a bound on the data likelihood. So I am not generating left to right; I start from a response that is entirely `[MASK]` and I have to fill it in by repeated calls to this parallel predictor. The question is how.

Let me get the dynamics straight first, because the constraints come from there. The model defines its distribution through a reverse process that undoes a forward masking process. Forward, each token is independently masked with probability `t`, so `q_{t|0}(x_t^i = M) = t` and `q_{t|0}(x_t^i = x_0^i) = 1 - t`; the schedule is linear in `t`, which is the natural choice if I think of information as roughly proportional to the number of tokens, so I should lose it at a constant rate. At `t = 1` everything is masked; at `t = 0` everything is clean. Generation is the reverse: go from `t = 1` down to `t = 0`. The reverse kernel, for a step from time `t` down to `s` with `s < t`, factorizes across positions, and for one position it is: if the position is already unmasked, keep it (probability 1 it stays as is); if it is masked, with probability `s/t` it stays masked, and with probability `(t-s)/t` it gets filled, drawn from the data-prediction distribution `q_{0|t}(· | x_t)`. Written out,

  q_{s|t}(x_s^i | x_t) = 1                                if x_t^i ≠ M, x_s^i = x_t^i,
                       = s/t                              if x_t^i = M, x_s^i = M,
                       = ((t-s)/t) · q_{0|t}(x_s^i | x_t) if x_t^i = M, x_s^i ≠ M,
                       = 0                                otherwise.

Two things I want to lean on hard. First, once a position is unmasked it is frozen — the reverse kernel leaves a visible token unchanged, and the implementation removes already-visible positions from the transfer candidates. That is the carry-over property of absorbing diffusion: the forward chain, once a token hits the absorbing `[MASK]` state, stays there, and the reverse step mirrors this by not overwriting committed tokens. Second, the only thing I actually need from the network is `q_{0|t}`, the clean-token distribution at the masked positions; and for an absorbing process that distribution is time-free — it equals `p_data(x_0^i | x_t^{UM})`, a function of nothing but the currently-unmasked context. So my predictor is literally a masked-token classifier, and I do not even feed it `t`. Good — that means at any step I just call the model on the current partially-masked sequence and read off, per masked position, a distribution over tokens.

Now discretize. I split `[0,1]` into `N` equal steps, so each step goes from `t` to `s = t - 1/N`. At a step the kernel says: of the currently-masked positions, keep a fraction `s/t` masked and fill the fraction `(t-s)/t`. I want to know whether the per-step fill count is actually constant across steps or just roughly so, because if it drifts I will have to track it. Let me just push the numbers through. Take a block of `m = 12` masks and `N = 4` steps. At step `k` (going from `t = 1 - k/N` down to `s = 1 - (k+1)/N`) the fraction filled of the currently-masked set is `(t-s)/t = (1/N)/(1 - k/N)`, and the currently-masked count is `m·(1 - k/N)`, so the expected fill is `m·(1/N)/(1 - k/N)·(1 - k/N) = m/N`. The `(1 - k/N)` cancels. Carrying it out step by step gives expected fills `[3.0, 3.0, 3.0, 3.0]`, summing to 12. So it is exactly `m/N = 3` every step, not approximately — the linear schedule makes the expected transfer count genuinely uniform, and the integer helper that hands me `[3,3,3,3]` (spreading any `m mod N` remainder over the first few steps) is faithful to that. Good: the budget per step is fixed and uniform, and I do not have to think about it again.

So the *number* to fill per step is settled by the schedule. The real decisions are: which masked positions do I fill, and what token do I write in each. Let me separate the mathematical kernel from the implemented baseline, because otherwise I will fool myself. The exact kernel would independently leave each currently masked position masked with probability `s/t`; if a position is filled, it would sample the token from `q_{0|t}(· | x_t)`. The released decoder instead first proposes a token everywhere by the same path used for generation — argmax at temperature zero, or a Gumbel-perturbed argmax when temperature is nonzero — and then decides which proposals survive. In random remasking, that survival decision is uniform at random, so the mask-count marginal is right, but the token proposal is not an exact draw from the full reverse kernel when I use greedy mode.

Let me run random remasking on a small concrete case to see what actually goes wrong, rather than just assert it does. Take a 4-position block, vocabulary `{a,b,c}`, and step one with everything masked. Suppose the predictor returns these per-position distributions over `(a,b,c)`:

  position 0: (0.90, 0.05, 0.05)   — sharply peaked, context pins it
  position 1: (0.40, 0.35, 0.25)   — flat
  position 2: (0.34, 0.33, 0.33)   — essentially uniform, a guess
  position 3: (0.04, 0.05, 0.91)   — sharply peaked

The budget says fill `k = 2` this step. The greedy proposals and their top-token probabilities are `0→a (0.90)`, `1→a (0.40)`, `2→a (0.34)`, `3→c (0.91)`. Random remasking ignores those numbers and picks 2 of the 4 positions uniformly — one of the six equally-likely pairs is `{1,2}`. If it draws `{1,2}` it commits `a` at position 1 (the model gave that token 0.40) and `a` at position 2 (0.34, barely above chance). Those two near-noise tokens are now frozen by carry-over, and every later step conditions on them. So the concrete failure is exactly this: random selection commits tokens the model is unsure about and can never take them back. It got the mask rate right and threw away the per-position certainty the predictor handed me for free.

Let me stare at *why* that is wasteful, because the fix should come out of the diagnosis. The model handed me a full distribution at *every* masked position, and the random rule used none of that for position selection. But those distributions are not equal. At some positions the distribution is sharply peaked — the surrounding context already pins the token, the model is nearly certain. At others it is flat — genuinely ambiguous given what is visible. The carry-over freeze means a commitment is irreversible, so the *cost of being wrong* is identical everywhere, but the model-assigned risk varies sharply across positions. If I am forced to commit `m/N` positions this step and freeze them forever, the natural thing is to commit the ones I am least likely to regret — the peaked ones — and leave the flat ones masked so that next step, after the certain tokens have sharpened the context, they might become certain too. On the toy example that would commit positions 3 and 0 (probabilities 0.91 and 0.90) and defer 1 and 2 — the opposite of the bad random draw above.

There is a second reason hiding in the parallel structure, and I want to check whether it actually points the same way or whether I am just rationalizing the first one. The reverse step used for fast decoding factorizes across positions, so when I fill several positions in one step I am treating their proposals as conditionally independent given the current context. That is a tau-leaping approximation: once I commit position `i`, the right context for deciding position `j` may change, and the independent sample did not account for it. The claim I want to test is that this approximation error is *smaller* when the committed position is high-confidence. Let me build a two-position joint and measure the drift. Positions `i, j`, tokens `{a, b}`. Case A, `i` peaked: joint `p(x_i,x_j)` = `[[0.50, 0.45], [0.04, 0.01]]`, so the marginal of `i` is `(0.95, 0.05)` — committing `i` is a near-sure bet. Case B, `i` flat: joint `[[0.40, 0.10], [0.10, 0.40]]`, marginal of `i` is `(0.50, 0.50)` — a coin flip, and `j` is strongly coupled to `i`. In each case parallel decoding would write `j` from its *marginal*, but the truth after I commit `i = argmax` is the *conditional* `p(x_j | i = i*)`. The error is how far those differ.

Case A: marginal of `j` is `(0.54, 0.46)`; conditioning on the committed `i = a` gives `(0.526, 0.474)`; total-variation drift `0.014`. Case B: marginal of `j` is `(0.50, 0.50)`; conditioning on `i = a` gives `(0.80, 0.20)`; drift `0.30`. So committing the high-confidence position moved the neighbor's distribution by 0.014, committing the flat one moved it by 0.30 — a factor of about twenty. The independence error really is smallest precisely at the confident positions, and it is largest exactly where committing freezes a guess that *should* have waited for its neighbor. Both arguments — irreversibility and independence error — land in the same place: commit the positions the model is most sure about, defer the rest.

So I want a per-position *confidence* score and I want to commit the top-budget positions by that score. What is the score? The cleanest scalar is the model's own belief in the token it would write. In the default greedy case I run the predictor, take `x0^i = argmax_v p_θ(v | x_t)` at each masked position, and let the confidence be `c^i = p_θ(x0^i | x_t)`, the probability mass on that chosen token. A peaked distribution gives `c^i` near 1; a flat one gives `c^i` small. There are two equivalent ways to encode the carry-over freeze in this scoring: assign already-unmasked positions confidence 1 so they always survive the keep step, or — what I will actually implement with a transfer-index — score only the currently masked positions in the active block and set every other position's selection score to `-inf`. Then this step's rule is: among the masked positions in the active region, take the `k` with the highest `c^i`, where `k` is the schedule's budget for this step, write their proposed tokens, and leave everyone else masked. Next step, re-run, re-score, commit the next-most-confident `k`. On the toy block this fills positions 3 and 0 first (the two `0.9`-ish slots), and only after their tokens are in context does it re-score 1 and 2 — which by then may have sharpened. The order that emerges is easy-first: the model fills the slots its context determines, those become context, and ambiguity collapses inward. That is the curriculum the random rule destroyed.

Let me sanity-check the token assignment, because I quietly switched from the exact kernel's token sample to the implementation's proposal. At temperature zero, the proposal is the mode. That is the annealed extreme of sampling. For autoregressive LMs it is well understood that greedy/low-temperature decoding suppresses diversity, which is bad for open-ended text but good when there is a single correct answer, because diversity there just means more ways to be wrong. On math and code I want the mode. So greedy token + commit-most-confident is a coherent package: it is the diffusion analogue of low-temperature autoregressive decoding. I should keep the code's temperature knob so I can re-inject diversity for open-ended generation: perturb the logits with the Gumbel race before the argmax; when temperature is nonzero, the selected token may not be the top-probability token, and its confidence is still the original model probability assigned to that selected token. But the default for accuracy is temperature zero.

Now, is "probability of the chosen token" really the right confidence, or is there something better? The natural competitor is the *margin* — the gap between the top-1 and top-2 predicted probabilities. The intuition is that a high absolute probability can still be a near-tie that is genuinely ambiguous, whereas a large gap is an unambiguous decision. I should take this seriously rather than wave it away, because if margin is better I want to know now. Let me construct a case where the two scores actually disagree. Position P has distribution with top-1 `0.46`, top-2 `0.44` (the rest `0.10`): top-token probability `0.46`, margin `0.02`. Position Q has top-1 `0.42` and a long flat tail of ten tokens at `0.058` each: top-token probability `0.42`, margin `0.42 − 0.058 = 0.36`. Top-token probability ranks `P (0.46) > Q (0.42)`, so it commits P first. Margin ranks `Q (0.36) > P (0.02)`, so it commits Q first. They genuinely disagree, and on this example margin is making the better call: committing P means writing a token the model split almost 50/50 against its runner-up, which is the near-coin-flip case the margin intuition is built to catch. So margin is not a strawman.

But here is the thing the disagreement also exposes: my whole reason for wanting confidence was the irreversible-commit risk, and the quantity that names that risk is the model-assigned probability of being wrong, `1 − c^i`. For P that is `0.54`; for Q it is `0.58`. By *that* yardstick — probability this specific commitment is a mistake — P and Q are nearly tied and P is marginally safer, which is exactly what the top-token-probability ranking says. Margin answers a different question (is the decision unambiguous?) and is a second-difference statistic, more sensitive to the runner-up tail and not directly tied to "probability this commit is wrong." So both scores are defensible; margin even wins the cherry-picked near-tie. I will take the selected-token probability as the primary signal because it is the direct estimate of the thing my two diagnoses care about, and keep margin in mind as a sensible, swappable alternative rather than a clear improvement. At temperature zero this primary signal is exactly `p_θ(argmax)`.

I should double-check this confidence-keeping idea against a place it has already been made to work, so I am not reinventing a broken wheel. The exact move — run a masked parallel predictor, score each masked location by the predicted probability of its sampled token, keep only the most confident, remask and re-predict the rest over `T` iterations — is the iterative-decoding recipe for masked image transformers (Chang et al. 2022): Predict all masked locations in parallel; Sample a token at each and record its predicted probability as a confidence (unmasked positions get confidence `1.0`); compute how many to keep masked from a schedule; keep the most confident, remask the rest. "The model predicts all tokens simultaneously but only keeps the most confident ones." So the structure transfers. But two of their choices are tuned for *image diversity* and I should not import them blindly. They sample the token stochastically with temperature annealing — because in image synthesis they want diverse samples; I want the mode for deterministic tasks, so I take argmax (temperature toward zero) by default. And they use a *cosine* schedule for how many to keep masked — concave, "few confident predictions early, many late." My schedule is dictated by my generative model, not chosen for aesthetics: I checked above that my linear forward masking makes the principled per-step budget the *uniform* one (equal expected count per step), and that is what the helper hands me. So I take their confidence-keeping skeleton but with a likelihood-consistent linear/uniform schedule and a greedy token, which is what my setting demands.

Now the regime question: do I decode the whole response as one parallel block, or carve it into pieces? Pure parallel decoding fills positions anywhere in the response, ordered only by confidence — fully bidirectional, maximally parallel, and a good fit for open-ended continuation where there is no strong left-to-right backbone. But for long structured outputs — a multi-step derivation, a function body — a coarse left-to-right ordering helps: you want the early reasoning settled before the later steps that depend on it. So I allow a block size: split the generation region into consecutive blocks and process them strictly left to right, running the full confidence-keeping diffusion *within* each block before moving on. When the block length equals the generation length there is a single block and this is exactly fully-parallel decoding; when it is smaller it is semi-autoregressive — blocks in order, diffusion inside each. The constraint `gen_length % block_length == 0` keeps the blocks even, and I divide the total step budget evenly across blocks. One thing I must guard: when scoring confidences I have to forbid committing into positions of *future* blocks — set their confidence to `-inf` so the top-`k` never selects them — otherwise the parallel predictor, which sees the whole sequence, would happily fill ahead of the current block and break the left-to-right discipline. Inside the current block, carry-over already protects already-committed tokens (their confidence is effectively `-inf` for selection too, since I only ever score and select *masked* positions).

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

Let me hand-trace this slot once on the toy block to be sure it does what I argued. Step one, all four masked, `k = 2`. `conf = (0.90, 0.40, 0.34, 0.91)`, all in-block so none zeroed to `-inf`; `topk(conf, 2)` returns positions `{3, 0}`; `x` becomes `(a, M, M, c)` with positions 1,2 still masked. Step two re-runs the model on `(a, M, c)` context — positions 1 and 2 now scored against a sharper context, the remaining `k` from the budget commits them. That is exactly the easy-first order I wanted, and it is the precise opposite of the `{1,2}`-first draw random remasking could have made. The slot does the right thing.

So the whole strategy is: selected token, confidence = its model probability, commit the top-`k` most confident masked positions in the active block, repeat. Everything else — the mask layout, the block loop, the uniform budget, the one-forward-pass cost — was already given.

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
