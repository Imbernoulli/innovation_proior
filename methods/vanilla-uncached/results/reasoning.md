Let me start from the thing that actually bothers me, which is not a modeling trick but a principle. What a language model is supposed to do is capture an unknown distribution `p_data` over token sequences by fitting `p_θ`, and "fit" here means maximize `E_{p_data}[log p_θ(x)]`, equivalently minimize `KL(p_data ‖ p_θ)`. That is the whole game. Now the entire field has collapsed "do this well" into "do this with the left-to-right factorization" — `p_θ(x) = p_θ(x^1) Π_{i≥2} p_θ(x^i | x^{<i})` — until the two feel like the same statement. They are not. The factorization is one way to write a valid joint distribution; it is not the only way, and it carries baggage. The cleanest evidence that the baggage is real: a model that has learned "A is B" left-to-right fails on "B is ?". It never had a gradient that pushed it to predict A from B, because the order it was trained in only ever conditioned A→B. That asymmetry is not a property of language; it is a property of the factorization. So the question I actually want to answer is: can I keep the principle — maximum likelihood with an expressive transformer — and drop the commitment to a single generation order? If I can, scalability and in-context learning ought to follow from the principle (they are properties of fitting a rich conditional generator with a transformer, not of the arrow pointing rightward), and the reversal asymmetry ought to disappear on its own.

So I need three things that have to be mutually consistent: a *process* that builds a sequence out of nothing without picking an order up front, a *training loss* that is provably a bound on `-log p_θ` of that process (not a heuristic reconstruction loss — if it isn't tied to the likelihood, I've lost the principle and I'm just doing BERT again), and a *decoding rule* that turns the trained network into samples. Let me build the process first, because the loss and the decoder both fall out of it.

The diffusion idea is the natural template: define a forward corruption from clean data `x_0` toward pure noise, then learn to reverse it. For continuous data that's `z_t = √(α_t) x_0 + √(1-α_t) ε` and you train against a variational lower bound on the log-likelihood. But tokens are categorical; "add a little Gaussian noise" is meaningless. The discrete version is a Markov chain directly on the categorical state, `q(x_t | x_{t-1}) = Cat(x_t; Q_t x_{t-1})`, marginals `q(x_t|x_0) = Cat(x_t; Q̄_t x_0)`. Now I get to choose `Q_t`, and the choice decides everything. A uniform `Q_t` that lets any token flip to any other token is symmetric but it scrambles meaning — a corrupted token looks like a legitimate token, so the model can't even tell what was destroyed. What I want is a corruption where destruction is *recognizable*: a token is either still itself or visibly gone. That is the absorbing-state construction — a special `[MASK]` symbol that each token either stays as itself or jumps into, and once it's `[MASK]` it stays `[MASK]`. The stationary distribution then puts all its mass on the all-`[MASK]` sequence, which is exactly the "pure noise" endpoint I need to start sampling from. And empirically, on text, masking beats uniform and similarity-structured discrete diffusion — which makes sense, because the model's job becomes the clean, well-posed "fill in the holes" rather than "guess which tokens were silently swapped."

Let me write the forward process concretely and make it as simple as it can be. I'll index time continuously, `t ∈ [0,1]`, factorize across positions, and put

  `q_{t|0}(x_t | x_0) = Π_i q_{t|0}(x_t^i | x_0^i)`, with `q_{t|0}(x_t^i | x_0^i) = 1-t` if `x_t^i = x_0^i`, and `= t` if `x_t^i = M`.

So each token is independently masked with probability `t` and survives with probability `1-t`. At `t=0` nothing is masked, at `t=1` everything is, and the expected mask fraction rises *linearly* with `t`. Why linear and not some fancy schedule? Because on average the information content of text is roughly proportional to the number of tokens, so losing information at a constant rate in `t` is the honest default — there's no privileged corruption level that deserves a slower or faster schedule a priori. I'll keep it linear and see whether the loss it induces is clean; if linearity buys me algebraic simplicity downstream, that's confirmation rather than coincidence.

Now the reverse process, from `t` down to `s < t`. Because the forward process is independent across positions, the reverse posterior factorizes across positions too. For one token, condition on `x_t` and ask for `x_s`. If the token was already revealed at `t` (`x_t^i ≠ M`), it must stay revealed and unchanged — that's a hard constraint of the absorbing process, you can't un-reveal. If it's masked at `t`, then going to the *less*-noised `s` it either stays masked or gets revealed to its true value. The exact posterior that is consistent with the forward marginals is

  `q_{s|t}(x_s^i | x_t) = 1` if `x_t^i ≠ M, x_s^i = x_t^i`; `= s/t` if `x_t^i = M, x_s^i = M`; `= (t-s)/t · q_{0|t}(x_s^i | x_t)` if `x_t^i = M, x_s^i ≠ M`; and `0` otherwise.

Let me sanity-check the `s/t`. A token masked at `t` was masked independently with prob `t`; at the earlier time `s` it would have been masked with prob `s`. Given it's masked at `t`, the chance it's *still* masked at `s` is `s/t` (and `1 - s/t = (t-s)/t` is the chance it gets revealed in this step), which is exactly the ratio of the two marginal masking probabilities — the only choice that keeps the reverse chain's marginals equal to the forward marginals. Good, the fractions are forced, not tuned. The one piece I don't get for free is `q_{0|t}(x_s^i | x_t)` — *which* token a masked position reveals to. That's the data-prediction function, the discrete analogue of "predict the clean signal," and it's the only thing a network has to learn.

Here is a subtlety I want to nail before I parameterize the network, because it changes the architecture. What does `q_{0|t}(· | x_t)` actually depend on? A masked position's clean value depends on the *unmasked* tokens around it — those are literally copies of `x_0` (an unmasked token at time `t` is unchanged from `x_0`, that's the absorbing property), so they carry no time information; they're just clean context. And the *masked* positions carry no information about the answer at all, only their count and locations, which the unmasked context already implies. So `q_{0|t}(x_0^i | x_t)` should equal `p_data(x_0^i | x_t^{UM})` where `x_t^{UM}` is the collection of unmasked tokens — a conditional on a clean *subset*, which does not depend on `t` at all. That's a real simplification: the network does not need `t` as an input. Predicting a masked token from revealed context is the same problem whether 10% or 90% of the rest is masked; the corruption level is already visible in the input. So my predictor is a plain function `p_θ(· | x_t)` — feed in the partially-masked sequence, read out a distribution over clean tokens at every position, no timestep embedding. And since it must condition on context that can lie on *either side* of a masked position, it has to be bidirectional — no causal mask. That single fact is what kills the reversal asymmetry: a bidirectional masked predictor, by construction, learns to predict any position from any subset of the others.

Now the loss. I have a generative process `p_θ(x_0)` defined by running the reverse chain from all-masked with the learned `p_θ(· | x_t)` plugged in for `q_{0|t}`. The principled objective is the variational bound on `-log p_θ(x_0)`, which for a discrete diffusion is the D3PM-style sum of per-step KLs, `E_q[ D_KL(q(x_T|x_0)‖p(x_T)) + Σ_t D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t)) − log p_θ(x_0|x_1) ]`. If I take that literally I'm in D3PM's world: materialize the full transition matrices and posterior distributions, compare them termwise — expensive and opaque, a dense KL with no obvious reduction. Let me not take it literally. Let me use the structure of the masking process to collapse it.

Two facts about the denoiser, both forced by the absorbing process. First, the model should never put mass on `[MASK]` as an *output* — `[MASK]` is a corruption symbol, not a real token, so I'll fix `⟨p_θ, m⟩ = 0` (set the `[MASK]` logit to `-∞`). Second, an already-unmasked token is copied straight through — carry-over. Now look at a single reverse step's KL term, `D_KL(q(x_s|x_t,x_0) ‖ p_θ(x_s|x_t))`. For an unmasked position both sides are the delta on the carried-over token, KL `= 0`. For a position that stays masked both sides are the delta on `M`, KL `= 0`. The *only* positions that contribute are the masked-at-`t` positions, and for those the KL between the true posterior `(t-s)/t · q_{0|t}` and the model posterior `(t-s)/t · p_θ` reduces — after the `[MASK]`-zeroing and carry-over kill the cross terms — to a plain cross-entropy between the true clean token and `p_θ`, scaled by the step's reveal mass. In survival-probability notation, `α_t = 1-t`, so a reverse step from `t` down to `s < t` has weight `(α_s − α_t)/(1 − α_t) = (t-s)/t` on the clean-token cross-entropy. The dense KL has become a *weighted sum of masked-token cross-entropies*. That's the whole point of specializing to masking: the bound is the same likelihood bound, but it's now something I can actually compute by looking only at the masked indices.

Push the number of steps to infinity to get the cleanest continuous-time form. The per-step cross-entropy weight `(α_s − α_t)/(1−α_t)` over an infinitesimal step becomes `[-α'_t/(1−α_t)] dt`, and the negative-ELBO is

  `L∞ = E_q ∫_{t=0}^{1} [ -α'_t/(1−α_t) ] · [ -log⟨p_θ(x_t), x_0⟩ ] dt`.

Now substitute my linear schedule, `α_t = 1−t`, so `α'_t = −1` and `1−α_t = t`. The cross-entropy weight becomes exactly `1/t`, i.e. summing over the masked positions,

  `L(θ) = −E_{t, x_0, x_t} [ (1/t) Σ_i 1[x_t^i = M] · log p_θ(x_0^i | x_t) ]`,

and this is an upper bound on `−E_{p_data}[log p_θ(x_0)]`. *There* is where the famous `1/t` comes from — and it is not a hand-tuned reweighting, it is exactly the negative-ELBO weight of a masked-diffusion process with a linear schedule. This is the line I most wanted to land cleanly, because it's the difference between a principle and a heuristic. The linear schedule I chose "because information is lost linearly" is what makes the weight a clean `1/t` rather than some schedule-dependent mess; the two choices reinforce each other.

Let me make sure I see *why* the `1/t` has to be there by thinking about what it's correcting. Sample `t ~ U[0,1]` and mask each token with prob `t`. At small `t` only a handful of tokens are masked, so the inner sum `Σ_i 1[masked] log p_θ` has few terms; at large `t` almost everything is masked and the sum is huge. Without the `1/t`, the large-`t`, heavily-masked states would dominate the expected loss purely by sheer term count, and the model would be trained mostly on the near-impossible "reconstruct from almost nothing" regime and barely on the informative low-noise regime. The `1/t` exactly rescales each corruption level so every level contributes comparably to the gradient — it is the bookkeeping that makes "uniformly over all noise levels" mean what it says. Now I can also say precisely why BERT, which this superficially resembles, is *not* this: BERT fixes one mask ratio (~15%), so it only ever trains the single corruption level `t ≈ 0.15`, and it has no process linking all-masked to clean — it's a representation learner, not a generative model, and its loss is no bound on any data likelihood. And MaskGIT, which trains a bidirectional predictor with a *varying* mask ratio and even decodes iteratively, writes its loss as a flat `−E[Σ_{masked} log p]` with no `1/t` weight — so it's a sound reconstruction objective but it is not derived as a likelihood bound, it's missing exactly this term. The `1/t` is the whole bridge from "fill in masks" to "maximum likelihood generative model," and it's why I can claim the principle, not just the trick.

Let me double-check the claim that this is order-agnostic in a way that explains the reversal fix, because that was the motivation. Consider the any-order autoregressive objective, `−E_{x_0, π∼U_π}[ Σ_i log p_θ(x_0^{π(i)} | x_0^{π(<i)}) ]`: pick a random order, reveal tokens one at a time, predict the next under that order. But "predict `x^{π(i)}` given the already-revealed `π(<i)`" *is* "predict a masked token given an unmasked subset" — `x_0^{π(<i)}` is just an `x_t` whose masked set is `π(≥i)`. Averaging over all orders is averaging over all masked subsets, which is what my expectation over `t` and the random mask already does. So my objective is, up to the weighting that turns a sum-over-orders into the diffusion expectation, the AO-ARM objective — and AO-ARM trains *every* conditional direction, including B-given-A and A-given-B. The reversal curse was never about language; it was about training only one order. Train all orders and it's gone, with nothing special added for reversal.

I'll also note a practical equivalent of the loss that I'll want for *scoring* sequences (likelihood evaluation), because the sampling form above has a nuisance: when I sample `t` and then mask each token independently with prob `t`, the *actual* number of masked tokens fluctuates around `tL`, and for short sequences that fluctuation is large, which makes the Monte-Carlo estimate of `L` noisy. The fix is to mask *exactly* `l` tokens chosen uniformly without replacement, with `l ~ U{1,…,L}`, which has the same expectation but a deterministic mask count `l/L` per draw. The reweight changes from `1/t` to `L/l` (same thing with `t = l/L`), giving `−E_{l, x_0, x_l}[ (L/l) Σ_i 1[x_l^i = M] log p_θ(x_0^i | x_l) ]`. Empirically the independent-mask form needs on the order of a thousand Monte-Carlo draws to stabilize while the exact-`l` form needs around a hundred — pure variance reduction, same estimand. For training I'll use the independent-mask form (simpler to batch), for likelihood evaluation the exact-`l` form.

Training is now fully specified and embarrassingly close to ordinary LM training: draw `x_0`, draw `t ~ U(0,1]`, mask each token with prob `t` to get `x_t`, forward the bidirectional transformer once, take cross-entropy on the masked positions, scale by `1/(tL)`, backprop. No causal mask, no timestep input. For supervised fine-tuning I want `p_θ(r_0 | p_0)` instead of `p_θ(x_0)`: leave the prompt `p_0` completely unmasked, mask only the response `r_0`, and run the identical loss over the response positions, `−E[(1/(tL')) Σ_i 1[r_t^i = M] log p_θ(r_0^i | p_0, r_t)]`. This is literally pre-training with the masked set confined to the response, so concatenating `p_0` with clean `r_0` is a clean `x_0` and `p_0` with masked `r_t` is its `x_t` — no new machinery.

Now the part the whole construction exists to serve: turning the trained predictor into samples. The reverse process tells me exactly what to do. Start with the response region fully masked, `r_1 = [M…M]`. Discretize `[0,1]` into `N` steps, going `t` down to `s = t − 1/N`. At each step, feed `(p_0, r_t)` through the predictor, get a distribution over clean tokens at every masked position, and predict each masked token (I'll take the argmax for a deterministic, "greedy" reveal). Then I have to put back the right amount of mask to land on `r_s` instead of `r_0`: the reverse posterior says a token that's masked at `t` should *stay* masked with probability `s/t`. So if I just predicted all masked positions, I should re-mask a fraction `s/t` of them to respect the schedule. The simplest faithful rule: among the freshly-predicted tokens, send a `s/t` fraction back to `[MASK]`, keep the rest as committed.

The question is *which* `s/t` fraction to re-mask. The exact reverse process says: purely at random. Let me actually picture random re-masking running. I predict all the masked slots, many of them confidently right, a few of them garbage because the surrounding context was too sparse to pin them down. Random re-masking will, with equal probability, throw away a confidently-correct token and keep a garbage one. That's wasteful — I'm discarding good information and committing to bad guesses, and committed tokens are frozen for the rest of the rollout (carry-over). I want to keep what I'm sure of and re-mask what I'm unsure of, so the hard positions get more attempts with progressively more context filled in. This is exactly the iterative-decoding move from confidence-based parallel decoding of masked predictors: predict everything, *keep the most confident*, re-mask the least confident, repeat with the mask ratio annealing down. So instead of a random `s/t` fraction, I re-mask the *lowest-confidence* `s/t` fraction, where the confidence of a predicted token is the probability the model assigned it, `c^i = p_θ(r_0^i | p_0, r_t)_{r_0^i}`. Already-unmasked tokens get confidence `1` (they're committed). Concretely, at the step landing on `s` I want `n_un = ⌊L(1−s)⌋` tokens unmasked total, so I keep the `n_un` highest-confidence positions and re-mask the rest.

I should be honest that this *departs* from the exact reverse process — the exact one re-masks at random, and low-confidence re-masking is a deterministic, confidence-greedy approximation, not the true posterior. So why is it the right call? Two reasons converge. One, it's the discrete analogue of the annealed / low-temperature sampling everyone already uses with autoregressive LLMs: committing to high-probability tokens first reduces the diversity of the output but sharpens accuracy, and on factual/coded/math tasks accuracy is what I want. Two, it builds a coarse-to-fine schedule for free — the easy, high-confidence tokens lock in early and become reliable context that makes the hard tokens easier on later steps, whereas a hard token I'm unsure about stays revisable until enough of its neighbourhood is settled. It is documented that this confidence-based re-masking substantially beats random re-masking on downstream accuracy — so I'll adopt it as the default and keep random re-masking only as the faithful-but-weaker reference. The cost is that the per-position confidences across the sequence have to be commensurable, which they are: they're all `p_θ`-probabilities from the same softmax.

One more thing about how many tokens to reveal per step, because the schedule and the linear noise process have to agree. Over `N` uniform steps with a linear schedule, the expected number of tokens that transition from masked to revealed should be *the same at every step* — linearity means a constant unmask rate. So if the block has `M` masked tokens and `N` steps, I reveal `M/N` per step. `M/N` won't be an integer in general, so I take the floor `⌊M/N⌋` as the base count and distribute the remainder `M mod N` one-each across the earliest steps; that keeps the per-step counts as equal as integers allow and guarantees all `M` tokens are revealed in exactly `N` steps. Under the usual setting `N ≤ M`, every step commits at least one token; with `N` equal to the generation length, that's one token per step — the finest, most accurate schedule.

Now, do I have to denoise the *entire* response at once? The pure-diffusion sampler does, and it's the most flexible — every position is revisable at every step. But there's a useful intermediate. I can partition the response into contiguous blocks and run the diffusion reverse process within a block while moving left-to-right *across* blocks — semi-autoregressive remasking. Inside a block it's full diffusion (predict all, keep-confident, re-mask the rest); across blocks it's ordered, so an earlier block is fully resolved before a later one starts. The knob is the block length: block length equal to the whole generation is pure diffusion; block length one is autoregressive; in between trades the global revisability of pure diffusion against the locality of AR decoding. I'll keep block length as a hyperparameter, with the plain control denoising one block at a time but, within the block, recomputing the full sequence at every step.

That last clause is where I have to be careful about a tempting shortcut, and the plain control deliberately *declines* it. Notice the absorbing property again: once a token is revealed it never changes. So across the whole rollout there are at most `L` distinct "states" of the input that the predictor ever sees — most steps only flip a few masked positions to revealed, leaving the rest of the input identical to the previous step. And since my predictor has no time input, its output on an unchanged input is identical. In principle, then, I could cache predictor work for parts of the state that truly have the same conditioning information. That's a real saving and it's mathematically licensed only when the conditioning state has not changed. But it's also where correctness gets subtle: the moment any masked token reveals, it changes the bidirectional context for *every other* position — there's no causal prefix, so an early position's correct prediction genuinely depends on tokens that get revealed later, and a stale cached value can be wrong. The conservative, exactly-correct thing is to run a full bidirectional forward over the entire current sequence at every single denoising step, recomputing everything, reusing nothing. No prefix KV cache (there's no causal prefix to cache, and the keys/values of "committed" tokens keep being re-attended to as new tokens reveal), no feature cache, no skipped positions. It's the plain reference rollout.

So let me assemble the decoder as the code I'd actually ship — the plain, uncached LLaDA generation procedure. Lay the response region as `gen_length` mask tokens after the prompt. Split the generation into blocks; per block, the steps are `N/num_blocks`. Per step: one full forward over the whole sequence, optionally with classifier-free guidance (run a second forward with the prompt also masked, duplicate the attention mask for the two branches, and push the logits along `un_logits + (cfg_scale + 1)(logits − un_logits)`); take the argmax prediction at every position (with optional Gumbel noise if sampling at nonzero temperature, done in float64 because low-precision Gumbel-max hurts generation quality); compute the confidence as the softmax probability of the chosen token for low-confidence re-masking, or a uniform random number for the random-re-masking reference; restrict the candidates to the *current* block by setting confidences past the block to `−∞`; carry over already-revealed tokens; then top-k the confidences with `k` = the precomputed transfer count for this step, commit those positions, and leave the rest masked for the next step. I'll also keep the EOS/EoT switches exposed by the generation code because heavy EOS padding in SFT can make low-confidence re-masking terminate responses too early.

```python
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    # Gumbel-max sampling of the categorical; float64 because low-precision
    # Gumbel-max degrades generation quality for masked diffusion.
    if temperature == 0:
        return logits                                  # temperature 0 -> plain argmax (greedy reveal)
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    # Linear schedule => equal expected number of tokens revealed per step.
    # Distribute floor(M/steps) per step and spread the remainder over the early steps.
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128,
             temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336,
             logits_eos_inf=False, confidence_eos_eot_inf=False):
    # Response region starts fully masked; prompt stays clean (carry-over keeps it fixed).
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype,
                        device=model.device)], dim=-1)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks                          # steps within each block

    for num_block in range(num_blocks):                  # semi-autoregressive across blocks
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length:
                 prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):                           # diffusion reverse process within the block
            mask_index = (x == mask_id)
            # --- the uncached control: a full bidirectional forward over the WHOLE sequence,
            #     recomputing everything, reusing nothing, every single step ---
            if cfg_scale > 0.:                           # optional unsupervised classifier-free guidance
                un_x = x.clone()
                un_x[prompt_index] = mask_id             # the prompt-masked "unconditional" branch
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = None
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf        # suppress early EOS for some tasks

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # predicted clean token at every position

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(                     # confidence = prob assigned to the chosen token
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)  # faithful-but-weaker reference
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf  # only the current block

            x0 = torch.where(mask_index, x0, x)          # carry over already-revealed tokens
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):         # keep the most-confident k; re-mask the rest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]        # commit; the rest stay masked for the next step

    return x
```

Let me trace the causal chain back to make sure nothing was bolted on. I wanted to keep the maximum-likelihood principle while dropping the left-to-right commitment, because that commitment is what produces the reversal asymmetry and forbids global revision. A diffusion process gives an order-free way to build a sequence; on tokens, the absorbing-`[MASK]` corruption is the choice that makes destruction recognizable and gives an all-masked noise endpoint, and a linear masking schedule is the honest default for text. The reverse process factorizes across positions, its `s/t` re-mask fractions are forced by matching the forward marginals, and the only learnable piece, the clean-token predictor `q_{0|t}`, turns out to depend only on the unmasked context — time-independent — so the network is a plain bidirectional transformer with no timestep input, which is exactly what makes it order-agnostic and kills the reversal curse. Taking the discrete-diffusion variational bound and collapsing it with the two structural facts (never predict `[MASK]`, carry over revealed tokens) reduces a dense per-step KL to a weighted sum of masked-token cross-entropies, and the continuous-time limit under the linear schedule produces the `1/t`-weighted cross-entropy as a genuine upper bound on `−log p_θ` — the `1/t` being the negative-ELBO weight that balances every corruption level, the precise term that separates this from BERT's fixed-ratio MLM and from MaskGIT's unweighted reconstruction. SFT is the same loss with the mask confined to the response. Sampling discretizes the reverse process from all-masked; re-masking the lowest-confidence `s/t` fraction instead of a random one is an annealed coarse-to-fine approximation that trades diversity for accuracy and is known to beat random re-masking; the per-step reveal count `⌊M/N⌋`-plus-remainder follows from the constant unmask rate of the linear schedule; block partitioning interpolates between pure diffusion and autoregression. And the plain control keeps the rollout exact by running a full bidirectional forward over the entire sequence at every step, reusing nothing.
