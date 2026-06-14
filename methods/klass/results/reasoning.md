Let me start from what actually hurts when I generate from one of these masked diffusion language models. The model fills a sequence not left-to-right but by unmasking: I set the whole generation region to the mask token, and over some fixed number of denoising steps the network looks at the entire partially-filled sequence and, for every still-masked position, hands me a categorical distribution over the vocabulary. So at each step I get to decide three things — how many positions to commit, which ones, and what token to write into each. The token to write is the easy part: I take the model's own prediction, the argmax (or a sample) of its distribution there, and because the carry-over parameterization fixes a position once it's revealed, whatever I commit stays committed for the rest of the generation. That last fact is the whole tension. Commit is permanent. If I commit one position per step I get good quality but I pay one full network evaluation per generated token — that's just autoregressive speed, and it throws away the one thing a diffusion model is supposed to give me, the ability to predict many positions at once. If I commit many positions per step on a fixed schedule I go fast, but I'm committing tokens the model isn't actually sure about, and since commit is permanent I'm locking in errors that the still-masked context would have fixed if I'd waited. The number of steps is my budget and the number of forward passes is my cost, so I want to cut steps; but every step I cut by committing more is a step where I might commit garbage. That's the trap. I want to commit *many* tokens per step exactly when it's safe and otherwise fall back to the ordinary schedule-sized confidence choice.

What do people do today? The oldest move, from the image side, is MaskGIT: predict every masked position at once, keep the most confident predictions — confidence being just the top probability, \(\mathrm{conf}^i = \max_v p^i(v)\) — remask the rest, and repeat under a decreasing schedule so fewer positions stay masked each round. LLaDA, the big language MDM, runs essentially the same thing under the name low-confidence remasking: predict all, remask the lowest-confidence ones, which is the same as unmasking the top-\(k\) by max probability, with \(k\) set by a fixed uniform schedule, \(\lfloor(\#\text{masks})/\text{steps}\rfloor\) per step. Dream sharpens the signal a little — instead of max probability it ranks by the *margin*, top-1 minus top-2, \(p^i_{(1)}-p^i_{(2)}\), which correctly punishes a position where two tokens are nearly tied (max probability can't see that the runner-up is breathing down its neck). And Fast-dLLM makes the count adaptive instead of fixed: drop the schedule, unmask *every* position whose confidence clears an absolute threshold \(\tau\), and if none clears it, unmask the single most-confident one so the chain never stalls. Fast-dLLM even gives me a reason to believe parallel commits can be safe: if for some target sequence the model's per-position marginals are all confident enough — \(p_\theta(x^*_{i_j}\mid E) > 1-\epsilon\) for every \(j\), with \((n+1)\epsilon \le 1\) — then greedily decoding from the product of marginals picks the same sequence as greedily decoding from the true joint, the bound tight at \(\epsilon = 1/(n+1)\). So in the high-confidence regime, committing many positions at once is provably the same as committing them one at a time. Good. That's a real license to parallelize.

But every one of these keys off a single number read at a single step: max probability, or the margin, or whether the probability clears \(\tau\). And here's the thing that's been nagging me — confidence at one step is not the same as being *right*. I keep seeing the same failure: a sampler commits a token it's highly confident about, and it's wrong; the correct token at that position had a lower confidence at that step, so the confidence-only rule passed it over. Fast-dLLM's theorem doesn't save me here, because its premise is "the model's marginal is correct enough," and the whole problem is that a confident marginal can be confidently wrong while the context that would overturn it is still masked. So confidence is necessary — I'm not going to commit a position the model is unsure of — but it is plainly not sufficient. There's a second axis I'm not using, and I don't yet know what it is. Let me stare at the failures instead of the formula.

When I take a position where a confidence-only sampler commits a high-confidence wrong token, and I watch that position *over the denoising steps* rather than at one frozen step, something jumps out. Its distribution is still moving. From one step to the next the model's categorical at that position keeps shifting — the argmax may even stay the same and look confident, but the shape underneath is churning, because the surrounding context is still being resolved and each newly revealed neighbor nudges the model's belief here. Now take a position whose commitment no longer looks fragile: its distribution has stopped moving. The model decided what goes there several steps ago and every subsequent step leaves it essentially unchanged. So the discriminating thing isn't *how tall* the peak is at this instant — it's whether the model's belief at this position has *settled*. Confidence is a snapshot; what I actually want is a measure of *temporal consistency*. And I can quantify "how much did the model's belief at position \(i\) move between the previous step and this one" directly from the distributions the sampler already has: it's a divergence between two categoricals, the one I had at the last step and the one I have now. KL divergence is the natural choice — \(d_t^i = D_{\mathrm{KL}}(p_t^i \,\|\, p_{t+1}^i)\), comparing the current step's distribution against the previous step's at the same position. Near zero means the model isn't changing its mind there; large means it still is. That gives me a KL score to put beside the confidence score.

I want to be careful that I'm not just curve-fitting an observation, so let me see whether I can argue from first principles that a wrong-but-should-be-correctable token *must* be unstable — that high KL is forced on it, not incidental. Set up the cleanest version. Fix a position \(i\). Suppose the model is a decent approximation of the task: there's a family of task-correct conditionals, and for every context \(c\), the model's distribution at \(i\) given \(c\) is within total variation \(\delta\) of a correct one. Now suppose at \(i\) there's a genuinely correct token \(x^\star\) that, at the resolved near-optimal context \(c^\star\), beats a particular suboptimal token \(x^\dagger\) by a margin \(\gamma\): \(\pi(x^\star\mid c^\star) \ge \pi(x^\dagger\mid c^\star) + \gamma\). And suppose that *right now*, at the current under-resolved context \(c_M\) — where most of the neighbors are still masked — the model actually prefers the wrong token, \(p_\theta(x^\dagger\mid c_M) \ge p_\theta(x^\star\mid c_M) + \beta\) for some \(\beta \ge 0\). This is exactly the "confident but wrong" situation: the model leans \(x^\dagger\) now but the truth is \(x^\star\). The denoising process is going to walk the context from \(c_M\) down to \(c^\star\) as the other positions get filled in, \(c_M \to c_{M-1} \to \dots \to c_0 = c^\star\), changing only variables outside \(i\). Write \(P_t = p_\theta(\cdot \mid c_t)\). The question is: how much must the distribution at \(i\) move along this path?

Pick a test function that reads off exactly the \(x^\dagger\)-vs-\(x^\star\) preference: \(f = \mathbf 1\{x_i = x^\dagger\} - \mathbf 1\{x_i = x^\star\}\), so \(\|f\|_\infty \le 1\). Total variation controls how much any bounded function's expectation can differ between two distributions. Let \(A_M = p_\theta(x^\dagger\mid c_M) - p_\theta(x^\star\mid c_M)\) and \(A_0 = p_\theta(x^\dagger\mid c^\star) - p_\theta(x^\star\mid c^\star)\). Then \(2\,\mathrm{TV}(P_M, P_0) \ge |\mathbb E_{P_M}[f] - \mathbb E_{P_0}[f]| = |A_M - A_0|\). The current-preference assumption gives \(A_M \ge \beta\). At \(c^\star\), the \(\delta\)-approximation gives \(p_\theta(x^\star\mid c^\star) \ge \pi(x^\star\mid c^\star) - \delta\) and \(p_\theta(x^\dagger\mid c^\star) \le \pi(x^\dagger\mid c^\star) + \delta\), so \(p_\theta(x^\star\mid c^\star) - p_\theta(x^\dagger\mid c^\star) \ge (\pi(x^\star\mid c^\star) - \pi(x^\dagger\mid c^\star)) - 2\delta \ge \gamma - 2\delta\). Equivalently, \(A_0 \le -(\gamma - 2\delta)\). Therefore \(A_M - A_0 \ge \beta + \gamma - 2\delta\). If that quantity is negative, the argument simply gives no positive endpoint separation, so I take the positive part: \(2\,\mathrm{TV}(P_M, P_0) \ge (\beta + \gamma - 2\delta)_+\), i.e. \(\mathrm{TV}(P_M, P_0) \ge \tfrac12(\beta + \gamma - 2\delta)_+ =: \Delta\). The endpoints of the path are far apart in total variation — at least \(\Delta\) — whenever the true margin and the current wrong-preference together beat twice the model's approximation error.

Now I have endpoint separation; I need to turn it into a statement about *per-step* movement, because per-step movement is what my KL score measures. Total variation obeys the triangle inequality along the path, so the per-step TVs must sum to at least the endpoint TV: \(\sum_{t=0}^{M-1} \mathrm{TV}(P_{t+1}, P_t) \ge \mathrm{TV}(P_M, P_0) \ge \Delta\). Let \(T_t = \mathrm{TV}(P_{t+1}, P_t)\). To connect TV to KL I reach for Pinsker, \(D_{\mathrm{KL}}(P_t \,\|\, P_{t+1}) \ge 2\,T_t^2\) (TV is symmetric in its two arguments, so it doesn't matter which order I wrote it). Then the average per-step KL is \(\frac1M \sum_t D_{\mathrm{KL}}(P_t\|P_{t+1}) \ge \frac2M \sum_t T_t^2\), and by Cauchy–Schwarz \(\big(\sum_t T_t\big)^2 \le M \sum_t T_t^2\), so \(\sum_t T_t^2 \ge \frac1M\big(\sum_t T_t\big)^2 \ge \frac{\Delta^2}{M}\), which gives \(\frac1M \sum_t D_{\mathrm{KL}}(P_t\|P_{t+1}) \ge \frac{2\Delta^2}{M^2}\). There it is. A token that is wrong at the under-resolved context but ought to be correct at the resolved one cannot keep its average per-step KL near zero — it is *forced* to be dynamically unstable somewhere along the denoising path, with average KL bounded below by \(2\Delta^2/M^2\). So if a position's per-step KL has stayed near zero across the recent steps, that is exactly the evidence I want before treating it as unlikely to be one of these will-flip tokens. The stability gap is not a coincidence; instability is the mathematical fingerprint of a token that's going to change its mind.

So now I know what to gate on. I will commit a masked position only if it is *stable* — its recent step-to-step KL is low — *and* confident — its top probability is high. Both, not either. Confidence alone is the prior art and admits the confident-but-wrong tokens; KL alone would let me commit a position whose belief has frozen but at a low, mushy probability where the model isn't actually committing to anything; together they say "the model has settled on this and is sure of it." Let me write the stable set. With a confidence threshold \(\tau\) and a KL threshold \(\epsilon_{\mathrm{KL}}\), the simplest version is: position \(i\) is ready at step \(t\) if its current KL is below \(\epsilon_{\mathrm{KL}}\) and its confidence exceeds \(\tau\).

But one KL reading bothers me. A single low KL between two consecutive steps could be a fluke — the distribution at that position might happen not to move for one step and then lurch the next, especially early when lots of context is still masked. The theory bounds the *average* KL away from zero for an unstable token, which is consistent with it being momentarily low; what I want is evidence of *sustained* stillness, not a one-frame accident. So I'll demand that the last \(n\) consecutive-step KLs are *all* below threshold, not just the latest one. That turns "it didn't move this step" into "it hasn't moved for \(n\) steps," which is exactly the robustness the average-KL bound tells me I need — a will-flip token can sneak one low KL past me but not a run of them. The stable set becomes
\[
S_t = \Big\{\, i \;\Big|\; \big(\forall k \in \{1,\dots,n\}:\, D_{\mathrm{KL}}(p_{t+k-1}^i \,\|\, p_{t+k}^i) < \epsilon_{\mathrm{KL}}\big) \;\wedge\; \mathrm{conf}_t^i > \tau \,\Big\}.
\]
How big should \(n\) be? \(n=1\) is the trigger-happy version that the fluke argument just warned me against — it'll commit on a single coincidental match and unmask prematurely. Large \(n\) is safe but stingy: requiring a long run of low KLs means I rarely declare anything stable, so I commit few tokens per step and lose the speedup I came for. There's an obvious sweet spot at small-but-greater-than-one, and \(n=2\) is the natural first stop — demand stability across two steps, enough to kill the single-step fluke without paying for a long history. I'll keep \(n\) a knob but expect 2 to be the default. One consequence to handle: with a history window of length \(n\), I shouldn't evaluate the stable condition until the window has had time to fill, so for the first \(n-1\) steps of a block nothing is eligible to be called stable. That guard is deliberately conservative; any startup artifact that remains in the rolling buffer will also fail the all-below-threshold test until it rolls out.

Now the unmasking rule itself, and this is where the speedup actually comes from. At each step I commit *every* position in \(S_t\) — all of them, in parallel, however many there are. At a step where the block has settled, many positions cross both thresholds at once and get written together in a single forward pass; at a step where things are still churning, \(S_t\) is small or empty. The count adapts to the model's own state instead of being dictated by a schedule — that's the lever Fast-dLLM introduced, but now gated on stability-and-confidence rather than confidence alone. The remaining worry is the empty case: if \(S_t\) is empty, committing nothing means the chain makes no progress, and a too-strict pair of thresholds could stall it forever. So I need a fallback that guarantees progress, and it should degrade gracefully to a sane baseline rather than do something exotic. When \(S_t = \emptyset\), unmask the Top-\(u\) masked positions by confidence, where \(u\) is the fixed schedule count — exactly what a plain confidence sampler would have done at that step. That way the worst case for KLASS is "behaves like the confidence baseline," and the best case is "commits a whole settled block in one shot," and it interpolates between them automatically. This mirrors Fast-dLLM's always-unmask-the-top guarantee, generalized to the schedule-sized count so the fallback can't itself stall.
\[
x_t^i = \begin{cases} \text{commit position } i, & i \in S_t,\\[2pt] \text{else commit the Top-}u\text{ positions by } \mathrm{conf}_t^i, & S_t = \emptyset. \end{cases}
\]

Let me now think about the things that will bite me in the implementation, because a stability signal built on tiny differences between near-identical distributions is exactly where numerical sloppiness shows up. The KL is \(\sum_v p_t^i(v)\,(\log p_t^i(v) - \log p_{t+1}^i(v))\), and when two consecutive distributions are nearly equal — which is the regime I most care about, the "settled" regime — this is a small difference of large logs, so float32 will give me noise where I need to read a threshold of order \(10^{-2}\). And separately, low-precision softmax is already known to hurt generation quality in these models. So I'll do the softmax and the KL in float64. The log terms need a floor against \(\log 0\): I add a tiny \(\epsilon\) (say \(10^{-12}\)) inside both logs, \(\log(p + \epsilon)\), which costs nothing in the settled regime and just keeps the masked-out tail from producing infinities.

There's a subtlety at the very first step of a block: I have no previous distribution yet, so the "previous" buffer starts at zero. The KL against an all-zero "distribution" is meaningless — every term is \(p\,(\log p - \log \epsilon)\), a huge positive number. But this never actually produces a false stability verdict. At step 0 the explicit guard forces stability off. At step \(n-1\) the guard first permits the all-below-threshold test, but the first zero-origin KL is still in the rolling window and is huge, so the test fails. Only after that startup KL has rolled out can the window contain \(n\) genuine consecutive-step KLs between real model distributions. I'll maintain the KL history as a rolling buffer per position: roll it left by one and write the new KL into the last slot each step, so after warmup the window always holds the most recent \(n\) real values.

Let me also pin down which two distributions the KL compares, because I want the implementation to match the theory's "per-step along the path." At each step I form \(p_{\mathrm{curr}}\), compute its KL against the \(p_{\mathrm{prev}}\) I saved last step, push that KL into the history, and *then* overwrite \(p_{\mathrm{prev}} \leftarrow p_{\mathrm{curr}}\) for next time. Diffusion steps are counted downward from \(T\): after moving from timestep \(t+1\) to timestep \(t\), the distribution I just computed is \(p_t\), and the saved distribution from the previous wall-clock evaluation is \(p_{t+1}\). So \(D_{\mathrm{KL}}(p_{\mathrm{curr}} \| p_{\mathrm{prev}})\) is exactly \(D_{\mathrm{KL}}(p_t^i \| p_{t+1}^i)\), current step against previous step, the same orientation as the path argument. Good, no direction confusion.

And I should make sure this whole thing is genuinely lightweight — no training, no second model, no growing planner state. It is: every signal I use is the base model's *own* output, read across its own successive predictions. I'm not training a planner, not running an auxiliary scorer, not aligning two models. The KL history is \(O(L \cdot n)\) and the previous-distribution buffer is \(O(L \cdot V)\) — the latter is the only nontrivial cost, one extra copy of the per-position categorical, which is the price of having a "previous step" to diff against, and it's a flat per-step buffer rather than something that grows with the number of denoising steps.

One more design point I want to be honest about: this has to work in both regimes the task throws at me. In semi-autoregressive block decoding the generation region is split into blocks decoded left-to-right; the stability machinery just runs *within* the current block — I restrict the masked set and the KL/confidence tests to the active block, and I keep a per-block fallback count from the uniform schedule. In fully-parallel decoding the block *is* the whole region (block length equals generation length), and the exact same code runs with a single block. Nothing about the stable-set logic cares which regime it's in; the only thing that changes is whether there's one block or several. That's the property the task demands — one decoder, both regimes — and I get it for free because the per-step decision rule is local to whatever the current block is.

So let me write the decoder. The structure is the shared harness — initialize the all-mask region with the prompt, walk blocks left-to-right, step within a block until it's filled, break early (and skip the forward pass) when a block is done — with the per-step body being: forward pass, softmax in float64, current confidence and argmax, KL against the saved previous distribution, roll it into the history, save the new previous, apply the warmup guard and all-low-history test, intersect stable-and-confident-and-still-masked, and commit either the whole ready set or, if it's empty, the schedule-sized top-\(k\) by confidence.

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
    """KLASS: commit a masked position only when its prediction is both
    high-confidence (top prob > conf_threshold) and dynamically stable
    (last `history_length` consecutive-step KLs all < kl_threshold)."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold        # tau: confidence gate
        self.kl_threshold = kl_threshold            # eps_KL: per-step stability gate
        self.history_length = history_length        # n: how many consecutive low-KL steps required

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()              # prompt + all-mask region
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks                      # == steps when fully parallel

        # vocab size, for the previous-distribution buffer
        V = model.lm_head.out_features if hasattr(model, "lm_head") \
                                       else model.config.vocab_size
        # rolling window of the last `history_length` per-position KLs, and the
        # previous-step categorical at every position (float64: KL of near-equal
        # distributions is a small difference of large logs)
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
        used = 0
        for b in range(num_blocks):                                # blocks left-to-right
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(                    # fixed fallback count u per step
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m                      # masks in the current block only
                if not mask_idx.any():
                    break                                          # block filled; skip the pass

                logits = model(x).logits
                if self.temperature > 0:
                    logits = add_gumbel_noise(logits, self.temperature)
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)                  # token to write if we commit
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)   # top prob

                # per-position KL of current vs previous step: D_KL(p_curr || p_prev)
                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)
                kl_hist = torch.roll(kl_hist, -1, dims=-1)         # window left, newest at the end
                kl_hist[..., -1] = kl
                p_prev = p_curr.clone()                            # become "previous" for next step

                # the guard matches the implementation; the first zero-origin KL
                # fails the all-below-threshold test until it rolls out
                if step >= self.history_length - 1:
                    stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
                else:
                    stable = torch.zeros_like(conf, dtype=torch.bool)
                ready = stable & (conf > self.conf_threshold) & mask_idx

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(ready.shape[0]):
                    rdy = torch.where(ready[j])[0]
                    if len(rdy) > 0:
                        xfer[j, rdy] = True                        # commit the whole ready set
                    else:                                          # fallback: top-u by confidence
                        c = conf[j].clone()
                        c[~mask_idx[j]] = -float("inf")            # only among current-block masks
                        _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                        xfer[j, topk] = True
                x = torch.where(xfer, x0, x)                       # write committed positions
                used += 1
        return x, used                                            # used = forward passes (efficiency)
```

Let me trace the causal chain once, so I'm sure each piece earns its place. The problem was that committing a token is permanent, so committing many per step is fast but risks locking in errors, and the prior strategies all gate on a single-step certainty number — max probability (MaskGIT, LLaDA), the top-1/top-2 margin (Dream), or whether confidence clears a threshold (Fast-dLLM) — which admits tokens the model is confident about *now* but that the still-masked context will overturn. Watching those failures across steps rather than at one frozen step showed the real discriminator: the wrong tokens' distributions keep moving while the correct ones' have settled, so the missing axis is temporal consistency, measured as the step-to-step KL between a position's consecutive categoricals. The δ-approximation argument made this necessary rather than incidental — a token wrong at the under-resolved context but correct at the resolved one is forced, via total-variation endpoint separation, the triangle inequality, and Pinsker, to have average per-step KL at least \(2\Delta^2/M^2\), so persistent low KL is evidence against it being one of these will-flip tokens. Gating on stability *and* confidence (each alone is insufficient) and requiring the last \(n\) consecutive KLs to be low (one is a fluke; \(n=2\) kills the fluke cheaply) gives the stable set; committing the whole stable set per step makes the count adapt to the model's own state and produces the speedup, while a Top-\(u\)-by-confidence fallback when nothing is stable guarantees progress and degrades to the confidence baseline. Float64 and the rolled history with the explicit warmup guard handle the numerics and the cold start. And because the rule is local to the current block, the identical code runs both block-by-block (semi-autoregressive) and on the whole region at once (fully parallel) — one decoder, both regimes, no training, no planner, no growing state.
