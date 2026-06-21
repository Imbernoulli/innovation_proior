When I generate from a masked diffusion language model, the model does not fill the sequence left-to-right; it fills it by unmasking. I set the whole generation region to the mask token, and over a fixed number of denoising steps the network looks at the entire partially-filled sequence and hands me, for every still-masked position, a categorical distribution over the vocabulary. At each step I decide three things: how many positions to commit, which ones, and what token to write into each. The token is the easy part — I take the model's own argmax (or a sample) — and because the carry-over (SUBS) parameterization fixes a position once it is revealed, whatever I commit stays committed for the rest of the generation. That permanence is the whole tension. If I commit one position per step the quality is good, but I pay one full forward pass per generated token, which is just autoregressive speed and throws away the diffusion model's defining ability to predict many positions at once. If I commit many positions per step on a fixed schedule I go fast, but I commit tokens the model is not actually sure about, and since commit is permanent I lock in errors that the still-masked context would later have corrected. Steps are my budget and forward passes are my cost, so I want to cut steps; but every step I cut by committing more is a step where I might commit garbage. I want to commit *many* tokens per step exactly when it is safe, and otherwise fall back to a conservative schedule-sized choice — training-free, with no auxiliary model and no growing memory, working both in semi-autoregressive block decoding and in fully-parallel decoding.

The prior strategies all share one blind spot. MaskGIT predicts every masked position at once, keeps the most confident — confidence being the top probability $\mathrm{conf}^i = \max_v p^i(v)$ — remasks the rest, and repeats under a decreasing schedule. LLaDA's low-confidence remasking is the same idea at LM scale: predict all, keep the high-confidence ones, equivalently unmask the top-$k$ by max probability with $k$ fixed by the uniform schedule $\lfloor(\#\text{masks})/\text{steps}\rfloor$. Dream sharpens the signal to the margin $p^i_{(1)}-p^i_{(2)}$, which correctly punishes a near-tie that max probability cannot see. Fast-dLLM makes the count adaptive: drop the schedule and unmask *every* position whose confidence clears an absolute threshold $\tau$, falling back to the single most-confident position if none clears it, and it even proves that if the model's per-position marginals are all confident enough — $p_\theta(x^\star_{i_j}\mid E) > 1-\epsilon$ with $(n+1)\epsilon \le 1$ — then greedy parallel decoding from the product of marginals picks the same sequence as greedy sequential decoding from the true joint, the bound tight at $\epsilon = 1/(n+1)$. That is a real license to parallelize in the high-confidence regime. But every one of these keys off a single number read at a single step. Confidence at one step is not the same as being right: I keep seeing a sampler commit a high-confidence token that is wrong, while the truly correct token had lower confidence at that instant and got passed over. Fast-dLLM's theorem cannot save me here, because its premise is "the marginal is correct enough," and the entire failure is a confident marginal that is confidently wrong while the context that would overturn it is still masked. Confidence is necessary — I will not commit a position the model is unsure of — but it is plainly not sufficient.

So I stopped staring at the formula and watched the failures across steps instead of at one frozen step. Take a position where a confidence-only sampler commits a high-confidence wrong token, and watch its distribution over the denoising steps: it is still moving. The argmax may stay put and look confident, but the shape underneath is churning, because the surrounding context is still being resolved and each newly revealed neighbor nudges the belief here. A position whose commitment no longer looks fragile has the opposite signature: its distribution has stopped moving — the model decided several steps ago and every subsequent step leaves it essentially unchanged. The discriminating thing is therefore not *how tall* the peak is at this instant but whether the belief at the position has *settled*. Confidence is a snapshot; what I want is temporal consistency, and I can read it directly from the distributions the sampler already produces, as a divergence between the categorical I had last step and the one I have now.

I propose KLASS — KL-Adaptive Stability Sampling. The defining move is to gate each commit on **two** signals rather than one: a confidence score $\mathrm{conf}_t^i = \max_v p_t^i(v)$, and a step-to-step KL score $d_t^i = D_{\mathrm{KL}}(p_t^i \,\|\, p_{t+1}^i)$ comparing the current step's distribution against the previous step's at the same position. Near-zero KL means the model is not changing its mind there; large KL means it still is. I commit a masked position only if it is both stable (recent KL low) and confident (top probability high) — not either. Confidence alone is the prior art and admits confident-but-wrong tokens; KL alone would let me commit a position whose belief has frozen but at a low, mushy probability where the model is not actually committing to anything. Together they say: the model has *settled* on this and is *sure* of it.

Before trusting an empirical observation I wanted to know whether instability is *forced* on a wrong-but-correctable token, not merely incidental. Suppose the model is a conditional $\delta$-approximation of the task: for every context $c$, its distribution at position $i$ is within total variation $\delta$ of a correct one. Fix $i$ with a genuinely correct token $x^\star$ that, at the resolved context $c^\star$, beats a suboptimal $x^\dagger$ by margin $\gamma$, while *right now* at the under-resolved context $c_M$ (most neighbors still masked) the model prefers the wrong token, $p_\theta(x^\dagger\mid c_M) \ge p_\theta(x^\star\mid c_M)+\beta$. As denoising fills the other positions the context walks $c_M\to\cdots\to c_0=c^\star$, changing only variables outside $i$; write $P_t = p_\theta(\cdot\mid c_t)$. Pick the test function $f=\mathbf 1\{x_i=x^\dagger\}-\mathbf 1\{x_i=x^\star\}$ with $\|f\|_\infty\le 1$, and let $A_M=p_\theta(x^\dagger\mid c_M)-p_\theta(x^\star\mid c_M)$, $A_0=p_\theta(x^\dagger\mid c^\star)-p_\theta(x^\star\mid c^\star)$. Total variation controls any bounded function's change in expectation, so $2\,\mathrm{TV}(P_M,P_0) \ge |\mathbb E_{P_M}[f]-\mathbb E_{P_0}[f]| = |A_M-A_0|$. The current-preference assumption gives $A_M\ge\beta$; the $\delta$-approximation at $c^\star$ gives $p_\theta(x^\star\mid c^\star)-p_\theta(x^\dagger\mid c^\star)\ge \gamma-2\delta$, i.e. $A_0\le-(\gamma-2\delta)$. Hence $A_M-A_0\ge\beta+\gamma-2\delta$, and taking the positive part,
$$\mathrm{TV}(P_M,P_0) \;\ge\; \tfrac12(\beta+\gamma-2\delta)_+ \;=:\; \Delta.$$
The endpoints are far apart whenever the true margin and the current wrong-preference together beat twice the approximation error. To turn endpoint separation into per-step movement — which is what the KL score measures — I use the triangle inequality, $\sum_{t} \mathrm{TV}(P_{t+1},P_t)\ge \mathrm{TV}(P_M,P_0)\ge\Delta$, then Pinsker, $D_{\mathrm{KL}}(P_t\|P_{t+1})\ge 2\,\mathrm{TV}(P_{t+1},P_t)^2$, and Cauchy–Schwarz, $(\sum_t T_t)^2 \le M\sum_t T_t^2$, to land at
$$\frac1M \sum_{t=0}^{M-1} D_{\mathrm{KL}}(P_t \,\|\, P_{t+1}) \;\ge\; \frac{2\Delta^2}{M^2}.$$
A wrong-but-correctable token cannot keep its average per-step KL near zero — it is forced to be dynamically unstable somewhere along the path. So persistent low KL is exactly the evidence I want before treating a position as unlikely to flip; instability is the mathematical fingerprint of a token about to change its mind.

One KL reading is not enough, though. A single low KL between two consecutive steps can be a fluke — a position can happen not to move for one step and then lurch the next, especially early when most context is still masked. The theory bounds the *average* KL away from zero, which is fully consistent with a momentary dip, so I demand evidence of *sustained* stillness: the last $n$ consecutive-step KLs must *all* be below threshold. That converts "it didn't move this step" into "it hasn't moved for $n$ steps," and a will-flip token can sneak one low KL past me but not a run of them. The stable set is therefore
$$S_t = \Big\{\, i \;\Big|\; \big(\forall k \in \{1,\dots,n\}:\, D_{\mathrm{KL}}(p_{t+k-1}^i \,\|\, p_{t+k}^i) < \epsilon_{\mathrm{KL}}\big) \;\wedge\; \mathrm{conf}_t^i > \tau \,\Big\}.$$
For $n$ there is an obvious sweet spot. $n=1$ is the trigger-happy version the fluke argument just warned against; large $n$ is safe but stingy, declaring almost nothing stable and surrendering the speedup. $n=2$ is the natural default: it kills the single-step fluke without paying for a long history. A history window of length $n$ also means I must not evaluate stability until the window has filled, so for the first $n-1$ steps of a block nothing is eligible — a deliberately conservative guard, and any startup artifact still in the rolling buffer fails the all-below-threshold test until it rolls out.

The unmasking rule is where the speedup actually comes from. At each step I commit *every* position in $S_t$ in parallel, however many there are. At a settled step many positions cross both thresholds at once and get written together in one forward pass; at a churning step $S_t$ is small or empty. The count adapts to the model's own state instead of a schedule — Fast-dLLM's adaptivity, but gated on stability-and-confidence rather than confidence alone. The one remaining worry is the empty case: committing nothing makes no progress, and too-strict thresholds could stall the chain forever. So when $S_t=\emptyset$ I unmask the Top-$u$ masked positions by confidence, where $u$ is the fixed schedule count — exactly what a plain confidence sampler would do at that step.
$$x_t^i = \begin{cases} \text{commit position } i, & i \in S_t,\\[2pt] \text{else commit the Top-}u\text{ positions by } \mathrm{conf}_t^i, & S_t = \emptyset. \end{cases}$$
The worst case for KLASS is therefore "behaves like the confidence baseline," the best case is "commits a whole settled block in one shot," and it interpolates between them automatically; the schedule-sized fallback generalizes Fast-dLLM's always-unmask-the-top guarantee so the fallback itself cannot stall.

A few implementation points are load-bearing. The KL is $\sum_v p_t^i(v)\,(\log p_t^i(v)-\log p_{t+1}^i(v))$, and in the settled regime I care about most this is a small difference of large logs, so float32 would give me noise where I need to read a threshold of order $10^{-2}$; low-precision softmax is also known to hurt MDM quality. I therefore do both the softmax and the KL in float64, and floor the logs with a tiny $\epsilon=10^{-12}$ inside $\log(p+\epsilon)$ to keep the masked-out tail from producing infinities. At the first step of a block there is no previous distribution, so the previous buffer starts at zero and the zero-origin KL is huge — but this never yields a false stability verdict: step 0 has stability forced off, step $n-1$ first permits the test but the huge zero-origin KL is still in the window and fails it, and only after that value rolls out does the window hold $n$ genuine consecutive-step KLs. I keep the KL history as a rolling buffer per position, rolling left and writing the newest KL into the last slot each step. I also pin the orientation: each step I form $p_{\mathrm{curr}}$, compute $D_{\mathrm{KL}}(p_{\mathrm{curr}}\|p_{\mathrm{prev}})$ against the saved previous distribution, push it, then overwrite $p_{\mathrm{prev}}\leftarrow p_{\mathrm{curr}}$ — and since diffusion steps count down from $T$, the current distribution is $p_t$ and the saved one is $p_{t+1}$, so this is exactly $D_{\mathrm{KL}}(p_t^i\|p_{t+1}^i)$, matching the path argument. The whole thing is genuinely lightweight: every signal is the base model's own output across its own successive predictions — no planner, no second model, no alignment — with the KL history $O(L\cdot n)$ and the previous-distribution buffer $O(L\cdot V)$, a flat per-step copy of the per-position categorical rather than anything that grows with the step count. And because the rule is local to the current block, the identical code runs both block-by-block (semi-autoregressive) and on the whole region at once (fully parallel): one decoder, both regimes, for free.

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
    """KLASS: KL-adaptive stability sampling. Commit a masked position only when
    its prediction is both high-confidence and dynamically stable (the last
    `history_length` consecutive-step KL divergences are all below kl_threshold).
    Works in semi-autoregressive (block_length < gen_length) and fully-parallel
    (block_length == gen_length) regimes; returns (sequence, forward passes)."""

    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold        # tau
        self.kl_threshold = kl_threshold            # eps_KL
        self.history_length = history_length        # n

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        V = model.lm_head.out_features if hasattr(model, "lm_head") \
                                       else model.config.vocab_size
        # rolling window of last `history_length` per-position KLs; previous-step
        # categorical at every position (float64: KL of near-equal distributions
        # is a small difference of large logs).
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
        used = 0
        for b in range(num_blocks):
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(             # fixed fallback count u per step
                (x[:, bs:be] == mid), steps_per_block)
            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m               # masks in the current block only
                if not mask_idx.any():
                    break                                   # block filled; skip the pass

                logits = model(x).logits
                if self.temperature > 0:
                    logits = add_gumbel_noise(logits, self.temperature)
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)

                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)  # D_KL(p_curr || p_prev)
                kl_hist = torch.roll(kl_hist, -1, dims=-1)
                kl_hist[..., -1] = kl
                p_prev = p_curr.clone()

                if step >= self.history_length - 1:
                    stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
                else:
                    stable = torch.zeros_like(conf, dtype=torch.bool)
                ready = stable & (conf > self.conf_threshold) & mask_idx

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(ready.shape[0]):
                    rdy = torch.where(ready[j])[0]
                    if len(rdy) > 0:
                        xfer[j, rdy] = True                 # commit the whole stable+confident set
                    else:
                        c = conf[j].clone()
                        c[~mask_idx[j]] = -float("inf")     # fallback: Top-u by confidence
                        _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                        xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used                                      # used = model forward passes
```
