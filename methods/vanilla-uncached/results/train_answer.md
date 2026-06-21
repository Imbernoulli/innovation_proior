What a language model is supposed to do is capture an unknown distribution $p_{\text{data}}$ over token sequences by fitting $p_\theta$, and "fit" here means maximize $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]$, equivalently minimize $\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)$. That is the whole game. The field has collapsed "do this well" into "do this with the left-to-right factorization," $p_\theta(x) = p_\theta(x^1)\prod_{i\geq 2} p_\theta(x^i \mid x^{<i})$, until the two feel like one statement. They are not. The autoregressive factorization is one way to write a valid joint; it carries baggage, and the cleanest evidence the baggage is real is the reversal curse: a model that learns "A is B" left-to-right fails on "B is ?", because the order it trained in only ever conditioned A→B and it never had a gradient pushing it to predict A from B. That asymmetry is a property of the factorization, not of language. So the question is whether I can keep the principle — maximum likelihood with an expressive transformer — and drop the commitment to a single generation order; if I can, scalability and in-context learning should follow from the principle, and the reversal asymmetry should disappear on its own.

The existing options each fall short of this. BERT's masked language modeling trains a bidirectional transformer to fill in masked positions, but it fixes a single mask ratio (~15%), so it only ever trains one corruption level and supplies no process turning an all-masked sequence into a sample and no likelihood of the data — it is a representation learner, not a generative model. D3PM-style discrete diffusion gives a genuine variational bound, but built for arbitrary transition matrices it materializes full $\bar Q_t$ and compares dense true-versus-model posteriors at every step, which is heavy and trails autoregressive likelihood by a wide margin. MaskGIT trains a bidirectional predictor with a varying mask ratio and decodes iteratively by confidence, but it writes its loss as a flat sum of masked cross-entropies with no weighting across corruption levels and no derivation as a likelihood bound — a sound reconstruction objective, not a maximum-likelihood one, and built for image tokens. And autoregressive LLMs themselves are the yardstick, but the causal factorization is exactly what bakes in the reversal asymmetry and forbids global revision of already-emitted tokens. What I need are three mutually consistent pieces: a process that builds a sequence out of nothing without picking an order up front, a training loss provably bounding $-\log p_\theta$ of that process, and a decoding rule that turns the trained network into samples.

I propose LLaDA — Large Language Diffusion with mAsking — a masked-diffusion language model defined by a forward token-masking process, a learned reverse unmasking process, and a confidence-driven decoder; what I assemble here is its vanilla, uncached form, the plain reference that runs a full bidirectional forward over the entire sequence at every denoising step and reuses nothing. I build the process first, because the loss and the decoder both fall out of it. Tokens are categorical, so the corruption is a Markov chain directly on the categorical state. The choice of transition decides everything: a uniform process that flips any token to any other scrambles meaning so the model cannot tell what was destroyed, whereas the absorbing-state construction makes destruction recognizable — a special $[\text{MASK}]$ symbol $M$ that each token either stays as itself or jumps into, and once masked it stays masked, so the stationary distribution is the all-$M$ sequence, exactly the noise endpoint to start sampling from. On text, masking beats uniform and similarity-structured diffusion, because the model's job becomes the well-posed "fill in the holes" rather than "guess which tokens were silently swapped." Indexing time continuously with $t\in[0,1]$ and factorizing across positions, the forward process masks each token independently with probability $t$,
$$q_{t\mid 0}(x_t^i \mid x_0^i) = 1-t \ \text{ if } x_t^i = x_0^i, \qquad q_{t\mid 0}(x_t^i \mid x_0^i) = t \ \text{ if } x_t^i = M,$$
so at $t=0$ nothing is masked, at $t=1$ everything is, and the expected mask fraction rises linearly. I choose the linear schedule deliberately: text information content is roughly proportional to the number of tokens, so losing information at a constant rate in $t$ is the honest default with no privileged corruption level, and as it turns out linearity is what buys the algebraic simplicity downstream.

The reverse process, from $t$ down to $s<t$, factorizes across positions because the forward process does. A token revealed at $t$ must stay revealed and unchanged (you cannot un-reveal under absorbing corruption); a token masked at $t$ either stays masked or reveals to its true value:
$$q_{s\mid t}(x_s^i \mid x_t) = \begin{cases} 1 & x_t^i \neq M,\ x_s^i = x_t^i \\ s/t & x_t^i = M,\ x_s^i = M \\ \tfrac{t-s}{t}\, q_{0\mid t}(x_s^i \mid x_t) & x_t^i = M,\ x_s^i \neq M \\ 0 & \text{otherwise.}\end{cases}$$
The $s/t$ is forced, not tuned: a token masked at $t$ was masked with probability $t$ and at the earlier $s$ with probability $s$, so given it is masked at $t$ the chance it is still masked at $s$ is the ratio $s/t$, the only choice that keeps the reverse chain's marginals equal to the forward marginals; $1-s/t = (t-s)/t$ is then the reveal mass. The only piece I do not get for free is $q_{0\mid t}(\cdot\mid x_t)$ — which token a masked position reveals to — and this is the lone learnable object. Its crucial property is that it is time-independent: a masked position's clean value depends only on the unmasked tokens around it, which are literal copies of $x_0$ and carry no time information, while the masked positions carry only their count and locations, already implied by the context. So $q_{0\mid t}(x_0^i\mid x_t) = p_{\text{data}}(x_0^i \mid x_t^{\text{UM}})$, a conditional on a clean subset that does not depend on $t$ at all. The predictor therefore needs no timestep embedding, and because it must condition on context on either side of a masked position it must be bidirectional — no causal mask. That single fact kills the reversal asymmetry: a bidirectional masked predictor, by construction, learns to predict any position from any subset of the others.

Now the loss, and this is the load-bearing derivation. The principled objective is the discrete-diffusion variational bound on $-\log p_\theta(x_0)$, the D3PM sum of per-step KLs. Taken literally that is dense and opaque, so I collapse it using two structural facts about the denoiser, both forced by the absorbing process. First, the model never puts mass on $M$ as an output ($\langle p_\theta, m\rangle = 0$, the $M$ logit set to $-\infty$), since $M$ is a corruption symbol, not a real token. Second, an already-unmasked token is carried over unchanged. Under these, look at one reverse step's KL term: for an unmasked position both sides are the delta on the carried-over token, KL $=0$; for a position that stays masked both sides are the delta on $M$, KL $=0$; only masked-at-$t$ positions contribute, and there the cross terms vanish and the KL reduces to a plain cross-entropy between the true clean token and $p_\theta$, scaled by the step's reveal mass. With survival probability $\alpha_t = 1-t$, a reverse step from $t$ to $s$ carries weight $(\alpha_s - \alpha_t)/(1-\alpha_t) = (t-s)/t$ on that cross-entropy. The dense KL has become a weighted sum of masked-token cross-entropies. Pushing the number of steps to infinity gives the continuous-time negative-ELBO,
$$L_\infty = \mathbb{E}_q \int_0^1 \frac{-\alpha'_t}{1-\alpha_t}\,\big[-\log \langle p_\theta(x_t,t), x_0\rangle\big]\,dt,$$
a weighted average of masked-language-modeling losses. Substituting the linear schedule $\alpha_t = 1-t$ (so $\alpha'_t = -1$, $1-\alpha_t = t$) makes the weight exactly $1/t$, giving the training objective
$$L(\theta) = -\,\mathbb{E}_{t,\,x_0,\,x_t}\!\left[\frac{1}{t}\sum_{i=1}^L \mathbb{1}[x_t^i = M]\,\log p_\theta(x_0^i \mid x_t)\right],\qquad t\sim U(0,1],$$
an upper bound on $-\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x_0)]$. The $1/t$ is the whole point. Sampling $t\sim U[0,1]$ and masking each token with probability $t$, at small $t$ only a handful of positions are masked and the inner sum has few terms, while at large $t$ almost everything is masked and the sum is huge; without the $1/t$ the heavily-masked states would dominate the gradient purely by term count, and the model would train mostly on the near-impossible reconstruct-from-nothing regime. The $1/t$ rescales each corruption level so every level contributes comparably — it is the negative-ELBO weight, not a heuristic reweighting, and the linear schedule I chose for honesty is precisely what makes it a clean $1/t$ rather than a schedule-dependent mess. This is exactly what BERT lacks (one fixed ratio, no bound) and what MaskGIT lacks (an unweighted reconstruction sum); the $1/t$ is the bridge from "fill in masks" to "maximum-likelihood generative model." And the construction is order-agnostic in the precise sense that explains the reversal fix: predicting $x_0^{\pi(i)}$ from the already-revealed $\pi(<i)$ is just predicting a masked token from an unmasked subset, so averaging over all masked subsets is the any-order autoregressive objective, which trains every conditional direction including B-given-A and A-given-B.

Training is then embarrassingly close to ordinary LM training: draw $x_0$, draw $t\sim U(0,1]$, mask each token i.i.d. with probability $t$, run one bidirectional forward, take cross-entropy on the masked positions, scale by $1/(tL)$, backprop — no causal mask, no timestep input. Supervised fine-tuning is the identical loss with the mask confined to the response: leave the prompt $p_0$ fully unmasked, mask only the response $r_0$, and minimize $-\mathbb{E}\big[\frac{1}{tL'}\sum_i \mathbb{1}[r_t^i = M]\log p_\theta(r_0^i\mid p_0, r_t)\big]$. For likelihood scoring I switch to a lower-variance estimator with the same expectation: mask exactly $l\sim U\{1,\dots,L\}$ tokens uniformly without replacement so the mask count is deterministic at $l/L$, and reweight by $L/l$; this stabilizes with roughly a hundred Monte-Carlo draws against roughly a thousand for the independent-mask form.

Sampling discretizes the reverse process into $N$ steps from a fully-masked response. At each step from $t$ down to $s = t - 1/N$, I forward $(p_0, r_t)$, predict every masked position (argmax for a deterministic greedy reveal), then put back the right amount of mask to land on $r_s$ rather than $r_0$: the posterior says a token masked at $t$ stays masked with probability $s/t$, so a fraction $s/t$ of the freshly-predicted tokens is re-masked. The exact reverse process re-masks at random, but that wastes information — it throws away confidently-correct tokens and commits to garbage, and committed tokens are frozen for the rest of the rollout by carry-over. So the default is low-confidence remasking: keep the highest-confidence predictions, re-mask the lowest-confidence ones, with confidence $c^i = p_\theta(r_0^i\mid p_0, r_t)_{r_0^i}$ and committed tokens given $c=1$, keeping $n_{\text{un}} = \lfloor L(1-s)\rfloor$ positions unmasked at time $s$. This departs from the true posterior deliberately: it is the discrete analogue of annealed low-temperature autoregressive sampling, trading diversity for accuracy on factual, math, and code tasks, and it builds a coarse-to-fine schedule for free since easy tokens lock in early and become context that makes hard tokens easier later — and it is documented to beat random remasking substantially, so I keep random only as the faithful-but-weaker reference. The per-step reveal count follows from the linear schedule's constant unmask rate: over $N$ steps with $M$ masked tokens, reveal $\lfloor M/N\rfloor$ each step and spread the remainder $M \bmod N$ one-each over the early steps, which reveals all $M$ in exactly $N$ steps. I also expose block partitioning: run full diffusion within a contiguous block and move left-to-right across blocks, with block length equal to the whole generation giving pure diffusion and block length one giving autoregression, the knob trading global revisability against locality.

The defining choice of the vanilla form is to decline the cache the construction makes possible. Because the predictor is time-independent and revealed tokens never change, at most $L$ distinct inputs ever occur over a rollout, which invites caching. But there is no causal prefix — the moment any masked token reveals it changes the bidirectional context for every other position, so an early position's correct prediction genuinely depends on tokens revealed later, and a stale cached value can be wrong. The conservative, exactly-correct thing is a full bidirectional forward over the entire current sequence at every single denoising step: no prefix KV cache (the keys and values of committed tokens keep being re-attended as new tokens reveal), no feature cache, no skipped positions. That is the plain reference rollout below, with optional classifier-free guidance (a second forward with the prompt also masked, logits pushed along $\text{un\_logits} + (\text{cfg\_scale}+1)(\text{logits}-\text{un\_logits})$), Gumbel-max sampling in float64 because low-precision Gumbel-max hurts generation quality, and the EOS switches kept exposed because heavy EOS padding in SFT can make low-confidence remasking terminate responses too early.

```python
import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Gumbel-max categorical sampling; float64 because low-precision Gumbel-max
    degrades generation quality for masked diffusion. temperature 0 -> argmax."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Linear schedule => equal expected reveals per step: floor(M/steps) each,
    remainder spread over the early steps."""
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
    """Vanilla no-cache reverse generation: a full bidirectional forward over the
    whole sequence at every denoising step. block_length < gen_length => semi-AR."""
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
            if cfg_scale > 0.:                            # optional unsupervised classifier-free guidance
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = None
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits  # full uncached forward

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf        # suppress early EOS for some tasks

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf  # current block only

            x0 = torch.where(mask_index, x0, x)          # carry over revealed tokens
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):         # keep top-k confident; re-mask the rest
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
```
