The GRPO run told me, in numbers, where the per-group std was hurting me. On the easy split it held — GSM8K landed at 0.4668, right where a working critic-free estimator should be, so the loop is learning and the group-mean baseline is doing its job. But the harder splits came in exactly where I feared: MATH-500 at 0.2973 and AMC at 0.0934, with an aggregate `score_mean` of $-0.5166$, the worst of the estimators I have to compare against. That pattern is the tell. The difficulty distortion I flagged bites hardest precisely on prompts whose reward spread is large or whose 16-sample groups are near-unanimous — disproportionately the MATH and AMC prompts — so the floor is not failing to learn; it is leaving accuracy on the table on the harder benchmarks, and the suspect is the one term I was never sure about. Time to interrogate it from the gradient rather than on a hunch.

The reading I had been carrying — "whitening advantages to zero mean and unit variance stabilizes training" — is true, but its *scope* is where GRPO goes wrong. Whitening across an entire batch divides every advantage by one global number, a uniform rescale that folds into the learning rate and changes no relative weighting. But GRPO computes the std *per prompt*, so different prompts are divided by different numbers, and that does change their relative weight. Pull out the centered score $\tilde A_i = r_i - \operatorname{mean}(r)$ and read GRPO as a reweighting: its advantage is $\tilde A_i / \operatorname{std}(r)$. A prompt whose 16 responses are almost all correct has tiny std; one almost all wrong also has tiny std; a genuinely mixed prompt — the informative one — has large std. So $1/\operatorname{std}$ multiplies *up* the weight of the too-easy and too-hard prompts and tamps *down* the mixed ones. That is a difficulty bias that exists purely because the normalization is scoped to a 16-sample group, and it is exactly why the harder splits suffered most.

But "delete the std because it looks bad" is not a derivation. I deleted it on a gradient hunch and then had to check that what is *left* is principled. So I propose **Dr. GRPO** — GRPO with the standard-deviation normalization removed — and I justify it by deriving the gradient from scratch. Maximizing $J(\pi_\theta) = \mathbb{E}_q\,\mathbb{E}_{o\sim\pi_\theta}[R(q,o)]$, the Monte-Carlo policy gradient is $\nabla J = \mathbb{E}[\nabla \log \pi_\theta(o|q)\,R(q,o)]$, and since the log-prob factorizes over tokens, $\nabla J = \mathbb{E}[\sum_t \nabla \log \pi_\theta(o_t|\cdot)\,R(q,o)]$. The standard sharpening: a token at position $t$ cannot influence earlier rewards, so I may replace the return multiplying the $t$-th score by the reward-to-go without changing the expectation. Then subtract an action-independent baseline $B$, which is free in expectation — $\mathbb{E}_{o_t}[\nabla \log \pi_\theta(o_t|\cdot)\,B] = B\,\nabla_\theta \sum_{o_t}\pi_\theta = 0$ — leaving the advantage `reward-to-go` $- B$. Now use the structure of *my* reward: it is outcome-level, zero everywhere except the last token, so the reward-to-go from *any* position $t$ is the same whole-trajectory return $R(q,o)$. That collapses the per-token problem entirely — the advantage is one scalar per response, broadcast across its tokens, and the whole estimator reduces to choosing $B$, computing $r_i - B$, and broadcasting. No std appears anywhere in the gradient. No length term appears. With 16 samples whose mean estimates the prompt's expected return, the natural baseline is the group mean:

$$\hat A_{i,t} = r_i - \operatorname{mean}(r_{\text{group}(i)})\quad\text{for all tokens } t.$$

I then check this centered baseline is not a biased fragment. The mean includes $r_i$ itself, so it is not literally action-independent; the clean multi-sample baseline is leave-one-out, $B_i = \frac{1}{G-1}\sum_{j\ne i} r_j$. Scaling the centered advantage by $G/(G-1)$ recovers it exactly:

$$\frac{G}{G-1}\Big(r_i - \tfrac{1}{G}\textstyle\sum_j r_j\Big) = r_i - \frac{1}{G-1}\sum_{j\ne i} r_j,$$

so mean-only centering *is* the unbiased leave-one-out (RLOO, Ahmadian et al. 2024) policy gradient up to the global constant $(G-1)/G$ that folds into the learning rate. That is the confirmation I needed: deleting the std does not leave a biased remnant of GRPO; it lands me on the unbiased RLOO estimator, rescaled. Cross-checking from the other side, GRPO's effective advantage $(r_i - \operatorname{mean}(r))/\operatorname{std}(r)$ is exactly this correct centered advantage multiplied by a per-prompt $1/\operatorname{std}$ — the one multiplicative distortion the numbers exposed, and the one thing I am removing.

Two consequences make this clean rather than merely smaller. The unanimous-group case self-resolves: with the std gone, an all-correct or all-wrong group simply gives $\tilde A_i = r_i - \operatorname{mean}(r) = 0$ — no relative signal, no contribution this step, which is exactly right — so the $\epsilon$ floor GRPO needed in its denominator is no longer load-bearing. The genuine singleton edge case stays: a single-response group has nothing to compare against, so I set its mean to 0 and let the raw return through. As in step 1, returns equal advantages and everything runs under `torch.no_grad()`.

A scope note: the full Dr. GRPO recipe also replaces the actor loss's per-response $1/|o|$ token-aggregation — a length bias that penalizes long wrong answers less per token — with a constant divisor. But that lives in the *fixed* actor loss, outside the editable region, so I cannot reach it from here; whatever the loop does with `mask.sum(-1)`, it keeps doing. My one lever is the std, and the derivation says removing it is the right and unbiased move. So this fill is exactly GRPO minus the std, and nothing about loss aggregation.

I expect the recovery to be concentrated where the std did its damage: MATH-500 and AMC should edge up from 0.2973 and 0.0934, and the aggregate should move above GRPO's $-0.5166$; GSM8K, mostly easy near-unanimous prompts either way, should hold near 0.4668. Clearing GRPO on the aggregate confirms the difficulty-bias diagnosis. What this *cannot* fix is the length-aggregation bias I can no longer reach, and the loss of any cross-prompt scale — different prompts' advantages are now back on incomparable raw scales, since the per-group std was crudely (and wrongly) the only thing providing that. If Dr. GRPO improves but a *global* scale normalization would do better still, that is the diagnosis already waiting for step 3.

```python
# =====================================================================


@register_adv_est("custom")
def compute_custom_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dr. GRPO: GRPO without standard deviation normalization.

    Computes outcome-level advantages by subtracting the group mean reward,
    without dividing by the group standard deviation.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```
