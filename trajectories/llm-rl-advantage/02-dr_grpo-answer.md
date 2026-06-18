**Problem (from step 1).** GRPO learns (GSM8K 0.4668) but lands worst on the aggregate
(`score_mean` −0.5166) with depressed harder splits (MATH-500 0.2973, AMC 0.0934). The suspect is the
per-group std: scoped to a 16-sample group, `1/std` reweights prompts by reward spread — inflating
too-easy and too-hard prompts, tamping down the informative mixed-outcome ones — a difficulty bias that
bites hardest exactly on the harder benchmarks.

**Key idea (Dr. GRPO — drop the std).** Keep the group-mean baseline; delete the per-group standard
deviation. The advantage is the mean-only centered return, broadcast to every token:

$$\hat A_{i,t} = r_i - \operatorname{mean}(r_{\text{group}(i)})\quad\text{for all tokens } t.$$

**Why it works.** Deriving the policy gradient from scratch with an action-independent baseline, and
using the outcome-reward fact that the advantage is one scalar per response, shows the gradient asks
only for a *centered* return — no std appears. The algebra confirms it is principled, not a fragment:
`(G/(G−1))·(r_i − mean(r)) = r_i − mean_{j≠i}(r_j)`, so mean-only centering is the unbiased
leave-one-out (RLOO) policy gradient up to the global constant `(G−1)/G`. Removing the std removes the
difficulty distortion; the unanimous-group case self-resolves (advantage → 0), so the `ε` floor is no
longer load-bearing.

**Edit surface note.** The full Dr. GRPO recipe also replaces the loss's per-response `1/|o|`
token-aggregation (a length bias) with a constant divisor — but that lives in the **fixed** actor loss,
outside the editable region. Only `compute_custom_advantage` is mine, so this fill is exactly GRPO
*minus the std* and nothing about loss aggregation. Returns equal advantages; runs under
`torch.no_grad()`.

**Hyperparameters.** No std; group-mean baseline only. Singleton group → mean 0 (raw return passes
through). Fixed loop: 16 samples/prompt, batch 128, 100 steps.

**What to watch.** The harder splits are where the std reweighting did its damage, so MATH-500 and AMC
should recover from grpo's 0.2973 / 0.0934 and the aggregate should clear grpo's −0.5166; GSM8K should
hold near 0.4668. What this cannot fix: the length-aggregation bias (unreachable from this edit) and the
loss of any cross-prompt scale — advantages are now on incomparable raw scales across prompts. If a
*global* scale normalization would do better, that is the step-3 diagnosis.

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
