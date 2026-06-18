**Problem.** Online RL fine-tuning of Qwen2.5-0.5B on math with a verifiable outcome reward (one
scalar per response, at the last token). PPO would need a policy-sized critic to produce per-token
advantages via GAE, but that critic is a heavy cost and — with the reward only at EOS — nearly
impossible to fit accurately per token. The only editable region is the advantage estimator, so the
floor is the most established critic-free way to turn 16 graded responses per prompt into advantages
the fixed PPO actor loss can ascend.

**Key idea (GRPO advantage).** Replace the learned value baseline with the **empirical group**: for
each prompt's group of `G=16` responses, the group mean reward is a free Monte-Carlo estimate of the
prompt's response-level value, and the group std puts every prompt on a comparable footing. Center each
response's scalar score by its group mean and divide by its group std — a per-prompt z-score — then
broadcast that scalar to every valid token (outcome supervision; with a single terminal reward there is
no within-sequence signal to allocate):

$$\hat A_{i,t} = \frac{r_i - \operatorname{mean}(r_{\text{group}(i)})}{\operatorname{std}(r_{\text{group}(i)}) + \epsilon}\quad\text{for all tokens } t \text{ of response } i.$$

**Why it works.** The group mean is an action-(near-)independent baseline that costs nothing — the
samples are already drawn — and matches the comparative nature of a verifier reward. The z-score makes
advantages scale-invariant across prompts so one high-variance prompt cannot dominate the batch.

**Edit surface note.** In this task only `compute_custom_advantage` is editable; the clipped actor
loss, its token-aggregation, and the KL-to-reference loss are fixed *outside* the edit and applied by
the loop after this function returns. So GRPO here is exactly the advantage half — no KL term, no clip
in the fill. Returns equal advantages (no bootstrapped value target). Everything runs under
`torch.no_grad()`.

**Hyperparameters.** `epsilon=1e-6` denominator floor; `norm_adv_by_std_in_grpo=True` (the std on);
singleton group → mean 0, std 1 (pass the raw score). Fixed loop: 16 samples/prompt, batch 128, 100
steps.

**What to watch.** Should learn (GSM8K holds), but the per-group std reweights prompts by reward spread
— over-weighting too-easy and too-hard prompts, distorting difficulty — and is estimated from only 16
samples, so I expect accuracy left on the table on the harder splits (MATH-500, AMC). That is the test
for step 2: delete the std and see whether the harder splits recover.

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
    """GRPO: Group Relative Policy Optimization advantage estimator.

    Computes outcome-level advantages by normalizing rewards within each
    prompt group by the group mean and standard deviation.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    norm_adv_by_std = True
    if config is not None:
        norm_adv_by_std = getattr(config, "norm_adv_by_std_in_grpo", True)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```
