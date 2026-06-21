I am RL-tuning Qwen2.5-0.5B on math with a verifiable outcome reward — one scalar per response, emitted by a rule-based verifier at the last valid token and zero everywhere else — and the only thing I get to design is the advantage estimator the fixed PPO actor loss will ascend. The textbook route is actor-critic: train a policy-sized value network $V_\psi$ and build per-token advantages with GAE, $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. On one GPU that roughly doubles the trainable footprint, but the deeper trouble is that GAE needs an accurate $V$ at *every* timestep, and the reward here is a single end-of-sequence bit. Asking a critic to smear that terminal scalar backward into accurate per-token values is precisely the regression the last-token-only reward makes hardest, so I would be paying for a heavy network whose central job is the one this setting sabotages. I want the stable clipped update the loop already gives me and a variance-reducing baseline; I do not want a learned per-token critic.

The way out is to recall *why* the critic was ever there. A baseline reduces variance for free in expectation: any $b$ that depends on the state but not the action being scored leaves the gradient unbiased, since $E_a[\nabla_\theta \log \pi_\theta(a|s)\, b(s)] = b(s)\,\nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\,\nabla_\theta 1 = 0$, and the variance-minimizing choice is the state value. So the baseline need only *approximate the expected return from here* — it need not be a learned network. And I already have, lying around unused, exactly the samples that estimate that quantity: the loop draws $G=16$ responses per prompt and grades each one. Their empirical mean is a Monte-Carlo estimate of the prompt's response-level value at the only granularity the reward actually exists. It costs nothing extra; the rollouts are already paid for.

I propose **GRPO** — Group Relative Policy Optimization at the level of the advantage. For each prompt's group of $G$ responses, I sum the per-token rewards to recover each response's scalar score $r_i$, center it by the group mean, and standardize by the group standard deviation, giving a per-prompt z-score that I broadcast unchanged to every valid token of the response:

$$\hat A_{i,t} = \frac{r_i - \operatorname{mean}(r_{\text{group}(i)})}{\operatorname{std}(r_{\text{group}(i)}) + \epsilon}\quad\text{for all tokens } t \text{ of response } i.$$

The group mean is the load-bearing piece. It is a (near-)action-independent baseline that costs nothing, and it matches the *comparative* nature of the verifier signal: "this solution beat the typical solution to this problem" is far more informative than "this raw reward is high," which mostly tells me the problem was easy. It is not exactly action-independent — $r_i$ is one of the terms in its own mean — and the cleanest REINFORCE-style construction would use a leave-one-out mean over the other $G-1$ responses; the full group mean is symmetric, cheap, and at $G=16$ differs only by a $1/G$ self-inclusion effect, so I take it as the practical estimator and keep the sign structure I need.

The standard-deviation denominator is there because the spread of $r_i$ differs wildly across problems. An easy prompt where all 16 samples are correct has its rewards bunched at the top with tiny variance; a hard one has them splayed across the range. With raw $r_i - \operatorname{mean}(r)$ the easy prompt would contribute microscopic advantages and the hard one huge ones, purely because of reward scale rather than because one matters more for learning. Dividing by the group std makes the advantage scale-invariant across problems — each prompt now says "this response was *this many standard deviations* better than typical" — and controls update magnitude so one high-variance prompt cannot dominate the batch. The $\epsilon$ floor guards the zero-variance group (all 16 samples identical) against division by zero; a singleton group has no spread, so I set its mean to 0 and std to 1 and let the raw score through.

One thing falls out of the information I have. The fixed actor loss wants a genuinely per-token advantage, but with no critic and a single terminal reward there is nothing to distinguish tokens within a response — the reward-to-go from any position equals the whole-sequence return, since the only reward is at EOS. So the honest minimal choice is outcome supervision: assign the same normalized scalar $\hat A_i$ to every token of the response. With no bootstrapped value target to compute, the returns tensor the loop expects is the same tensor as the advantages.

A note on scope, because the established critic-free recipe is more than this formula — it also moves the KL-to-reference anchor out of the reward and into the loss and dual-clips the surrogate. But the only editable region here is `compute_custom_advantage`; the actor loss, its clipping, and the KL-loss setting are fixed outside it and applied by the loop after my function returns. So GRPO in this task is precisely the advantage half — group-mean center, divide by group std, broadcast, mask — with no KL term and no clip in the fill, and everything computed under `torch.no_grad()`.

I expect this floor to learn — GSM8K accuracy should hold up — but the per-group std is the term I trust least. Scoping the whiten to a 16-sample group means different prompts are divided by different numbers, so $1/\operatorname{std}$ silently inflates the weight of near-unanimous (too-easy and too-hard) prompts and tamps down the informative mixed-outcome ones, a difficulty distortion that bites hardest exactly on the harder splits (MATH-500, AMC), and the std itself is the noisiest object in the estimator. If the per-group std turns out to be a distortion rather than a stabilizer, the cleanest next test is to delete it and watch whether the harder splits recover.

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
