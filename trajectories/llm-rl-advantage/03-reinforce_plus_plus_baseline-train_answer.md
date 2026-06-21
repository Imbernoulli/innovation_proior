Dr. GRPO confirmed the difficulty-bias diagnosis on exactly the splits I predicted: deleting the per-group std moved the aggregate `score_mean` from $-0.5166$ up to $-0.5049$, with MATH-500 going $0.2973 \to 0.3070$ and AMC $0.0934 \to 0.1009$ while GSM8K barely twitched ($0.4668 \to 0.4671$). The harder splits recovered, the easy split held — the per-group std was a distortion, not a stabilizer. But the gain was only $\sim 0.012$ in score and AMC at 0.1009 is still leaving a lot on the table, and I know why: deleting the std also threw away the *one legitimate job* it was doing badly — putting different prompts' advantages on a comparable scale. Dr. GRPO's centered returns now enter the gradient on incomparable raw scales. So the question for this step is sharp: can I get the legitimate scale normalization back *without* bringing back the difficulty bias?

To attack the right thing I have to be precise about why the std was bad, and it failed for two reasons that share a root. First, it is *scoped to the group* — 16 samples — so dividing by it reweights prompts by their reward spread. Second, even ignoring scope, a 16-sample std is a brittle scale estimate: near-unanimous groups, common with outcome rewards, make it tiny and noisy. Both come from estimating the scale from the same handful of samples as the numerator, and from too few of them. So the fix is not to "estimate the std more carefully within the group" — the group is small by construction and I cannot enlarge it without paying $16\times$ the rollout cost. The fix is to change the *pool* I standardize over.

To see that the pool size is the controlling variable, model a group as $r_i = \theta + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$ i.i.d. The std-normalized advantage is $A_i = (\varepsilon_i - \bar\varepsilon)/D$ with $D = \sqrt{\tfrac{1}{N}\sum_j (\varepsilon_j - \bar\varepsilon)^2}$. Conditioning on $\varepsilon_i$, the denominator's second moment is

$$\mathbb{E}[D^2 \mid \varepsilon_i] = \frac{(N-1)^2}{N^2}\sigma^2 + \frac{N-1}{N^2}\varepsilon_i^2 \;=\; \alpha + \beta\,\varepsilon_i^2,\qquad \beta = \frac{N-1}{N^2} > 0.$$

The conditional scale *grows with* $\varepsilon_i^2$: a sample with large $|\varepsilon_i|$ inflates the very spread it is divided by, compressing large advantages. A cleaner non-asymptotic view: with $z_j = \varepsilon_j - \bar\varepsilon$ and $\sum z_j = 0$, the others sum to $-z_i$, so $\sum z_j^2 \ge N z_i^2/(N-1)$ and hence $|A_i| \le \sqrt{N-1}$ — bounded for every finite $N$ while $\varepsilon_i$ is unbounded, so the std-normalized estimator is biased. But watch the $N$ dependence: as $N \to \infty$, $\beta \to 0$ and $\alpha \to \sigma^2$, the dependence on the individual sample vanishes, and $D$ converges to the true $\sigma$. The disease is *small $N$*. The cure is more samples in whatever pool I standardize over.

I cannot grow the per-prompt group, but the loop hands me a much larger pool every step: 128 prompts $\times$ 16 $\approx$ 2000 responses across many prompts. So I propose **REINFORCE++-baseline** — center locally, whiten globally. Keep the group-mean baseline that worked, and add scale normalization back, but compute it from the global batch rather than the 16-sample group. By the limit above, batch statistics behave like stable constants, the small-sample bias shrinks, and no single uninformative prompt can set the scale for itself; both failures of the group std weaken together for the same reason. This is not exotic — per-minibatch advantage whitening before the policy loss is a standard PPO implementation detail (Andrychowicz et al. 2020). What is deliberate is the *pool*: the global batch, precisely because pool size controls both the bias and the stability.

The construction is two steps. First reshape with a local baseline, exactly the Dr. GRPO advantage that already worked, $A'_i = r_i - \operatorname{mean}_{\text{group}}(r)$; then standardize $A'$ over the whole batch:

$$A'_i = r_i - \operatorname{mean}(r_{\text{group}(i)}),\qquad A^{\text{norm}} = \text{masked\_whiten}\big(\text{broadcast}(A'),\ \text{response\_mask}\big).$$

I keep the *group mean* and throw away only the *local std* for two reasons the Dr. GRPO numbers vindicated. The per-prompt mean carries real information — "this response beat the typical response *to this prompt*" is far more meaningful than "this raw reward is high" — and it *reshapes* the reward to be robust to scale: a 0/1 verifier with group mean 0.5 and a $-1/1$ verifier both center advantages about zero, so I never have to redesign rewards for the two conventions. The group mean still includes the sample itself, carrying the same finite $(1 - 1/N)$ shrinkage I accounted for in Dr. GRPO, but it is a *stable centering* step; it was the local *denominator* that was biased and brittle. So group mean for per-prompt reshaping, batch std for cross-prompt scale computed from a pool large enough to be a constant — scale normalization back without the difficulty bias, because the scale is no longer per-prompt. That is the precise answer to the question this step opened with.

Two choices make the second step concrete. The singleton edge case: a prompt with one sample in the batch has no "other responses," and subtracting it from itself would zero its signal, so I set its group mean to 0 — keep the raw score and let the *global* whitening center it relative to the batch. And the *when* of whitening, because the actor loss is per-token while $A'_i$ is one scalar per response. If I whiten the per-sequence scalars and then broadcast, every response counts once regardless of length; if I broadcast *first* and whiten over the pool of tokens, a 600-token response contributes 600 entries to the statistics and a 50-token one contributes 50, making them length-weighted. For a token-level policy loss the consistent choice is the latter: standardize the actual per-token quantities that enter the gradient, over the actual set of tokens that enter it. So I broadcast the centered scalar to all valid tokens, mask the padding, whiten over all valid response tokens in the batch via `verl_F.masked_whiten`, then mask again. The variance floor inside the whiten is now a numerical guard under a *batch* variance, not a load-bearing scale.

A scope note: the reasoning-task variant of this method also adds a separate KL-to-reference loss term, and the right estimator there is subtle — sampling from $\pi_\theta$ constrains the reverse KL, and of the usual candidates only $k_2 = \tfrac{1}{2}(\log(\pi_\theta/\pi_{\text{ref}}))^2$ differentiates to the reverse-KL gradient without an exploding importance weight. But the KL-loss setting is fixed outside my edit surface, so I do not write the KL term; my job is purely the advantage. This fill is exactly the two-step estimator: group-mean center, broadcast to tokens, masked-whiten over the batch's valid tokens, mask.

So this is Dr. GRPO plus one operation — where Dr. GRPO stopped after group-mean centering, I broadcast to tokens and whiten over the global batch — and the delta is the global scale the deleted std could never safely provide. Dr. GRPO already fixed the difficulty bias, so I do not expect a second large jump on MATH/AMC from the *same* mechanism; the gain this step should come from putting all prompts on one stable scale, helping most where incomparable raw scales and varying lengths still inject noise (MATH and AMC again). GSM8K — short, uniform prompts — should hold near 0.4671, with least to gain from length-weighted global whitening. If the batch-whitened estimator clears Dr. GRPO's $-0.5049$ on the aggregate, the missing-scale diagnosis is confirmed: the std was never the problem, its *scope* was, and the fix is to keep the legitimate scale normalization but compute it from a pool large enough to be a constant.

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
    """REINFORCE++-baseline: group-centered reward, token-level batch whitening.

    Subtract per-prompt group mean from each response's scalar reward,
    broadcast to token level, then masked-whiten over all valid response
    tokens in the batch (longer responses contribute more to the whitening
    statistics).
    """
    response_length = token_level_rewards.shape[-1]
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
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores
```
