**Problem (from step 2).** Dr. GRPO confirmed the difficulty-bias diagnosis — deleting the per-group
std lifted the aggregate (`score_mean` −0.5166 → −0.5049) with MATH-500 and AMC recovering (0.2973 →
0.3070, 0.0934 → 0.1009) and GSM8K flat. But deleting the std also removed the one *legitimate* thing
it did: cross-prompt scale normalization. Different prompts' centered returns now enter the gradient on
incomparable raw scales, which still costs accuracy (AMC stuck at 0.1009).

**Key idea (REINFORCE++-baseline — center locally, whiten globally).** Keep the group-mean baseline
(the part that worked); add back scale normalization, but compute it from a pool large enough to be a
*constant* rather than from the 16-sample group. Two steps:

$$A'_i = r_i - \operatorname{mean}(r_{\text{group}(i)}),\qquad A^{\text{norm}} = \text{masked\_whiten}\big(\text{broadcast}(A'),\ \text{response\_mask}\big).$$

The whitening pools over **all valid response tokens in the batch** (~2000+ tokens), so it is
length-weighted and consistent with the per-token policy loss.

**Why it works.** The group std failed because of its *scope* (per-prompt → difficulty bias) and its
*size* (16 samples → brittle, biased: `E[D²|ε_i]` grows with `ε_i²`, so large advantages get
compressed). Both vanish as the pool grows: a global-batch std behaves like a constant. Group-mean
centering stays for per-prompt difficulty and reward-scale reshaping (0/1 and −1/1 both center near
zero); the global whiten supplies cross-prompt scale without the per-group distortion.

**Edit surface note.** The reasoning-task variant also adds a separate reverse-KL loss (k2 =
`½(log(π_θ/π_ref))²` differentiates correctly), but the KL-loss setting is **fixed outside** the
editable region. Only `compute_custom_advantage` is mine, so this fill is purely the two-step advantage.
Singleton group → mean 0 (global whiten centers it). Returns equal advantages; runs under
`torch.no_grad()`.

**Hyperparameters.** Group-mean center → broadcast → `verl_F.masked_whiten` over valid tokens. No
per-group std. Fixed loop: 16 samples/prompt, batch 128, 100 steps.

**What to watch.** The gain should come from one stable cross-prompt scale, not a second
difficulty-bias fix; I expect the aggregate to clear dr_grpo's −0.5049, with the token-level whitening
helping most where lengths/reward scales vary (MATH, AMC). GSM8K (short, uniform prompts) should hold
near 0.4671. Clearing dr_grpo confirms the std's *scope*, not the std itself, was the problem.

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
