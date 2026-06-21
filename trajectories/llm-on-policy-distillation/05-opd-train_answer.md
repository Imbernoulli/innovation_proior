Symmetric JSD did exactly what its half-mode-seeking construction promised. `gkd` reads GSM8K 0.4716, MATH-500 0.312: the MATH-500 recovery I predicted is there and then some — 0.312 is the highest of any rung so far, reversing TAID's 0.280 collapse and clearing the dagger/RS-KD cluster decisively, the cleanest confirmation that the MATH-500 failures were a *divergence-direction* problem, not a target or data problem. GSM8K crept to 0.4716, a touch over TAID, exactly the smaller move I expected where mass-covering hurt least. The divergence direction is the binding axis, and moving toward mode-seeking is what pays. The open question $\beta=0.5$ leaves is whether half is far enough: the MATH-500 win says mode-seeking helps, the modest GSM8K move says half of it may be leaving value on the table. So I push the divergence *all the way* to the reverse-KL endpoint and ask whether full mode-seeking on the student's own rollouts beats the balanced compromise.

I propose **OPD**: per-token reverse KL,
$$\mathrm{KL}(p_S \,\|\, p_T) = \sum_v p_S(v)\,\big(\log p_S(v) - \log p_T(v)\big),$$
summed analytically over the vocabulary at each on-policy token. Weighted by the student's own probability, it is zero-forcing: the student withdraws mass from anything the teacher dislikes and concentrates on the modes the teacher genuinely favors, and under capacity mismatch picking one teacher behavior and executing it cleanly is exactly right — the half-JSD MATH-500 win is direct evidence that the more mode-seeking I am, the better the hard set goes. This is the $\beta=1$, on-policy corner of the same (divergence $\times$ data) family the whole ladder lives in.

The reason this is worth deriving from first principles, rather than just dialing $\beta\to 1$, is that the moment I write reverse KL I owe an account of why it is not hopeless to optimize — the objection that makes people reach for JSD compromises. Reverse KL is $\mathbb{E}_{p_S}[\log p_S - \log p_T]$, an expectation *under the student's own distribution*, so the thing I differentiate sits inside an expectation over the distribution whose parameters I am changing. At the sequence level that is an RL problem: the student is a policy sampling trajectories $y \sim p_S(\cdot\mid x)$ and I minimize the sequence reverse KL — which, notice, *automatically* re-derives the on-policy data the trainer already gives me, the expectation being over the student's own rollouts, so reverse KL and the on-policy loop are the same idea seen from two sides. The naive estimator is the policy gradient, and it is a mess for nameable reasons: the reward-to-go $R_t = \sum_{t'\ge t}(\log p_T - \log p_S)$ is a high-variance sum over the sampled future; a small student will *reward-hack* it (degenerate repeated phrases score high teacher probability locally while being garbage reasoning); and there is a length bias in $R_t$ that pushes toward short or empty answers. Running that demands variance baselines, importance weights, length normalization, PPO clipping — a pile of machinery that drags me away from the stable supervised loop the ladder has lived in.

What makes OPD work is to see *where* the mess is, because not all of the gradient is bad. Decompose the per-step gradient into an immediate term and a long-horizon term. The immediate term's inner object is
$$\mathbb{E}_{y_t\sim p_S}\!\Big[\log\tfrac{p_T(y_t)}{p_S(y_t)}\Big] = -\sum_v p_S(v)\log\tfrac{p_S(v)}{p_T(v)} = -\mathrm{KL}\big(p_S(\cdot\mid y_{<t}) \,\|\, p_T(\cdot\mid y_{<t})\big),$$
the per-token reverse KL evaluated *analytically over the whole vocabulary* at the current prefix. I have both full distributions at every position (the trainer hands me both logit tensors), so I compute this expectation exactly by summing over the vocabulary with no sampling noise at all. Every pathology — the variance, the reward-hacking, the length bias — lives in the *long-horizon* term with $R_{t+1}$, the REINFORCE score factor, and the sampled future. The immediate term is clean. And I do not need the long-horizon term: its only job is credit assignment over a *sparse* reward, letting token $t$ be rewarded for good future tokens it enabled — but my reward is *not* sparse. The teacher gives a dense, informative signal at every single token, telling me now, at token $t$, how good the student's choice was by how much probability it assigns. The entire reason to tolerate long-horizon variance is sparse reward, which I do not have. So I drop the long-horizon term — set its weight to zero, a discount factor of zero — and keep only the exact immediate conditional-KL gradient on the states the student actually visits. What remains is structurally a *supervised* loss: a differentiable function of the two logit tensors at matching positions, that happens to be evaluated on student-generated prefixes the trainer's $\lambda$ already supplies. Every stabilizer the policy gradient needed existed to tame the long-horizon term; with it gone, I need none of them.

The KL direction is the one trap, and getting it backwards would silently train forward KL — the very mass-covering objective the ladder has been climbing away from. The framework's `kl_div(input=log_q, target=log_p, log_target=True)` computes $\mathrm{KL}(p \,\|\, q)$, summing $\sum_v p_{\text{target}}(\log p_{\text{target}} - \text{input})$. I want $\mathrm{KL}(p_S \,\|\, p_T) = \sum_v p_S(\log p_S - \log p_T)$, so the *student* log-probs are the target and the *teacher* log-probs are the input — the call is `kl_div(input=teacher_log_probs, target=student_log_probs, log_target=True)`. I divide both logit tensors by the shared temperature first (symmetric, so the KL measures behavior rather than a sharpness mismatch), mask to completion tokens, and reduce per token with `batchmean` — the length-scale-free per-token mean that makes the discount-zero design's killing of the reward-to-go length bias explicit.

One note on scope. The full method is sometimes written with the policy-gradient / sampled-advantage realization (the $(r_t - 1)\cdot\nabla\log p_S$ score-function form) and pins $\lambda = 1.0$ (always on-policy) as part of the recipe. This task's body is the *analytic vocabulary-sum* reverse KL — the preferred form when the teacher's full logits are available, which they are — and it does *not* change $\lambda$: it relies on the trainer's default $\lambda = 0.5$ mixing and applies reverse KL regardless of whether a batch came from the student or the dataset. What makes this OPD is the reverse-KL formulation, not a change to the on-policy fraction; the only thing this rung changes versus GKD is the divergence, from symmetric JSD to the pure reverse-KL endpoint.

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,
    total_steps: int = 0,
    lmbda: float = 0.5,
) -> torch.Tensor:
    # OPD (Lu et al. 2025 / Qwen3 report) — per-token reverse KL: KL(p_S || p_T).
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_S || p_T) per token, summed over vocab.
    # F.kl_div(input=log_q, target=log_p, log_target=True) = Σ_v p · (log p - log q).
    # With input=teacher_log_probs and target=student_log_probs this gives KL(p_S || p_T).
    per_token = F.kl_div(
        teacher_log_probs, student_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)  # [B, T]

    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(per_token.numel(), 1), device=per_token.device, dtype=per_token.dtype)

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```
