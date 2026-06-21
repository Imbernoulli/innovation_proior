TAID confirmed the split I was watching for, and it points straight at the next move. `taid` reads GSM8K 0.4685, MATH-500 0.280: on GSM8K a real jump, $+0.014$ over RS-KD and the largest single-rung gain so far, exactly what the moving target should buy — the student walked up from a reachable near-self target instead of being asked to match the teacher cold. But MATH-500 went the *wrong* way, 0.280, below both RS-KD (0.292) and dagger (0.290). The reachability fix worked; the loss it reaches the teacher *through* is still forward KL. Three rungs have all used forward KL in some guise — dagger's hard target is its argmax limit, RS-KD's top-$K$ KL, TAID's KL to the intermediate teacher — and the one axis none of them touched is which way the KL points. That is the binding constraint now. Forward KL $\mathrm{KL}(p_T \,\|\, p_S) = \sum_v p_T(v)\log\frac{p_T(v)}{p_S(v)}$ weights each token by the *teacher's* probability, so as $p_S(v)\to 0$ wherever $p_T(v)>0$ the term blows up — it forces the student to put *some* mass everywhere the teacher does. Mass-covering. A 0.5B student cannot represent every mode of a 7.6B math model's next-token distribution, so "cover all the teacher's modes" spreads its limited mass thin, including over the teacher's long tail, and free-run generation becomes a smeared, hedge-everything distribution that samples incoherent continuations. That is precisely the MATH-500 regression: TAID got the student *to* the teacher, but a mass-covering match parks it in the valley between the teacher's modes, where reasoning on a long chain goes to die.

The opposite is reverse KL $\mathrm{KL}(p_S \,\|\, p_T) = \sum_v p_S(v)\log\frac{p_S(v)}{p_T(v)}$, weighted by the *student's* probability, so the penalty is large exactly where the student puts mass the teacher dislikes — zero-forcing, mode-seeking: the student concentrates on the teacher's major modes and abandons the tail. Fitting a single Gaussian to a two-bump mixture, forward KL parks in the valley to cover both, reverse KL snaps onto one bump and commits, and for compression under capacity mismatch committing to one coherent teacher behavior is what the MATH-500 failure says I want. But I do not want to slam all the way to pure reverse KL in one step: it can overshoot into the opposite ditch and collapse onto one or two dominant modes, and raw KL in either direction is *unbounded* when the two distributions have near-disjoint support — which happens constantly early in training — producing enormous destabilizing gradients. I want a one-parameter family with forward KL at one end and reverse KL at the other, sensible in between, and *bounded* even on disjoint supports, so I can sit at a balanced interior point rather than commit to an endpoint.

I propose **GKD** with the **generalized Jensen-Shannon divergence at $\beta = 0.5$**. Introduce $\beta \in (0,1)$, form the mixture $M = \beta\,p_T + (1-\beta)\,p_S$, and define
$$D_{\mathrm{JSD}}^{\beta}(p_T \,\|\, p_S) = \beta\,\mathrm{KL}(p_T \,\|\, M) + (1-\beta)\,\mathrm{KL}(p_S \,\|\, M).$$
It interpolates the two directions, though the boundary needs care, because the raw $D_{\mathrm{JSD}}^{\beta}$ itself goes to zero as $\beta\to 0$ or $\beta\to 1$ — the KLs appear as *scaled* limits, not the literal endpoint values. As $\beta\to 0$, $M\to p_S$, the second term is second-order in $\beta$, and $\lim_{\beta\to 0} D_{\mathrm{JSD}}^{\beta}/\beta = \mathrm{KL}(p_T \,\|\, p_S)$, the forward direction. By the symmetry $D_{\mathrm{JSD}}^{\beta}(p_T \,\|\, p_S) = D_{\mathrm{JSD}}^{1-\beta}(p_S \,\|\, p_T)$ the $\beta\to 1$ end gives reverse KL. So small $\beta$ is mass-covering, large $\beta$ is mode-seeking, and $\beta = 0.5$ is the symmetric Jensen-Shannon divergence sitting exactly between. The second gift comes free: JSD is *bounded* even when $p_T$ and $p_S$ are disjoint, whereas plain KL is infinite there, so the early-training blow-up pure reverse KL would risk is tamed automatically.

This is the move that subsumes the whole ladder, which is what convinces me it is the right generalization rather than a fourth ad-hoc loss. The general object behind every rung is: pick a divergence on the forward$\leftrightarrow$reverse family, and pick how much data is the student's own (the trainer's $\lambda$). Off-policy supervised KD is the forward-KL corner; the scaffold default is forward KL; RS-KD is forward KL over a sparse support; TAID's forward KL to a moving target is a forward-KL-flavored point with a curriculum bolted on. None moved off the forward-KL *face* of the family. Symmetric JSD steps onto the *interior* of the divergence axis for the first time — half mass-covering, half mode-seeking — exactly the medicine the MATH-500 regression prescribed: pull the student off the pure mass-covering match without slamming it into mode-collapse.

The per-token loss has two traps. For the interior $\beta = 0.5$ I need $\log M$ where $M = \beta\,p_T + (1-\beta)\,p_S$; computing it as $\log((1-\beta)\exp(\log p_S) + \beta\exp(\log p_T))$ underflows, so I form it as a log-sum-exp of the two shifted log-prob tensors, $\log M = \mathrm{logsumexp}([\log p_S + \log(1-\beta),\ \log p_T + \log\beta])$ stacked along a new axis, getting $\log M$ without ever leaving log space. Then the direction: the framework's `kl_div(input=log_q, target=log_p, log_target=True)` computes $\mathrm{KL}(p \,\|\, q)$, treating the *input* as the log of the denominator and the *target* as the distribution the KL is from. So $\mathrm{KL}(p_T \,\|\, M)$ is mixture-input, teacher-target, and $\mathrm{KL}(p_S \,\|\, M)$ is mixture-input, student-target; getting it backwards minimizes a quietly wrong objective with no error raised. I combine $\text{per\_token} = \beta\,\mathrm{KL}(p_T \,\|\, M) + (1-\beta)\,\mathrm{KL}(p_S \,\|\, M)$, mask to completion tokens, reduce per token, with temperature dividing both logit tensors before the softmax.

One thing this task's loss does *not* do: the full GKD object exposes $\beta$ as a tunable knob with explicit $\beta=0$ and $\beta=1$ endpoint branches plus the interior JSD, so a trainer can sweep the divergence. I do not expose that sweep — I hard-pin $\beta_{\text{use}} = 0.5$ and always compute the genuine interior generalized JSD via the mixture, ignoring the `beta` argument the signature passes in. So what I land is specifically the *symmetric Jensen-Shannon* point of GKD, not the configurable family and not an endpoint. The data axis stays the trainer's static $\lambda$ mixing, untouched, exactly as on every prior rung; the only thing this rung changes versus TAID is the divergence, from forward KL to symmetric JSD.

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
    # GKD (Agarwal et al., ICLR'24) — Generalized JSD with β = 0.5.
    beta_use = 0.5
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    beta_t = torch.tensor(beta_use, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
        dim=0,
    )
    # PyTorch's F.kl_div(input=log_q, target=log_p) computes KL(p || q).
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    per_token = beta_t * kl_teacher + (1 - beta_t) * kl_student  # [B, T]

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
