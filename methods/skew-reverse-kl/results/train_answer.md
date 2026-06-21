The problem is to compress a large autoregressive language teacher $p(y\,|\,x)$ into a much smaller student $q_\theta(y\,|\,x)$ while keeping the teacher's task ability, framing distillation as minimizing a per-token divergence $D(p, q_\theta)$ averaged over a sequence. The student is capacity-limited and cannot represent the teacher's full distribution, so the choice of divergence direction — and where the training sequences come from — decides whether the student ends up smeared, collapsed, or genuinely good. Two prior choices each fail in their own way. Forward KL $D_{KL}(p, q_\theta) = \sum_v p(v)\log\!\big(p(v)/q_\theta(v)\big)$, weighted by the teacher, is mass-covering: it forces an under-capacity student to spread mass over the teacher's whole support, so it mode-averages into an over-smooth, hedging, incoherent model. Reverse KL $D_{KL}(q_\theta, p)$, weighted by the student, is mode-seeking and cures that smearing by committing the student to the teacher's dominant modes — so the configuration I actually want is reverse KL trained on the student's own generations, since training on a fixed corpus is off-policy and the train-inference prefix mismatch compounds errors over the sequence, whereas student-generated outputs (SGOs) train on the student's familiar states. The trouble is that this exact, well-motivated combination trains badly, and I want to know precisely why before reaching for a fix, because the failure should dictate the remedy.

The gradient is what the optimizer feels, so I look there. For forward KL, differentiating $-\sum p\log q_\theta$ and collapsing onto the sampled sequence gives $\nabla_\theta D_{KL}(p, q_\theta) = -\,r_{p,q_\theta}\,\nabla_\theta q_\theta(y\,|\,x)$, the model-probability gradient weighted inversely by that probability through the ratio $r_{p,q_\theta} = p/q_\theta$; where $q_\theta(y\,|\,x)\approx 0$ — the student starves a token the teacher likes — the ratio explodes and the optimizer takes a huge, noisy step. The reverse direction is the mirror image. The minimized scalar loss (the token-loss code accumulates the negative integrand $\sum q_\theta(\log p - \log q_\theta)$ and negates at reduction) has $\nabla_\theta D_{KL}(q_\theta, p) = (\log r_{q_\theta,p} + 1)\,\nabla_\theta q_\theta(y\,|\,x)$ with $r_{q_\theta,p} = q_\theta/p$, and that coefficient blows up at the other end, where $p(y\,|\,x)\approx 0$ so $\log(q_\theta/p)\to+\infty$. Here is the wall: a student-generated sequence is not drawn from the teacher, so on SGOs $p(y\,|\,x)\approx 0$ is not a remote edge case but a frequent, even dominant, event — the very data that fixes the off-policy mismatch detonates the reverse-KL gradient. Every self-generated token the teacher finds surprising throws a giant, noisy gradient, which is why people bolt variance baselines, teacher-mixed sampling, importance weights, and clipping onto reverse-KL policy gradients. I want to kill the explosion at its source rather than patch it downstream.

The source is structural: in both directions a raw distribution sits in a denominator and goes to zero — $q_\theta$ in $r_{p,q_\theta}=p/q_\theta$, $p$ in $r_{q_\theta,p}=q_\theta/p$. So I floor that denominator by computing the KL not against the raw target but against a mixture of the two distributions, which cannot vanish unless both vanish together. I propose Skew (Reverse) KL — SKL and its reverse-direction form SRKL — the distillation loss that replaces the raw target with an $\alpha$-skewed mixture. For the direction I care about, mix the teacher with a sliver of the student, $\tilde p = (1-\alpha)\,p + \alpha\,q_\theta$, and minimize
$$D_{\mathrm{SRKL}}^{\alpha}(p, q_\theta) = D_{KL}\!\big(q_\theta,\ (1-\alpha)\,p + \alpha\,q_\theta\big).$$
At $\alpha = 0$ this is exactly reverse KL; as $\alpha$ grows the denominator floor rises. Carrying the $\theta$-dependence through both the outer $q_\theta$ weight and the $\alpha\,q_\theta$ leg inside $\tilde p$, the minimized-loss gradient becomes
$$\nabla_\theta D_{\mathrm{SRKL}}^{\alpha}(p, q_\theta) = \big(\log r_{q_\theta,\tilde p} + 1 - \alpha\, r_{q_\theta,\tilde p}\big)\,\nabla_\theta q_\theta(y\,|\,x),\qquad r_{q_\theta,\tilde p} = q_\theta/\tilde p.$$
Two things change against plain reverse KL's $\log r_{q_\theta,p} + 1$, and both help. The ratio inside the log is now $q_\theta/\tilde p$, and on an SGO where $p\approx 0$ the mixture still carries the $\alpha\,q_\theta$ leg, so $\tilde p \ge \alpha\,q_\theta$ and $r_{q_\theta,\tilde p}\le 1/\alpha$ — the log can no longer diverge. And the new $-\alpha\,r_{q_\theta,\tilde p}$ term subtracts from the positive coefficient and grows as the ratio grows, pulling the coefficient back down exactly where it would otherwise run away. For completeness the forward direction skews symmetrically: $D_{\mathrm{SKL}}^{\alpha}(p, q_\theta) = D_{KL}(p,\ \alpha\,p + (1-\alpha)\,q_\theta)$, with $\nabla_\theta D_{\mathrm{SKL}}^{\alpha}(p, q_\theta) = -(1-\alpha)\,r_{p,\tilde q_\theta}\,\nabla_\theta q_\theta$ where $\tilde q_\theta = \alpha\,p + (1-\alpha)\,q_\theta$, so $\tilde q_\theta\ge\alpha\,p$ floors the student leg and $r_{p,\tilde q_\theta}\le 1/\alpha$. The asymmetry I keep is which raw distribution I protect: skew-forward floors the student leg (mass-covering, for when the student can vanish on teacher-favored tokens), skew-reverse floors the teacher leg (mode-seeking, for when the teacher can vanish on SGOs). Since my configuration is mode-seeking on SGOs, SRKL is the one whose denominator floor sits exactly where my data puts the zeros.

The remaining question is how much to skew. Skewing also helps estimation: the empirical L2 error of the SKL estimator obeys a bound of the form $c_1(\alpha)/n^2 + c_2\log^2(\alpha n)/n + c_3\log^2(c_4 n)/(\alpha^2 n)$ with $c_1(\alpha) = \min\{1/\alpha^2,\ \chi^2(p,q)^2/(1-\alpha)^2\}$ (for $\alpha < 1/8$, $\chi^2$ the chi-square divergence), so moving away from $\alpha = 0$ shrinks the inverse-$\alpha$ contribution that makes the raw-KL limit a noisy mini-batch objective. But that does not mean push $\alpha$ toward $1$, because the gradient-norm benefit is partly illusory: an Adam-style optimizer normalizes by a running estimate of the gradient scale, so a uniformly smaller coefficient is divided back out. To read the error in the units the optimizer actually moves in, I rescale the deviation by the inverse gradient scale $1/(1-\alpha)$ and re-bound, which adds inverse-$(1-\alpha)$ terms; the inverse-$\alpha$ pieces want $\alpha$ large, the inverse-$(1-\alpha)$ pieces want it small, and the normalized curve is convex over the useful range. The normalized SRKL check singles out its smallest variance at $\alpha = 0.1$ and worsens beyond, so I take that mild point: $\tilde p = 0.9\,p + 0.1\,q_\theta$ — enough to floor the denominator while leaving the reverse target $90\%$ teacher. This is also the analytical reason to prefer one skewed KL over generalized JSD, which is $D_{\mathrm{JSD}}^{\beta}(p, q_\theta) = \beta\,D_{\mathrm{SKL}}^{\beta}(p, q_\theta) + (1-\beta)\,D_{\mathrm{SRKL}}^{1-\beta}(p, q_\theta)$: a sum of two skewed KLs whose skew parameters are tied to the same $\beta$, so $\alpha = 0.1$ on the reverse leg structurally forces $0.9$ on the forward leg. A single skewed reverse KL with a freely chosen mild $\alpha$ reaches an operating point JSD's coupled dial cannot. I keep the loss separate from the data-side machinery — an adaptive SGO scheduler that ramps self-generated data up under validation loss, and an off-policy replay buffer with a decaying replay ratio to cut generation cost — which is orthogonal scheduling, though it rides on the skewed loss's fast early convergence to avoid stale-policy bias.

In code the one subtlety is the difference between a literal divergence value and the optimized training loss. For reverse KL both legs of $\sum_v q_\theta(v)\big(\log q_\theta(v) - \log\tilde p(v)\big)$ depend on $\theta$, so I keep both — discarding $\sum q_\theta\log q_\theta$ would drop the $+1$ normalization gradient I derived. I form the mixture in probability space, take its log, accumulate $q_\theta\log\tilde p$ minus $q_\theta\log q_\theta$, mask any $\pm\inf$ logit positions to zero so they contribute nothing rather than $\mathrm{nan}$, sum over the vocabulary to a per-token value, mask to the completion tokens ($\text{label}\neq -100$), and average; the leading sign is negative because the accumulator holds $\sum q_\theta(\log\tilde p - \log q_\theta)$ and is negated at reduction. The forward variant is the same skeleton with the mixture floored on the student leg, $\tilde q_\theta = \alpha\,p + (1-\alpha)\,q_\theta$, and the teacher as outer weight, $\sum p\log\tilde q_\theta$; there the target entropy $\sum p\log p$ is constant in $\theta$ and dropped for training, to be added back only if the numeric divergence is reported.

```python
import torch
import torch.nn.functional as F


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # SRKL: KL(q_theta, (1-lam)*p + lam*q_theta). lam = alpha = 0.1. Mode-seeking,
    # with q_theta added to the teacher-side denominator for stability on SGOs.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs        # p~ = (1-a)p + a q

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # SKL: KL(p, lam*p + (1-lam)*q_theta). Mass-covering; p is added to the
    # student-side denominator for stability where q_theta is near zero. This
    # canonical training loss drops the theta-constant target entropy Sum p log p.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs        # q~ = a p + (1-a) q
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
```
