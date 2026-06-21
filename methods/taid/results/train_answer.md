We want to compress a large, well-trained autoregressive language-model teacher $p$ into a much smaller student $q_\theta$ that we can actually deploy, and the lever we pull is the distillation objective: a per-token divergence $J(p, q_\theta)$ between the teacher and student conditional distributions, averaged over the $S$ positions of a sequence. The whole quality of the distilled student is decided by this choice, and the choice is hard precisely because the two models are very different. Three coupled failures lurk. The capacity gap means the student provably cannot represent every distribution the teacher can. Mode averaging is what happens when an objective forces the student to cover every region of teacher mass it cannot faithfully reach: it smears its probability thinly across all of them and ends up oversmoothed and confident about nothing. Mode collapse is the opposite ditch: an objective that only rewards staying where the teacher is confident drives a low-capacity student to pile all its mass on a few dominant modes and drop the rest of the teacher's structure.

The standard objectives each sit at one end of this axis. Hinton's forward KL $J_{KL}(p, q_\theta) = \frac{1}{S}\sum_s \sum_y p\log(p/q_\theta)$ transfers the teacher's "dark knowledge," the relative weights over wrong answers, but it is mass-covering — the $p\log(p/q)$ term explodes wherever $p$ has mass and $q$ does not — so under a capacity gap the student oversmooths to cover all modes: mode averaging. Flipping to reverse KL $J_{KL}(q_\theta, p)$ makes the objective mode-seeking and cures the oversmoothing, but it overshoots: the low-capacity student concentrates on a couple of dominant teacher modes and drops the tail: mode collapse. Total variation sits somewhere between but does not resolve the tension. The natural next move, interpolating the two KL directions, has been tried — generalized JSD weights $\beta\,KL(p\|r) + (1-\beta)KL(q_\theta\|r)$ through a probability-space mixture $r = \beta p + (1-\beta)q_\theta$, and skew-KL uses $KL(p,\,\alpha p + (1-\alpha)q_\theta)$ to keep the KL denominator away from zero and tame the gradient. Both are real improvements, but both dial in a fixed point between averaging and collapse with a constant coefficient, against a target anchored at the fixed teacher. The target never moves. And the thing actually doing the damage is not the averaging-versus-collapse dial; it is the capacity gap itself. There is a documented, ugly phenomenon — hold a small student fixed, distill it from a 410M, then 1B, then 2.8B, then 6.9B teacher, and the student gets worse as the teacher grows — because a fixed objective against a fixed, far-away target is hard to optimize no matter how cleverly the two KL directions are balanced. The on-policy line, sampling from the current student and labeling those samples with the teacher, has the right instinct of letting the target depend on where the student currently is, but pays for it with expensive autoregressive sampling at every step. We want that adaptivity without the sampling, and from logits alone.

I propose TAID, Temporally Adaptive Interpolated Distillation. The idea is to make the target itself move: instead of distilling toward the fixed teacher, distill toward a time-dependent intermediate teacher $p_t$ that begins as the student's own distribution and migrates to the teacher's as a scalar $t$ ramps from $0$ to $1$ over training, so the target stays only modestly ahead of the current student at every moment — close enough to fit, far enough to pull. The most reachable target at step zero is the student's own current distribution (fitting it is essentially free), and the place we ultimately want to be is the teacher; an intermediate teacher interpolates between them. The blend is done in logit space, not probability space, and this is a deliberate choice: a convex combination of two peaked probability distributions is bimodal — it literally puts a bump on each and can wash out or double the structure I am trying to hand the student — whereas a convex combination in logit space, because softmax is $\exp(\text{logit})/Z$, becomes a normalized geometric blend, a smooth tempering that preserves the relative confidences within each distribution. So the intermediate teacher is

$$p_t(y_s \mid y^{<s}) = \mathrm{softmax}\!\big((1-t)\,\mathrm{logit}_{q'_\theta}(y_s\mid y^{<s}) + t\,\mathrm{logit}_p(y_s\mid y^{<s})\big),$$

with $t=0$ recovering the student's softmax, $t=1$ the teacher's, and a logit-interpolated bridge in between. The objective is forward KL against this moving target,

$$J_{TAID}^{(t)}(p, q_\theta) = J_{KL}(p_t, q_\theta) = \frac{1}{S}\sum_s \sum_y p_t(y_s\mid y^{<s})\,\log\frac{p_t}{q_\theta}.$$

One subtlety is load-bearing: the $(1-t)$ leg of $p_t$ contains the student's own logits, written $q'_\theta$ to mark a stop-gradient. If gradients were allowed to flow through that copy, the student would be differentiating its target with respect to the very parameters being updated — a degenerate "move the goalposts toward me" channel. So the student logits inside $p_t$ are detached; $p_t$ is a constant for the step, a genuine target, and gradient flows only through the $q_\theta$ in the KL's denominator. Because $p_t$ is detached, its entropy term $\sum_y p_t\log p_t$ is constant in $\theta$, so minimizing the KL is gradient-equivalent to minimizing the cross-entropy $-\sum_y p_t\log q_\theta$ — the form actually implemented, masked to completion tokens and guarded so masked-out or $-\infty$ logits contribute zero rather than NaN.

The dynamics this produces are exactly the medicine for the capacity gap. At small $t$, $p_t$ is almost the student's own detached distribution, so TAID is essentially self-distillation: the student consolidates its own modes and reinforces what it already does well rather than drowning under a teacher it cannot represent. As $t$ grows, the target tilts toward the teacher and the student is progressively asked to take on richer structure, but always from a target only modestly beyond its current reach. Late in training the target is essentially the teacher and TAID becomes ordinary forward-KL distillation, but now from a student that has been walked up to the teacher's neighborhood rather than dropped into it cold.

The self-distillation observation is also a warning, because self-distillation provably collapses: Mobahi et al. show that repeatedly distilling a model into itself drives the solution to zero after enough rounds, and $p_t$ has the student baked into it. I had to verify the constant teacher term saves me rather than dressing up a doomed recursion. In their least-square-regression proxy (interpolation regime, kernel regularizer, nonlinear-ridge solution $f = V^\top D(\lambda I + D)^{-1}V y$, collapse exactly when $\|y\|^2 \le N\varepsilon$), the TAID update with interpolated label $\tilde y_t = (1 - t/T)\,y_t + (t/T)\,y_0$ becomes, in rotated coordinates $z = Vy$ with $A_t = D(\lambda_t I + D)^{-1}$,

$$\tilde z_t = (1 - t/T)\,A_{t-1}\,\tilde z_{t-1} + (t/T)\,z_0.$$

Self-distillation is the homogeneous version $z_t = A_{t-1}z_{t-1}$ — pure contraction, since each $A_t$ is diagonal with entries $d_k/(\lambda_t + d_k)\in(0,1)$, so the signal bleeds away with nothing added back and the norm eventually drops below $\sqrt{N\varepsilon}$ and dies. TAID's recursion carries a constant inhomogeneous term $(t/T)z_0$, a fixed slice of the teacher signal that does not shrink because it does not depend on the student. Unrolling to $z_0$ gives $\tilde z_t = \bar A_t z_0$, where $\bar A_t$ is a sum of diagonal PSD pieces including the bare $(t/T)I$ from the last-injected teacher slice; since the smallest singular value of a sum of diagonal PSD matrices is at least that of any one piece, $\sigma_{\min}(\bar A_t) \ge t/T$, hence

$$\|\tilde z_t\| \ge (t/T)\,\|z_0\| = (t/T)\,\|y_0\|.$$

This floor grows with $t$ instead of shrinking, so the non-collapse criterion $\|\tilde z_t\| > \sqrt{N\varepsilon}$ is guaranteed in the late phase whenever $t > T/r_0$ with $r_0 = \|y_0\|/\sqrt{N\varepsilon}$ — precisely the regime where self-distillation is doomed. The early phase needs a separate argument because that floor is small there; using the reverse triangle inequality on the one-step recursion, $\sigma_{\min}(A_{t-1}) \ge (r_{t-1}-1)/(r_{t-1}-1+\kappa)$ with condition number $\kappa = d_{\max}/d_{\min}$, and Mobahi's linear under-estimate of that convex bound, the recursion becomes affine and unrolls geometrically, giving safety for $t$ below roughly $(r_0 - \gamma)/(\gamma+\kappa)$. Combining the windows, TAID does not collapse for any $t$ with $t < \min\{(r_0-\gamma)/(\gamma+\kappa) + o(1),\ (\gamma/r_0)T\}$ or $t > T/r_0$, for some $\gamma\in[0,1]$. Setting $\gamma=1$ makes the windows meet and cover the whole run, yielding the clean corollary that if the teacher signal satisfies $\|y_0\| = \Omega\big(\frac{1+\sqrt{1+4T(1+\kappa)}}{2}\sqrt{N\varepsilon}\big)$ — order $\sqrt{T\varepsilon}$ strong — the student never collapses at any step. Self-distillation, by contrast, survives only for $t \le (r_0-1)/\kappa$ and then inevitably dies; TAID covers the entire late phase at the price of a constant-factor weaker early window (critical step $\sim r_0/(\gamma+\kappa)$ versus $\sim r_0/\kappa$, so $\gamma$ must stay bounded away from zero). That is exactly the bias I want: a little less freedom in the early phase where the student is consolidating its own modes anyway, total stability in the late phase where a fixed-teacher method would have oversmoothed or collapsed.

The linear ramp $t_{\text{linear}} = t_{\text{start}} + (t_{\text{end}} - t_{\text{start}})\,n/N$ is the schedule the proof is written for, and it is the baseline I can trust under the signal condition. But a fixed linear ramp ignores how fast the student is actually learning, and the whole point of an intermediate teacher is to keep the target just ahead of the student — advance fast when fitting is easy, slow down when the target is teacher-like and genuinely hard. So I make the speed of $t$ adaptive to the student's progress. The right progress signal is scale-free, because absolute loss magnitudes drift as $t$ changes and across runs, so I use the relative drop $\delta_n = (J^{(t_{n-1})} - J^{(t_n)})/(J^{(t_{n-1})} + \varepsilon)$, large when the student is rapidly closing the gap. A single step's $\delta_n$ is jittery from minibatch noise, so I smooth it with a momentum EMA $m_n = \beta m_{n-1} + (1-\beta)\delta_n$ with $\beta$ near $1$, the same first-moment logic momentum uses on gradients applied to the relative-progress signal. The increment for $t$ must be bounded so it never lurches, increasing in $m_n$ so faster progress means bigger steps, and diminishing as $t\to 1$ so it cannot overshoot the teacher: a sigmoid handles the first two ($\mathrm{sigmoid}(m_n)\in(0,1)$, monotone, saturating) and a $(1-t_n)$ factor the third, giving

$$\Delta t = \alpha\,\mathrm{sigmoid}(m_n)\,(1 - t_n),$$

with a small step-size $\alpha$ setting the overall pace. Finally $t$ must stay monotone and in range, so I floor each update by the linear schedule (the proof baseline, which guarantees the schedule cannot stall in the self-distillation-like region) and cap at the teacher:

$$t_{n+1} = \min\!\big(t_{\text{end}},\ \max(t_{\text{linear}},\ t_n + \Delta t)\big).$$

Under the $\gamma=1$ signal condition the early and late safe windows cover the whole training range, so these monotone, at-least-linear advances stay inside the non-collapse regime. The defaults follow from the roles: $\beta = 0.99$ to smooth the noisy relative-progress signal over a long window; $\alpha = 5\times 10^{-4}$, small because $t$ is a delicate global knob and I want gradual adaptation; $t_{\text{start}}\in[0.2,0.4]$ to skip the trivially easy earliest phase where the target is almost exactly the student (around $0.4$ when the capacity gap is large); and $t_{\text{end}} = 1.0$ so the final target is the genuine teacher. The cost per step is one forward-KL loss against $p_t$ — the same teacher forward pass ordinary distillation already needs — plus a handful of scalars for the schedule: no student sampling, no second frozen network, logit-only and cheap.

```python
from typing import Dict, Union
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError


def forward_kl(logits, teacher_logits, mask, teacher_probs=None, student_logprobs=None):
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class TAID(DistilLoss):
    def __init__(self, t_start=0.4, t_end=1.0, alpha=5e-4, beta=0.99, disable_adaptive=False):
        super().__init__()
        assert 0.0 <= t_start < 1.0
        assert 0.0 < t_end <= 1.0
        assert 0.0 <= alpha <= 1.0
        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        self.register_buffer("t", torch.tensor(float(t_start), dtype=torch.float32))
        self.register_buffer("prev_loss", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("momentum", torch.zeros((), dtype=torch.float32))

    def update_t(self, loss, global_step, num_train_steps):
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return

        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        adaptive_delta = torch.sigmoid(self.momentum)
        t_target = self.t_start + (self.t_end - self.t_start) * (global_step / num_train_steps)
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (min(self.t_end, max(t_target, self.t + delta_t))
             if not self.disable_adaptive else t_target)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(self, logits, teacher_logits, mask):
        p_t = (1 - self.t) * logits.detach() + self.t * teacher_logits
        p_t = F.softmax(p_t, dim=-1, dtype=torch.float32)
        return forward_kl(
            logits=logits,
            teacher_logits=teacher_logits,
            mask=mask,
            teacher_probs=p_t,
        )

    def forward(self, lightning_module: LightningModule, logits, teacher_logits, mask, **kwargs):
        loss = self.compute_loss(logits, teacher_logits, mask)
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )
        return {
            "distil_loss": loss,
            "tiki_t": self.t,
            "delta_t": delta_t,
        }
```
