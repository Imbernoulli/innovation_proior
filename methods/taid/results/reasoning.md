Let me start from what actually goes wrong when I distill a big language-model teacher into a small student. I have a teacher `p` and a student `q_theta`, both autoregressive, both softmax over the vocabulary at every position, and I train the student to match the teacher's per-token distribution: minimize some divergence `J(p, q_theta)` averaged over the `S` positions of a sequence. The textbook choice is Hinton's forward KL, `J_KL(p, q) = (1/S) sum_s sum_y p(y_s|y^{<s}) log( p / q )`. The pitch is "dark knowledge": the teacher's soft probabilities tell the student not just the right token but the relative plausibility of the wrong ones, which is more signal than a one-hot label. Fine. But forward KL is mass-covering — the `p log(p/q)` term blows up wherever `p` has mass and `q` doesn't, so to keep the loss finite the student must put *some* mass everywhere the teacher does. When the student is much smaller than the teacher and simply can't represent the teacher's full distribution, it does the only thing it can: it spreads its mass thin to cover all the teacher's modes, and I get an oversmoothed, mushy student that's confident about nothing. Mode averaging.

So flip the divergence. Wen and others switch to reverse KL, `J_RKL(p, q) = J_KL(q, p) = sum_y q log(q/p)`. Now the penalty is on the student for putting mass where the teacher has little, so it's mode-seeking: the student stays where the teacher is confident and ignores the rest. That cures the oversmoothing — but for a low-capacity student it overshoots into the opposite ditch. The student concentrates everything on a couple of the teacher's dominant modes and drops the entire tail of the teacher's structure. Mode collapse. And total variation, the other f-divergence on offer, sits somewhere in between but doesn't really resolve the tension — it's still a single fixed yardstick against a fixed teacher.

Two pathologies, one at each end of an axis. The obvious move is to interpolate, and that's what's been tried. Generalized JSD mixes the two KL directions through a probability-space blend `r = beta p + (1-beta) q` and weights `beta KL(p||r) + (1-beta) KL(q||r)`, which slides from forward-KL behavior at `beta -> 0` to reverse-KL behavior at `beta -> 1`. Skew-KL does something adjacent: `KL(p, alpha p + (1-alpha) q)`, blending the student into the comparison distribution so the KL ratio's denominator can't hit zero, which tames the gradient. Both are real improvements. But sit with what they actually fix and what they don't. They let me *dial in* a point between averaging and collapse, with a *constant* coefficient, against a target *anchored at the fixed teacher*. The target never moves. And the thing that's actually killing me isn't the averaging-vs-collapse dial — it's the capacity gap. There's a documented, ugly phenomenon: take a fixed small student, distill it from a 410M teacher, then a 1B, then 2.8B, then 6.9B, and the student gets *worse* as the teacher grows. A bigger, better teacher producing a worse student. That's not a bug in any one divergence; it's that the teacher distribution is simply too far from anything the small student can reach, and a fixed objective against a fixed too-far target is hard to optimize no matter how cleverly I balance the two KL directions. Dialing `beta` doesn't help when the whole target is out of reach from step one and stays equally out of reach at step one million.

Let me say the discomfort precisely, because the precision is the lead. The problem isn't *which* divergence; it's that all of these distill toward a target the student can't get to. The on-policy / student-generated-output line does adapt to the student — it samples from the current student and labels those samples with the teacher — and that's exactly the right instinct, *let the thing the student is being graded against depend on where the student currently is*. But it does it through expensive autoregressive sampling at every step, which at LLM scale is brutal. I want the adaptivity without the sampling. I want the *target itself* to start somewhere the student can reach and migrate toward the teacher as the student improves. A moving target, not a fixed one — and moving in a way that's cheap, from logits alone.

What would "somewhere the student can reach" even mean at step zero? The most reachable target for the student is the student's own current distribution — fitting that is trivial, the loss is essentially already minimized. And the place I eventually want to be is the teacher. So I want a target that is the student at the start and the teacher at the end, and slides continuously between them. Let me write the blend with a time knob `t in [0,1]`: a target distribution `p_t` that is `q_theta` at `t=0` and `p` at `t=1`. As `t` ramps from 0 to 1 over training, the target peels away from the student and walks toward the teacher, and at every moment it's only a little ahead of where the student currently is — close enough to fit, far enough to pull. An intermediate teacher.

Now, *how* do I blend two distributions into `p_t`? The lazy option is a convex combination of probabilities, `t p + (1-t) q`, like JSD's mixture. But I don't love mixing in probability space. A probability mixture of two peaked distributions is bimodal — it literally puts a bump on each — and softmax outputs aren't naturally averaged that way; the mixture can wash out or double the structure I'm trying to hand the student smoothly. The logits are where the additive geometry lives: softmax is `exp(logit)/Z`, so a convex combination *in logit space* turns into a normalized geometric blend of the two distributions, a smooth tempering between them that preserves the *relative* confidences within each. That's the object I want — a target whose internal ranking degrades gracefully from student-like to teacher-like rather than a literal two-humped average. So define the intermediate teacher at the logit level:

  p_t(y_s | y^{<s}) = softmax( (1 - t) * logit_{q'_theta}(y_s | y^{<s}) + t * logit_p(y_s | y^{<s}) ),

`t=0` recovers the student's softmax, `t=1` the teacher's, and in between a logit-interpolated bridge. And the objective is just forward KL against *this* target instead of against the fixed teacher:

  J_TAID^(t)(p, q_theta) = J_KL(p_t, q_theta) = (1/S) sum_s sum_y p_t(y_s|y^{<s}) log( p_t / q_theta ).

One subtlety I have to get right or this is incoherent. `p_t` contains the student's own logits in the `(1-t)` leg, and I'm using `p_t` as the *target* in a KL whose other argument is also `q_theta`. If I let gradients flow through the student copy inside `p_t`, then I'm differentiating the target with respect to the very parameters I'm updating — the student is chasing a target made partly of itself, and the optimization has a degenerate "move the goalposts toward me" channel that does nothing useful. The target must be a target. So I detach the student logits inside `p_t` — treat that `(1-t) logit_{q_theta}` as a constant — and let gradient flow only through the `q_theta` in the denominator of the KL. Write it `q'_theta` to mark the stop-gradient. Then `p_t` is a fixed (for this step) distribution, and the loss honestly pulls the student toward it.

Let me sanity-check the dynamics this produces before I trust it. At small `t`, `p_t` is almost the student's own (detached) distribution, so I'm distilling the student into a slightly-perturbed copy of itself. That's *self-distillation*, essentially — and self-distillation is a known generalization booster; it sparsifies the model, amplifying its dominant structure. So early on, TAID lets the student consolidate its own modes, reinforce what it already does well, rather than immediately drowning under a teacher it can't represent. As `t` grows the target tilts toward the teacher and the student is progressively asked to take on the teacher's richer structure, but always from a target that's only modestly beyond its current reach. Late in training the target is essentially the teacher and I'm doing ordinary forward-KL distillation, but now from a student that's been walked up to the teacher's neighborhood instead of dropped into it cold. That's exactly the medicine for the capacity gap: never confront the small student with the full far-away teacher all at once.

But the self-distillation observation should make me nervous, not comfortable, because self-distillation has a dark side I need to make sure I'm not importing. If early TAID *is* self-distillation, and self-distillation is known to collapse — Mobahi, Farajtabar and Bartlett proved that repeatedly distilling a model into itself drives the solution to zero after enough rounds — then am I building a method that collapses by construction? `p_t` has the student baked into it; the student's modes get fed back and amplified in the fitting recursion; that's the precise recipe for the collapse they describe. I need to check whether the `t p` term — the constant teacher signal mixed in — actually saves me, or whether I've just dressed up self-distillation. Let me work through their analysis with my interpolation in place and find out.

Their tractable proxy is least-square regression in the interpolation regime: find `f` minimizing a kernel regularizer `R(f) = integral u(x,x') f(x) f(x')` subject to fitting the labels to tolerance `eps`, `(1/N) sum_i (f(x_i) - y_i)^2 <= eps`. Why this and not the actual KL/language objective? Because it has a closed-form solution and a clean notion of "collapse," and collapse-to-zero in this regression is the analogue of mode collapse in the categorical case — if I extend it per-class with one-hot teacher signals `y_{i,c}`, a class whose label vector `y_c` shrinks to zero is a mode the model has dropped. So studying when `f = 0` becomes the solution is studying mode collapse. The solution of the constrained problem is the nonlinear-ridge form: stacking predictions over the training inputs, `f = V^T D (lambda I + D)^{-1} V y`, where the kernel matrix `G = V^T D V` has orthogonal `V`, positive diagonal `D = diag(d_i)`, and the multiplier `lambda` is pinned by the constraint to `lambda = alpha sqrt(N eps) / (||y|| - sqrt(N eps))` for some `alpha in [d_min, d_max]`. The collapse criterion is `||y||^2 <= N eps`: if the label signal is weaker than the tolerance allows, the trivial `f = 0` already satisfies the constraint with minimal regularizer, and the solution dies.

Self-distillation in this frame: round `t` fits the previous round's prediction as its label, so with `z = V y` (an orthonormal change of variables that preserves norms), `z_t = D(c_{t-1} I + D)^{-1} z_{t-1}`, where `c_{t-1}` plays the role of `lambda`. The matrix `D(cI+D)^{-1}` is diagonal with every entry `d_k/(c+d_k) in (0,1)`, so it *strictly shrinks* the signal every round. The norm `||z_t||` monotonically decreases, and once it dips below `sqrt(N eps)` the next solution collapses to zero and stays there. They count the guaranteed safe rounds as about `(r_0 - 1)/kappa` where `r_0 = ||y_0|| / sqrt(N eps) > 1` and `kappa = d_max/d_min` is the condition number — and crucially, for enough rounds it *always* collapses, because there's nothing in the recursion replenishing the signal. The target at every round is built only from the model's own shrinking prediction. That's the doom.

Now mine. TAID's intermediate label in this regression proxy is `y~_t = (1 - t/T) y_t + (t/T) y_0`, where `y_0` is the strong teacher signal (constant, not learned) and `y_t` is the current student prediction, with `t` ramping linearly to `T`. The update solves the same variational problem with this interpolated label: `y_{t+1} = V^T D(lambda_t I + D)^{-1} V y~_t`. In rotated coordinates, with `A_t = D(lambda_t I + D)^{-1}` and `z~_t = V y~_t`,

  z~_t = (1 - t/T) A_{t-1} z~_{t-1} + (t/T) z_0.

There it is in one line, and the difference from self-distillation jumps out: the recursion has a *constant inhomogeneous term* `(t/T) z_0`. Self-distillation was the homogeneous recursion `z_t = A_{t-1} z_{t-1}`, pure contraction, signal bleeding away with nothing added back. TAID injects a fixed slice of the teacher signal `z_0` every step, and that slice doesn't shrink because it doesn't depend on the student. So even as the `A_{t-1}` contraction tries to drain the student leg, the teacher leg keeps topping the tank up. Let me see whether that's enough, and exactly when.

Unroll the recursion all the way to `z_0`. Each step multiplies the carried-over student leg by `(1 - (t-tau)/T) A_{t-1-...}` and adds another `(t-tau)/T z_0` term, so telescoping,

  z~_t = [ prod_{tau=0}^{t} (1 - (t-tau)/T) ] [ prod_{tau=0}^{t-1} A_tau ] z_0
        + sum_{tau=1}^{t-1} [ prod_{s=0}^{tau-1} (1 - (t-s)/T) ] (t-tau)/T [ prod_{s=1}^{tau} A_{t-s} ] z_0
        + (t/T) z_0
       =: A_bar_t z_0.

Collecting the scalar products into the factorial coefficients (the `(1 - (t-tau)/T)` are just `(T - (t-tau))/T`, whose products are ratios of factorials), this is a single matrix `A_bar_t` acting on the constant `z_0`. Every `A_tau` is diagonal with entries in `(0,1)` — that follows from `(A_tau)_k = d_k/(lambda_tau + d_k) = ( (alpha_tau/d_k)/(||z~_tau||/sqrt(N eps) - 1) + 1 )^{-1}`, which is squeezed into `[0, 1]` by bounding `alpha_tau/d_k` between `1/kappa` and `kappa` (and is provable by induction over `tau`). So `A_bar_t` is a sum of three diagonal pieces: the all-products term, the inhomogeneous sum, and the bare `(t/T) I` from the last-injected teacher slice — and all three diagonals have nonnegative entries.

The useful piece is the explicit `(t/T) I` term. The smallest singular value of a sum of diagonal PSD matrices is at least the smallest singular value of any one of them, and the cleanest one is `(t/T) I`. So

  sigma_min(A_bar_t) >= sigma_min( (t/T) I ) = t/T,

which means

  ||z~_t|| >= sigma_min(A_bar_t) ||z_0|| >= (t/T) ||z_0|| = (t/T) ||y~_0||,

using `z_0 = z~_0` at the start. The teacher-signal floor is `(t/T) ||z_0||`, and it *grows* with `t` rather than shrinking. The non-collapse criterion `||z~_t|| > sqrt(N eps)` is therefore guaranteed the moment `(t/T) ||y_0|| > sqrt(N eps)`, i.e.

  t > (sqrt(N eps) / ||y_0||) T = T / r_0.

So in the *late* phase, once `t` is past `T/r_0`, TAID provably cannot collapse — precisely the regime where self-distillation is doomed, because there the homogeneous contraction has long since drained the student. The constant `(t/T) z_0` injection is exactly what self-distillation lacks, and it's the late-phase savior. That's the qualitative win stated quantitatively.

I shouldn't stop there, because I also need the *early* phase to be safe, and there the floor `(t/T)||z_0||` is small (t near 0), so that argument is vacuous and I need a different one. Suppose `t` is small enough that `t <= (gamma/r_0) T` for some `gamma in (0,1)`. Lower-bound `r_t := ||z~_t|| / sqrt(N eps)` directly from the one-step recursion. By the reverse triangle inequality on `z~_t = (1 - t/T) A_{t-1} z~_{t-1} + (t/T) z~_0`,

  r_t >= (1 - t/T) || A_{t-1} z~_{t-1} / sqrt(N eps) || - (t/T) || z~_0 / sqrt(N eps) ||
       >= (1 - t/T) sigma_min(A_{t-1}) r_{t-1} - (t/T) r_0.

In the small-`t` regime `t/T <= gamma/r_0`, so `(1 - t/T) >= (1 - gamma/r_0)` and `(t/T) r_0 <= gamma`, giving

  r_t >= (1 - gamma/r_0) sigma_min(A_{t-1}) r_{t-1} - gamma.

Now `sigma_min(A_{t-1})` is the smallest diagonal entry, `>= ( kappa/(r_{t-1} - 1) + 1 )^{-1} = (r_{t-1} - 1)/(r_{t-1} - 1 + kappa)`, which is a convex function of `r_{t-1}`, and Mobahi's linear lower bound `(r-1)/(r-1+kappa) >= beta_0 r - beta_1` (a tangent-style under-estimate) with

  beta_0 = ( (r_0-1)^2 + kappa(2 r_0 - 1) ) / (r_0 - 1 + kappa)^2,
  beta_1 = r_0^2 kappa / (r_0 - 1 + kappa)^2,

turns the recursion into an affine one: `r_t >= (1 - gamma/r_0)(beta_0 r_{t-1} - beta_1) - gamma`. Write `beta_bar_0 = (1 - gamma/r_0) beta_0` and `beta_bar_1 = (1 - gamma/r_0) beta_1`; unrolling the affine recursion geometrically,

  r_t >= beta_bar_0^t r_0 - beta_bar_1 (beta_bar_0^t - 1)/(beta_bar_0 - 1) - gamma =: r_t_lower.

To find when this lower bound reaches the collapse threshold, set `r_t_lower = 1` and solve for the critical `t`:

  t = log( ( (1+gamma)(1 - beta_bar_0) + beta_bar_1 ) / ( beta_bar_1 + r_0 (1 - beta_bar_0) ) ) / log( beta_bar_0 ).

This is exact but opaque; squeeze it with `1 - 1/x <= log x <= x - 1` on the relevant numerator and denominator logs and grind the algebra (substituting the `beta_bar` definitions, which are rational in `r_0`, `kappa`, `gamma`), then take the large-`r_0` asymptotics. The critical time is lower-bounded by

  t >= ( (gamma + o(1)) / (gamma + kappa + o(1)) ) / ( (gamma + o(1)) / ((r_0 - gamma)(1 + o(1))) )
     = (1/(gamma + kappa)) (r_0 - gamma) + o(1).

So in the small-`t` regime, every step before `(1/(gamma+kappa))(r_0 - gamma) + o(1)` is safe. Combine this with the assumption `t <= (gamma/r_0)T` for the early argument and with the late teacher-floor argument: TAID does not collapse for any `t` satisfying

  t < min{ (1/(gamma+kappa))(r_0 - gamma) + o(1),  (gamma/r_0) T }   or   t > T/r_0,

for some `gamma in [0,1]`. The early window plus the late guarantee, with a gap in between governed by `gamma`. Setting `gamma = 1` collapses this to a clean sufficient condition: solving `(1/(1+kappa))(r_0 - 1) + o(1) >= T/r_0` (so the two windows meet and cover every `t`) is a quadratic in `r_0`, and it holds when the initial teacher signal satisfies

  ||y_0|| = Omega( ( (1 + sqrt(1 + 4 T (1 + kappa))) / 2 ) sqrt(N eps) ).

In words: if the teacher's signal is at least on the order of `sqrt(T eps)` strong, the student never collapses, at any step. Compare self-distillation, which Mobahi show survives only for `t <= (r_0 - 1)/kappa` and then inevitably dies. TAID covers the entire late phase that self-distillation can't, at the cost of a constant-factor weaker early window — its critical step scales as `r_0/(gamma+kappa)` versus self-distillation's `r_0/kappa`, so `gamma` has to be bounded away from zero (otherwise the early window vanishes and the student learns nothing meaningful early on). That's the trade the intermediate teacher buys: a little less freedom early, total stability late. Exactly the bias I want, because the early phase is where the student is consolidating its own modes anyway and the late phase is where a fixed-teacher method would have either oversmoothed or collapsed.

Good — the constant teacher injection is what makes the recursion safe, and the proof is written for the simple schedule where the interpolation fraction is the training fraction. So the simplest schedule, `t_linear = t_start + (t_end - t_start) * n/N`, is the baseline I can trust under the non-collapse condition. But I can do better on *efficiency* of learning. The whole point of an intermediate teacher is that the target should stay just ahead of the student. A fixed linear ramp ignores how fast the student is actually learning. Early on, when the target `p_t` is close to the student, fitting is easy and the loss drops fast — I'm wasting steps if I crawl `t` upward linearly when I could be advancing the target quickly. Later, when the target is teacher-like and genuinely hard, I should slow down and let the student fit carefully. So I want `t` to move *fast when the loss is dropping fast and slow when it's dropping slowly*. That's an adaptive schedule keyed to the student's progress.

How do I measure "the loss is dropping fast" in a scale-free way? Absolute loss decrements are no good — the loss magnitude drifts as `t` changes and across runs. The *relative* change is the right signal: `delta_n = (J^(t_{n-1}) - J^(t_n)) / (J^(t_{n-1}) + eps)`, the fractional drop in the TAID objective from the previous step, with a small `eps` to avoid dividing by zero. Positive and large when the student is rapidly closing the gap; near zero or negative when it's stalled. But `delta_n` computed step-to-step is jittery — minibatch noise makes a single step's loss change a poor estimate of the trend. So smooth it with a momentum EMA, `m_n = beta m_{n-1} + (1 - beta) delta_n`, with `beta` near 1 (I'll want ~0.99) so the schedule responds to the sustained trend, not to one noisy batch. This is the same first-moment-EMA logic momentum uses on gradients, applied here to the relative-progress signal.

Now turn the smoothed progress into a step for `t`. I want the increment to be: bounded (never lurch `t` forward uncontrollably), increasing in `m_n` (faster progress -> bigger step), and naturally diminishing as `t` approaches 1 (don't overshoot the teacher). A sigmoid handles the first two — `sigmoid(m_n) in (0,1)`, monotone in `m_n`, saturating so a huge `delta` can't produce an unbounded jump. The `(1 - t_n)` factor handles the third — the remaining distance to `t_end = 1` — so the step shrinks to zero as `t` saturates. With a small step-size `alpha` setting the overall pace,

  Delta t = alpha * sigmoid(m_n) * (1 - t_n).

And I have to keep `t` monotonically increasing and within range — the whole construction relies on the target marching from student toward teacher, never backsliding. So floor each update by the linear schedule, the proof baseline under the signal condition, and cap it at the teacher:

  t_{n+1} = min( t_end, max( t_linear, t_n + Delta t ) ).

The `max(t_linear, ...)` guarantees at least linear progress, so the schedule cannot stall in the self-distillation-like region, while the adaptive term lets `t` race ahead when learning is easy. Under the gamma-1 signal condition above, the early and late windows cover the whole training range, so these monotone advances stay inside the non-collapse regime. The `min(t_end, ...)` clamps at the teacher. The hyperparameters fall out of their roles: `beta = 0.99` to smooth the noisy relative-progress signal over a long window; `alpha = 5e-4`, small, because `t` is a delicate global knob and I want stable, gradual adaptation rather than abrupt jumps; `t_start` between 0.2 and 0.4 to skip the trivially-easy earliest phase where the target is almost exactly the student and there's nothing to learn — set it by the intuitive initial gap, smaller when student and teacher start close, around 0.4 when the gap is large; `t_end = 1.0` so the final target is the genuine teacher.

Let me write the full procedure to make sure it composes. Initialize `t_1 = t_start`, momentum `m_0 = 0`, and `J^(t_0) = infinity` (so the very first relative-change is degenerate and skipped). Each iteration: compute the linear floor `t_linear = t_start + (t_end - t_start) * n/N`; sample a batch; build `p_{t_n}` by logit-interpolating the detached student and the teacher and softmaxing; compute the forward-KL loss `J^(t_n) = J_KL(p_{t_n}, q_theta)`; backprop and step the student `theta <- theta - eta grad J`; then compute `delta_n`, update `m_n`, form `Delta t = alpha sigmoid(m_n)(1 - t_n)`, and set `t_{n+1} = min(t_end, max(t_linear, t_n + Delta t))`. One forward-KL loss per step, the same teacher forward pass ordinary distillation already needs, and a handful of scalars for the schedule — no student sampling, no second frozen network, logit-only. Cheap.

One implementation reality I should fold in. The forward KL of the intermediate teacher against the student, `KL(p_t || q_theta) = sum_y p_t (log p_t - log q_theta)`, has a first term `sum_y p_t log p_t` that is the (negative) entropy of `p_t` — and `p_t` is *detached*, a constant for this step. So as far as the student's gradient is concerned that term is a constant and can be dropped; minimizing `KL(p_t || q_theta)` is gradient-equivalent to minimizing the cross-entropy `- sum_y p_t log q_theta`. The clean way to implement the loss is therefore `- sum_v p_t(v) * log_softmax(student_logits)(v)`, summed over the vocabulary, masked to the completion tokens, averaged over valid positions. I should also guard the masked-out / `-inf` logit positions so they contribute zero rather than `nan`. Either the cross-entropy form or the full KL via a `kl_div` call gives the same student gradient; I'll write the cross-entropy form since it's the natural one and matches how the loss is actually computed.

So the whole reasoning lands here, in the loss-and-schedule module:

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
    # The buffers are the scalar state of the adaptive interpolation parameter.

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
        progress = global_step / num_train_steps
        t_target = self.t_start + (self.t_end - self.t_start) * progress
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (
            min(self.t_end, max(t_target, self.t + delta_t))
            if not self.disable_adaptive
            else t_target
        )
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(self, logits, teacher_logits, mask):
        # Intermediate teacher in logit space; the student leg is detached so
        # the target is fixed while the denominator student receives gradients.
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

The causal chain, start to finish. Forward-KL distillation oversmooths a small student under a capacity gap because it's mass-covering (mode averaging); reverse KL fixes that but is mode-seeking, so the student collapses onto a few modes (mode collapse); JSD and skew-KL interpolate between the two but against a *fixed* target anchored at the teacher, which under a large capacity gap stays out of reach for the whole run, so the student gets *worse* as the teacher grows. The fix is to make the target itself move: an intermediate teacher `p_t` that is the student's own (detached) distribution at `t=0` and the teacher at `t=1`, interpolated in logit space to preserve relative confidence, with forward KL against this moving target. Early on it acts like self-distillation, consolidating the student's modes; but self-distillation alone provably collapses, since its recursion only contracts the signal — so I checked, in the regression proxy, whether the teacher term saves me, and it does: the interpolated update `z~_t = (1 - t/T) A_{t-1} z~_{t-1} + (t/T) z_0` carries a constant teacher-signal injection that floors the norm at `(t/T)||z_0||`, guaranteeing non-collapse late (`t > T/r_0`) and, via an affine-recursion argument, early (`t < min{(r_0 - gamma)/(gamma+kappa), gamma T/r_0}` up to the asymptotic term), so with a teacher signal of order `sqrt(T eps)` the student never collapses — unlike self-distillation, which inevitably does. A linear `t` schedule is the proof baseline; the adaptive one keeps that lower-bound progression while keying `t`'s speed to the student's smoothed relative loss progress through a sigmoid-bounded, `(1-t)`-diminishing, linearly-floored, teacher-capped update. The whole thing is one forward-KL loss plus a scalar schedule update per step — logit-only, no student sampling, no extra frozen network.
