Let me start from the split that keeps frustrating me. When I train a CNN, Adam gets the loss down fast in the first few epochs, but the model it lands on generalizes worse than if I grind it out with plain SGD plus momentum. When I train a GAN, the situation flips: SGD is fragile and people reach for Adam because it is more stable. So I am always trading fast-but-generalizes-worse against slow-but-generalizes-better, and neither is reliably stable. I want one optimizer that is fast like Adam, generalizes like SGD, and is stable enough for the hard cases. And I want it to be a tiny change to Adam, because Adam's machinery, the moving averages, the bias correction, the per-coordinate denominator, clearly works; I do not want to throw it out, I want to find the one piece that is wrong.

So I look hard at what Adam's denominator actually does and where it misleads. Adam keeps m_t = beta_1 m_{t-1} + (1 - beta_1) g_t, an EMA of the gradient, and v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2, an EMA of the squared gradient, and steps by mhat_t / sqrt(vhat_t). The denominator sqrt(v_t) tracks the recent magnitude of the gradient in each coordinate: big gradients give small steps, small gradients give big steps. The usual story is that this is a cheap stand-in for curvature. But gradient magnitude and curvature are not the same object, so before I trust that story I want to stress-test the denominator with the simplest one-dimensional pictures and see whether magnitude and the ideal step really move together.

In a flat stretch, the gradient is small and barely changing, so curvature is small too. The right behavior is a large step because I am in a gentle region and should cross it. SGD takes a tiny step because its step is proportional to the small gradient. Adam takes a big step because sqrt(v_t) is small. Adam looks good here.

In a steep narrow valley, I oscillate across the bottom. The gradient is large in magnitude and it is changing or flipping quickly, so curvature is large. The right behavior is a small step. SGD takes a large step and overshoots. Adam takes a small step because sqrt(v_t) is large. Adam still looks good.

So far magnitude and the desired step agree, and that agreement is exactly why Adam looks curvature-aware. To find where the story breaks I need a case where they pull apart: a region where the gradient is large in magnitude but almost constant from step to step. Large gradient, tiny curvature. This can happen when the learning rate is small and I am crawling down a long steady slope. The surface is locally gentle and the gradient is pointing steadily downhill, so I should move boldly. Adam sees only the large magnitude, makes v_t large, and takes a small step. SGD, whose step is proportional to the large gradient, does the sensible thing. So in this third regime Adam is timid on long reliable slopes where bold progress is warranted.

Let me put a number on how badly Adam underperforms here, because "timid" is vague. Take a single coordinate with a constant gradient g = 10 and run Adam's two EMAs with the usual beta_1 = 0.9, beta_2 = 0.999 to a steady state. m_t converges to 10 and v_t converges to 100, so after bias correction the Adam step is mhat / sqrt(vhat) = 10 / 10 = 1. Adam moves by about one unit of learning rate per step even though the gradient is steady as a rock and there is nothing to be cautious about. That is the leak: a perfectly predictable slope is being throttled to the same step a noisy coordinate of the same magnitude would get.

Putting the three cases together, the ideal step depends on how quickly the gradient is changing, not on its bare magnitude. Magnitude and change happen to agree in the first two cases, and they decouple in the third. So the denominator I want should depend on gradient change or gradient surprise, not on g_t^2.

The most literal change signal is |g_t - g_{t-1}|, but I already maintain a smoother prediction of the next gradient: m_t. The EMA is not just a momentum vector; it is the running prediction of what the gradient should look like if the direction is stable. So the residual g_t - m_t is the natural surprise signal. If the gradient is steady, m_t tracks it and g_t - m_t is small. If the gradient is oscillating or noisy, g_t - m_t is large. Squaring and smoothing that residual is the obvious candidate for the denominator statistic.

So suppose I replace Adam's v_t = EMA(g_t^2) with s_t = beta_2 s_{t-1} + (1 - beta_2)(g_t - m_t)^2, and divide by sqrt(s_t). The rest of the Adam skeleton stays: the same m_t, the same bias correction, the same per-coordinate division. Now I have to actually check that this centered statistic gets all three cases right, not just assert it. The first two cases I can reason through quickly: in the flat region g_t is small and steady so the residual is small and the step is large; in the steep oscillating valley g_t flips quickly so the residual is large and the step is small. The third case is the one the whole design hinges on, so I will run it numerically rather than eyeball it. Constant g = 10 again, beta_2 = 0.999, cold start. m_t climbs to 10 and the residual g_t - m_t shrinks as it does, so s_t settles around 0.058 and sqrt(s_t) is about 0.24. The step is mhat / sqrt(shat), roughly 10 / 0.24 = 41. So on the long steady slope where Adam stepped by 1, the centered statistic steps by about 41 — a factor of forty more progress, in exactly the regime where bold progress is warranted. The third case is fixed, and the first two are unharmed, so the swap survives its own stress test.

This also gives the update a meaning that the raw second moment did not have. m_t is my prediction; g_t is my observation; g_t - m_t is the prediction error. A small prediction error means I believe the observed gradient, so 1 / sqrt(s_t) should be large and the step should be confident. A large prediction error means the observation disagrees with what the recent history predicted, so 1 / sqrt(s_t) should be small and the step should be cautious. The stepsize is adapted by my belief in the observed gradient.

The statistical reading lands in the same place. Once the EMA bias has decayed, m_t is close to E[g_t]. Then s_t = EMA[(g_t - m_t)^2] is close to E[(g_t - E[g_t])^2] = Var(g_t). Adam adapts to the uncentered second moment E[g_t^2]; this adapts to the centered second moment, the variance. That distinction matters because E[g_t^2] = (E[g_t])^2 + Var(g_t). If the mean gradient is large and the variance is small, Adam's denominator is large even though the direction is reliable. The centered statistic drops the (E[g_t])^2 term and stays small, which is exactly the forty-fold difference I just measured: there v was 100 = 10^2 + 0 and s was the leftover variance-like residual near 0.

The two-dimensional sign example should expose the same effect on a function I can write down, so let me build it and run the EMAs rather than wave at the limits. Let f(x,y) = |x| + |y|, so each gradient component is +1 or -1. Start with x << 0 and y near 0. The gradient of |x| is then a constant -1 in x, while y sits near the kink and its sign flips, so I model g_y as +1, -1, +1, -1, ... The trajectory I want keeps moving in x while damping the y oscillation. I run both Adam's v and the centered s for 2000 steps with beta_2 = 0.999 and bias-correct at the end. For x: m_x converges to -1, vhat_x = 1.000, and shat_x comes out to about 6.7e-4, so sqrt(shat_x) is about 0.026. For y: m_y hovers near -0.05, vhat_y = 1.000, and shat_y is about 0.90, so sqrt(shat_y) is about 0.95. So Adam's denominator is identical in the two coordinates — vhat_x = vhat_y = 1 — and it takes the same-size step in the consistent direction and the oscillating one; squaring the raw gradient erased the sign pattern. The centered statistic gives 1 / sqrt(shat_x) over 1 / sqrt(shat_y) of about 0.95 / 0.026 ≈ 36, so the consistent direction gets a roughly 36x larger step than the oscillating one. That is the behavior I wanted, and now I have the actual ratio, not a claim about it.

One honesty check on that example: I called shat_x "about 0.026", but is that a true floor or just a transient? If I let m_x sit at its converged value -1 from the start, the residual g - m is exactly 0 every step and s_x is exactly 0. Running from a cold start and extending the horizon, sqrt(shat_x) reads 0.29 at T=50, 0.05 at T=1000, 0.026 at T=2000, 0.005 at T=5000, 0.0004 at T=10000. So s_x is a decaying transient left over from m_x climbing toward -1, not a fixed value; on a perfectly steady coordinate the centered denominator keeps shrinking toward zero. Good to know, and it immediately flags a problem I have to handle: if s_t can go to zero, mhat_t / sqrt(shat_t) can blow up.

So I need to keep the denominator from vanishing. Adam already carries epsilon in the denominator; here I also add epsilon into the stored second statistic so the statistic itself has a positive lower bound and the blow-up I just saw coming cannot happen. The core recurrence becomes s_t = beta_2 s_{t-1} + (1 - beta_2)(g_t - m_t)^2 + epsilon, then shat_t = s_t / (1 - beta_2^t), and the update is theta_t = theta_{t-1} - alpha mhat_t / (sqrt(shat_t) + epsilon). The algorithmic knobs are still Adam's knobs: alpha, beta_1, beta_2, and epsilon, with no new core hyperparameter. For the plain algorithm epsilon is typically 1e-8; for the rectified, decoupled implementation variant I can default the same epsilon knob to 1e-16 when that is the more useful numerical setting.

There is a curvature link hiding in the residual. A diagonal finite-difference Hessian entry looks like H_ii approximately [g_i(theta + delta e_i) - g_i(theta)] / delta. If the predicted gradient m_t stands in for the nearby old gradient and the observed gradient g_t for the new one, then |g_t - m_t| is proportional to the local gradient change. So sqrt(s_t) is tracking the amplitude of a diagonal curvature signal, while sqrt(v_t) is tracking only gradient magnitude. That is consistent with what the numbers showed: on the constant-gradient slope the gradient change is zero, so the curvature-like signal is zero and the step is large, while Adam's magnitude signal stays at 10.

Now I want to know whether this centered swap has any optimization backbone or whether it is only an intuition that happens to work on my three toy cases. The honest answer is that I cannot prove convergence by running examples; I have to see whether it fits the kind of regret analysis the Adam family uses. Those proofs lean on a few stabilizers: a decaying learning rate alpha_t = alpha / sqrt(t), a convex feasible set F with diameter D_infinity for the online convex case, a weighted projection Pi_{F, sqrt(s_t)}, a lower bound s_{t,i} >= c > 0, and a monotone second statistic s_{t-1} <= s_t, which I can get with an AMSGrad-style running maximum if the proof needs it. I can omit bias correction inside the proof and put it back in the practical algorithm, as usual for this family. The lower bound c is already in hand — that is exactly the epsilon I just added to the recurrence — so at least the precondition I would otherwise have to bolt on is met for free. Let me work the regret algebra and see if it actually closes.

The projection step is theta_{t+1} = Pi_{F, sqrt(s_t)}(theta_t - alpha_t s_t^{-1/2} m_t). The weighted projection is non-expansive in the weighted norm, so against any optimum theta* in F,

||s_t^{1/4}(theta_{t+1} - theta*)||^2 <= ||s_t^{1/4}(theta_t - alpha_t s_t^{-1/2} m_t - theta*)||^2.

Expanding the right side gives

||s_t^{1/4}(theta_t - theta*)||^2 + alpha_t^2 ||s_t^{-1/4} m_t||^2 - 2 alpha_t <m_t, theta_t - theta*>.

The momentum is m_t = beta_{1t} m_{t-1} + (1 - beta_{1t}) g_t. I solve the inequality for <g_t, theta_t - theta*>. The beta_{1t} m_{t-1} term has the wrong sign for a clean telescope, so I bound it with Cauchy-Schwarz and Young:

- beta_{1t} / (1 - beta_{1t}) <m_{t-1}, theta_t - theta*> <= beta_{1t} / [2(1 - beta_{1t})] alpha_t ||s_t^{-1/4} m_{t-1}||^2 + beta_{1t} / [2 alpha_t(1 - beta_{1t})] ||s_t^{1/4}(theta_t - theta*)||^2.

Then the one-step inequality is

<g_t, theta_t - theta*> <= [||s_t^{1/4}(theta_t - theta*)||^2 - ||s_t^{1/4}(theta_{t+1} - theta*)||^2] / [2 alpha_t(1 - beta_{1t})] + alpha_t ||s_t^{-1/4} m_t||^2 / [2(1 - beta_{1t})] + beta_{1t} alpha_t ||s_t^{-1/4} m_{t-1}||^2 / [2(1 - beta_{1t})] + beta_{1t} ||s_t^{1/4}(theta_t - theta*)||^2 / [2 alpha_t(1 - beta_{1t})].

Convexity turns regret into a sum of these inner products. Since beta_{1t} <= beta_1, s_t is coordinatewise nondecreasing, and alpha_t is nonincreasing, the first difference telescopes after I insert and subtract the same point measured with s_{t-1} and alpha_{t-1}. The momentum-square terms combine into (1 + beta_1) / [2(1 - beta_1)] sum_t alpha_t ||s_t^{-1/4} m_t||^2. The leftover momentum inner-product price stays as sum_t beta_{1t} ||s_t^{1/4}(theta_t - theta*)||^2 / [2 alpha_t(1 - beta_1)].

The only non-obvious bound left is sum_t alpha_t ||s_t^{-1/4} m_t||^2, and this is where the lower bound c earns its keep: s_t >= c gives ||s_t^{-1/4} m_t||^2 <= ||m_t||^2 / sqrt(c). Unrolling the EMA in coordinate i gives m_{T,i} = sum_{j=1}^T (1 - beta_{1,j}) g_{j,i} prod_{k=1}^{T-j} beta_{1,T-k+1}. Since beta_{1,j} <= beta_1, I can upper-bound the weights by beta_1^{T-j}. Cauchy-Schwarz gives (sum_j beta_1^{T-j} g_{j,i})^2 <= (sum_j beta_1^{T-j})(sum_j beta_1^{T-j} g_{j,i}^2), and the geometric sum contributes 1 / (1 - beta_1). Doing the same recursively for every t, swapping the order of summation, and using sum_{j=t}^T beta_1^{j-t} <= 1 / (1 - beta_1), I get

sum_{t=1}^T alpha_t ||s_t^{-1/4} m_t||^2 <= alpha sqrt(1 + log T) / [sqrt(c)(1 - beta_1)^2] sum_{i=1}^d ||g_{1:T,i}^2||_2.

Putting that back into the regret sum gives

sum_{t=1}^T [f_t(theta_t) - f_t(theta*)] <= D_infinity^2 sqrt(T) / [2 alpha(1 - beta_1)] sum_i s_{T,i}^{1/2} + (1 + beta_1) alpha sqrt(1 + log T) / [2 sqrt(c)(1 - beta_1)^3] sum_i ||g_{1:T,i}^2||_2 + D_infinity^2 / [2(1 - beta_1)] sum_{t=1}^T sum_i beta_{1t} s_{t,i}^{1/2} / alpha_t.

If I choose beta_{1t} = beta_1 lambda^t with 0 < lambda < 1, the last term is controlled by the arithmetico-geometric sum sum_t lambda^{t-1} t <= 1 / (1 - lambda)^2, giving the constant term D_infinity^2 beta_1 G_infinity / [2(1 - beta_1)(1 - lambda)^2 alpha]. So the leading regret scales like O(sqrt(T)) under the same style of assumptions as the Adam-family analyses, with sum_i s_{T,i}^{1/2} replacing the raw second-moment term. The centered statistic did not break the telescope: every place where the Adam proof used v_t coordinatewise nondecreasing and lower-bounded, s_t plays the same role, provided I supply the c lower bound (the added epsilon) and, if I want the monotonicity literally, the running maximum.

For stochastic non-convex training I want a different check: does the expected gradient norm go to zero at the usual rate? I can lean on an Adam-type inequality that upper-bounds E[sum_t alpha_t <grad f(theta_t), grad f(theta_t) / sqrt(s_t)>] by three quantities: C_1 sum_t ||alpha_t g_t / sqrt(s_t)||^2, C_2 sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||_1, and C_3 sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||^2, plus a T-independent C_4. The work is to make each term finite with the centered statistic, and again the c lower bound is what carries it.

The first term is controlled by c:

E sum_t ||alpha_t g_t / sqrt(s_t)||^2 <= (1 / c) sum_t alpha_t^2 E||g_t||^2.

With unbiased independent noise g_t = grad f(theta_t) + zeta_t, E||g_t||^2 = E||grad f(theta_t)||^2 + E||zeta_t||^2 <= H^2 + sigma^2. Since alpha_t = alpha / sqrt(t), this contributes alpha^2 (H^2 + sigma^2)(1 + log T) / c.

The second term is a telescope because alpha_t decreases and s_t increases:

E sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||_1 = E sum_i [alpha_1 / sqrt(s_{1,i}) - alpha_T / sqrt(s_{T,i})] <= d alpha / sqrt(c).

The third term is bounded by multiplying the absolute difference term by the largest possible coordinate value alpha / sqrt(c), so it is at most d alpha^2 / c. For the lower side, the bounded-gradient assumption gives 1 / sqrt(s_{t,i}) >= 1 / H, so

E sum_t alpha_t <grad f(theta_t), grad f(theta_t) / sqrt(s_t)> >= (alpha sqrt(T) / H) min_{t <= T} E||grad f(theta_t)||^2.

Combining upper and lower bounds yields

min_{t <= T} E||grad f(theta_t)||^2 <= H / (sqrt(T) alpha) [C_1 alpha^2 (H^2 + sigma^2)(1 + log T) / c + C_2 d alpha / sqrt(c) + C_3 d alpha^2 / c + C_4],

which is (Q_1 + Q_2 log T) / sqrt(T), with Q_1 = H/alpha [C_1 alpha^2(H^2 + sigma^2)/c + C_2 d alpha/sqrt(c) + C_3 d alpha^2/c + C_4] and Q_2 = H C_1 alpha(H^2 + sigma^2)/c. So the min gradient norm goes to zero like (log T)/sqrt(T), the standard Adam-type non-convex rate. Both analyses go through for the same reason: the centered statistic is still a positive, lower-bounded, (optionally) monotone per-coordinate denominator, so it slots into the template wherever Adam used v_t.

The Bayesian reading is the last sanity check, and I want to write it out rather than gesture at it. Suppose the true gradient has prior g_tilde ~ N(0, sigma^2 I), and the observed minibatch gradient is g ~ N(g_tilde, C). The posterior precision is sigma^{-2} I + C^{-1}, so the posterior covariance is (sigma^{-2} I + C^{-1})^{-1}. The posterior mean is (sigma^{-2} I + C^{-1})^{-1} C^{-1} g. Because (sigma^{-2} I + C^{-1}) = C^{-1}(I + C / sigma^2), the mean simplifies to (I + C / sigma^2)^{-1} g = sigma^2 (sigma^2 I + C)^{-1} g, proportional to (sigma^2 I + C)^{-1} g. With epsilon = sigma^2, the reliability-weighted direction is (epsilon I + C)^{-1} g, and in practice I use m_t in place of a single noisy g_t.

Full inverse covariance is too expensive and can be numerically harsh, so I use the diagonal square-root form that adaptive optimizers already use: replace (epsilon I + C)^{-1} m_t by a learning-rate-scaled diagonal (epsilon I + C)^{-1/2} m_t. The only question is how to estimate C. Adam uses the uncentered diagonal E[g_t g_t^T], but C is covariance, so it should be centered. The diagonal centered estimate is EMA[(g_t - E[g_t])(g_t - E[g_t])^T], and m_t is the local estimate of E[g_t]. That brings me right back to s_t = EMA[(g_t - m_t)^2]. So the Bayesian derivation arrives at the same statistic I reached from the three toy cases, which is the agreement I was hoping for: in a coordinate where Var(g_t^i) is much smaller than |E[g_t^i]|, the centered C_i is much smaller than Adam's uncentered E[(g_t^i)^2], so the step is larger — the same large-gradient, low-curvature behavior, now falling out of the posterior rather than out of an intuition.

The optional engineering branches now make sense as orthogonal pieces rather than part of the core idea. A running maximum of s_t supplies the monotonicity the convex proof asked for. RAdam-style rectification protects the early steps when the adaptive statistic has too few effective samples; if the estimated degrees of freedom N_sma is below 5, fall back to the momentum update, and if it is at least 5, use the rectified adaptive step. Decoupled weight decay keeps regularization separate from the gradient statistic. These are inherited Adam-family controls, not new denominator logic.

Before I commit to that rectification branch I should trace N_sma on actual step counts, because the whole point is to know how long the fallback lasts. With beta_2 = 0.999, N_sma_max = 2/(1 - beta_2) - 1 = 1999. Plugging into N_sma = N_sma_max - 2 t beta_2^t / (1 - beta_2^t): at t = 1 it is 1.0, at t = 5 it is 5.0 (just under, 4.996), at t = 10 it is 9.98. So N_sma crosses the threshold of 5 between steps 5 and 10 — the first nine or so updates take the SGD/momentum fallback, and from step 10 on the rectified adaptive denominator is in use. That matches the intent (warmup only over the first handful of steps, then adaptive) and tells me the buffer-cached branch in the code is doing the right thing.

I can now write the optimizer in the same shape as Adam: one buffer for m_t, one buffer for the centered second statistic, optional max-statistic storage, decoupled or coupled weight decay, then either the plain adaptive step or the rectified branch. The crucial implementation detail is that epsilon is added into the stored statistic before taking the square root, so the recurrence really stores the lower-bounded s_t — the same c that the proofs leaned on and the same guard against the s_t -> 0 blow-up I saw in the steady-coordinate transient.

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True,
                 fixed_decay=False, rectify=True, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param_group in params:
                if 'betas' in param_group and param_group['betas'] != betas:
                    param_group['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.degenerated_to_sgd = degenerated_to_sgd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_var'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data)

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                elif group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4) / (N_sma_max - 4)
                                * (N_sma - 2) / N_sma
                                * N_sma_max / (N_sma_max - 2)
                            ) / bias_correction1
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / bias_correction1
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

        return loss
```

The chain holds together: Adam's denominator is sold as curvature scaling but really tracks gradient magnitude, and the constant-gradient test showed magnitude alone throttles a steady slope to a step of 1 where the centered statistic steps by about 40; magnitude is fine in flat regions and steep oscillating valleys but wrong on long steady slopes; the residual g_t - m_t measures whether the gradient agrees with its own running prediction, so its squared EMA estimates variance (the |x|+|y| run gave a 36x step ratio between the consistent and oscillating coordinates while Adam saw them as identical), preserves sign-consistency information, behaves like a diagonal curvature-change signal, falls out of the Bayesian posterior with epsilon = sigma^2, and slots into the same Adam-family regret and non-convex bounds once the added epsilon supplies the lower bound and an optional running maximum supplies monotonicity. The resulting optimizer is still an Adam-shaped first-order method, but its per-coordinate step is controlled by belief in the observed gradient rather than by the raw size of that gradient.
