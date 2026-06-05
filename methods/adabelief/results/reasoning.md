Let me start from the split that keeps frustrating me. When I train a CNN, Adam gets the loss down fast in the first few epochs, but the model it lands on generalizes worse than if I grind it out with plain SGD plus momentum. When I train a GAN, the situation flips: SGD is fragile and people reach for Adam because it is more stable. So I am always trading fast-but-generalizes-worse against slow-but-generalizes-better, and neither is reliably stable. I want one optimizer that is fast like Adam, generalizes like SGD, and is stable enough for the hard cases. And I want it to be a tiny change to Adam, because Adam's machinery, the moving averages, the bias correction, the per-coordinate denominator, clearly works; I do not want to throw it out, I want to find the one piece that is wrong.

So I look hard at what Adam's denominator actually does and where it misleads. Adam keeps m_t = beta_1 m_{t-1} + (1 - beta_1) g_t, an EMA of the gradient, and v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2, an EMA of the squared gradient, and steps by mhat_t / sqrt(vhat_t). The denominator sqrt(v_t) tracks the recent magnitude of the gradient in each coordinate: big gradients give small steps, small gradients give big steps. The usual story is that this is a cheap stand-in for curvature. But gradient magnitude and curvature are not the same object, so I need to stress-test the denominator with the simplest one-dimensional pictures.

In a flat stretch, the gradient is small and barely changing, so curvature is small too. The right behavior is a large step because I am in a gentle region and should cross it. SGD takes a tiny step because its step is proportional to the small gradient. Adam takes a big step because sqrt(v_t) is small. Adam looks good here.

In a steep narrow valley, I oscillate across the bottom. The gradient is large in magnitude and it is changing or flipping quickly, so curvature is large. The right behavior is a small step. SGD takes a large step and overshoots. Adam takes a small step because sqrt(v_t) is large. Adam still looks good.

The third case is where the story breaks: a region where the gradient is large in magnitude but almost constant from step to step. Large gradient, tiny curvature. This can happen when the learning rate is small and I am crawling down a long steady slope. The surface is locally gentle and the gradient is pointing steadily downhill, so I should move boldly. Adam sees only the large magnitude, makes v_t large, and takes a small step. SGD, whose step is proportional to the large gradient, does the sensible thing. That is the exact regime I care about if SGD's final behavior is better: Adam is timid on long reliable slopes where bold progress is warranted.

Putting the three cases together, the ideal step depends on how quickly the gradient is changing, not on its bare magnitude. Magnitude and change happen to agree in the first two cases, and that coincidence makes Adam look curvature-aware. They decouple in the third case, and Adam chooses the wrong step. So the denominator I want should depend on gradient change or gradient surprise.

The most literal change signal is |g_t - g_{t-1}|, but I already maintain a smoother prediction of the next gradient: m_t. The EMA is not just a momentum vector; it is the running prediction of what the gradient should look like if the direction is stable. So the residual g_t - m_t is the natural surprise signal. If the gradient is steady, m_t tracks it and g_t - m_t is small. If the gradient is oscillating or noisy, g_t - m_t is large. Squaring and smoothing that residual gives me exactly the denominator statistic I need.

So I replace Adam's v_t = EMA(g_t^2) with s_t = beta_2 s_{t-1} + (1 - beta_2)(g_t - m_t)^2, and I divide by sqrt(s_t). The rest of the Adam skeleton stays: the same m_t, the same bias correction, and the same per-coordinate division. With this centered statistic, the three cases line up. In the flat region, g_t is small and steady, so s_t is small and the step is large. In the steep oscillating valley, g_t changes quickly, so s_t is large and the step is small. In the large-gradient, small-curvature region, g_t is large but predictable, so g_t - m_t is small, s_t is small, and the step is large. The centered second moment keeps Adam's good behavior in the first two cases and fixes the third.

This also gives the update a meaning that the raw second moment did not have. m_t is my prediction; g_t is my observation; g_t - m_t is the prediction error. A small prediction error means I believe the observed gradient, so 1 / sqrt(s_t) should be large and the step should be confident. A large prediction error means the observation disagrees with what the recent history predicted, so 1 / sqrt(s_t) should be small and the step should be cautious. The stepsize is adapted by my belief in the observed gradient.

The statistical reading lands in the same place. Once the EMA bias has decayed, m_t is close to E[g_t]. Then s_t = EMA[(g_t - m_t)^2] is close to E[(g_t - E[g_t])^2] = Var(g_t). Adam adapts to the uncentered second moment E[g_t^2]; this adapts to the centered second moment, the variance. That distinction matters because E[g_t^2] = (E[g_t])^2 + Var(g_t). If the mean gradient is large and the variance is small, Adam's denominator is large even though the direction is reliable. The centered statistic stays small and lets the optimizer move.

The two-dimensional sign example makes the failure almost too clean. Let f(x,y) = |x| + |y|, so each gradient component is either +1 or -1. Start near the x-axis with x << 0 and y near 0. The trajectory should keep moving in x while damping the y oscillation. In the long run, m_x is approximately E[g_x] = 1 and m_y is approximately E[g_y] = 0. Adam sees v_x approximately E[g_x^2] = 1 and v_y approximately E[g_y^2] = 1, so it takes the same-size step in x and y. Squaring the raw gradient has destroyed the sign pattern. The residual statistic gives s_x approximately E[(1 - 1)^2] = 0 and s_y approximately E[(+/-1 - 0)^2] = 1, so 1 / sqrt(s_x) is much larger than 1 / sqrt(s_y). The consistent direction gets a large step and the oscillating direction gets a small one.

I still need to keep the denominator from vanishing. If the gradient becomes perfectly predictable, s_t can go to zero, and mhat_t / sqrt(shat_t) can blow up. Adam already has epsilon in the denominator; here I also add epsilon into the stored second statistic so the statistic itself has a positive lower bound. The core recurrence becomes s_t = beta_2 s_{t-1} + (1 - beta_2)(g_t - m_t)^2 + epsilon, then shat_t = s_t / (1 - beta_2^t), and the update is theta_t = theta_{t-1} - alpha mhat_t / (sqrt(shat_t) + epsilon). The algorithmic knobs are still Adam's knobs: alpha, beta_1, beta_2, and epsilon, with no new core hyperparameter. For the plain algorithm epsilon is typically 1e-8; for the rectified, decoupled implementation variant I can default the same epsilon knob to 1e-16 when that is the more useful numerical setting.

There is a curvature link hiding in the residual. A diagonal finite-difference Hessian entry looks like H_ii approximately [g_i(theta + delta e_i) - g_i(theta)] / delta. If the predicted gradient m_t stands in for the nearby old gradient and the observed gradient g_t for the new one, then |g_t - m_t| is proportional to the local gradient change. So sqrt(s_t) is tracking the amplitude of a diagonal curvature signal, while sqrt(v_t) is tracking only gradient magnitude. That explains why the centered denominator gets the three cases right.

I should check that this is not just an intuition with no optimization backbone. The Adam-style convergence proofs need a few stabilizers: a decaying learning rate alpha_t = alpha / sqrt(t), a convex feasible set F with diameter D_infinity for the online convex case, a weighted projection Pi_{F, sqrt(s_t)}, a lower bound s_{t,i} >= c > 0, and a monotone second statistic s_{t-1} <= s_t, which I can get with an AMSGrad-style running maximum if I need the proof condition. I can omit bias correction inside the proof and put it back in the practical algorithm, as usual for this family.

The projection step is theta_{t+1} = Pi_{F, sqrt(s_t)}(theta_t - alpha_t s_t^{-1/2} m_t). The weighted projection is non-expansive in the weighted norm, so against any optimum theta* in F I get

||s_t^{1/4}(theta_{t+1} - theta*)||^2 <= ||s_t^{1/4}(theta_t - alpha_t s_t^{-1/2} m_t - theta*)||^2.

Expanding the right side gives

||s_t^{1/4}(theta_t - theta*)||^2 + alpha_t^2 ||s_t^{-1/4} m_t||^2 - 2 alpha_t <m_t, theta_t - theta*>.

The momentum is m_t = beta_{1t} m_{t-1} + (1 - beta_{1t}) g_t. I solve the inequality for <g_t, theta_t - theta*>. The beta_{1t} m_{t-1} term has the wrong sign for a clean telescope, so I bound it with Cauchy-Schwarz and Young:

- beta_{1t} / (1 - beta_{1t}) <m_{t-1}, theta_t - theta*> <= beta_{1t} / [2(1 - beta_{1t})] alpha_t ||s_t^{-1/4} m_{t-1}||^2 + beta_{1t} / [2 alpha_t(1 - beta_{1t})] ||s_t^{1/4}(theta_t - theta*)||^2.

Then the one-step inequality is

<g_t, theta_t - theta*> <= [||s_t^{1/4}(theta_t - theta*)||^2 - ||s_t^{1/4}(theta_{t+1} - theta*)||^2] / [2 alpha_t(1 - beta_{1t})] + alpha_t ||s_t^{-1/4} m_t||^2 / [2(1 - beta_{1t})] + beta_{1t} alpha_t ||s_t^{-1/4} m_{t-1}||^2 / [2(1 - beta_{1t})] + beta_{1t} ||s_t^{1/4}(theta_t - theta*)||^2 / [2 alpha_t(1 - beta_{1t})].

Convexity turns regret into a sum of these inner products. Since beta_{1t} <= beta_1, s_t is coordinatewise nondecreasing, and alpha_t is nonincreasing, the first difference telescopes after I insert and subtract the same point measured with s_{t-1} and alpha_{t-1}. The momentum-square terms combine into (1 + beta_1) / [2(1 - beta_1)] sum_t alpha_t ||s_t^{-1/4} m_t||^2. The leftover momentum inner-product price stays as sum_t beta_{1t} ||s_t^{1/4}(theta_t - theta*)||^2 / [2 alpha_t(1 - beta_1)].

Now the only non-obvious bound is sum_t alpha_t ||s_t^{-1/4} m_t||^2. The lower bound s_t >= c gives ||s_t^{-1/4} m_t||^2 <= ||m_t||^2 / sqrt(c). Unrolling the EMA in coordinate i gives m_{T,i} = sum_{j=1}^T (1 - beta_{1,j}) g_{j,i} prod_{k=1}^{T-j} beta_{1,T-k+1}. Since beta_{1,j} <= beta_1, I can upper-bound the weights by beta_1^{T-j}. Cauchy-Schwarz gives (sum_j beta_1^{T-j} g_{j,i})^2 <= (sum_j beta_1^{T-j})(sum_j beta_1^{T-j} g_{j,i}^2), and the geometric sum contributes 1 / (1 - beta_1). Doing the same recursively for every t, swapping the order of summation, and using sum_{j=t}^T beta_1^{j-t} <= 1 / (1 - beta_1), I get

sum_{t=1}^T alpha_t ||s_t^{-1/4} m_t||^2 <= alpha sqrt(1 + log T) / [sqrt(c)(1 - beta_1)^2] sum_{i=1}^d ||g_{1:T,i}^2||_2.

Putting that back into the regret sum gives the bound I need:

sum_{t=1}^T [f_t(theta_t) - f_t(theta*)] <= D_infinity^2 sqrt(T) / [2 alpha(1 - beta_1)] sum_i s_{T,i}^{1/2} + (1 + beta_1) alpha sqrt(1 + log T) / [2 sqrt(c)(1 - beta_1)^3] sum_i ||g_{1:T,i}^2||_2 + D_infinity^2 / [2(1 - beta_1)] sum_{t=1}^T sum_i beta_{1t} s_{t,i}^{1/2} / alpha_t.

If I choose beta_{1t} = beta_1 lambda^t with 0 < lambda < 1, the last term is controlled by the arithmetico-geometric sum sum_t lambda^{t-1} t <= 1 / (1 - lambda)^2, giving the constant term D_infinity^2 beta_1 G_infinity / [2(1 - beta_1)(1 - lambda)^2 alpha] in that corollary. The important thing for the design is that the leading regret scales like O(sqrt(T)) under the same style of assumptions as the Adam-family analyses, with sum_i s_{T,i}^{1/2} replacing the raw second-moment term.

For stochastic non-convex training, I need a different check: does the expected gradient norm go to zero at the usual rate? I can lean on an Adam-type inequality that upper-bounds E[sum_t alpha_t <grad f(theta_t), grad f(theta_t) / sqrt(s_t)>] by three quantities: C_1 sum_t ||alpha_t g_t / sqrt(s_t)||^2, C_2 sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||_1, and C_3 sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||^2, plus a T-independent C_4. I need to make each term finite with the centered statistic.

The first term is controlled by the lower bound c:

E sum_t ||alpha_t g_t / sqrt(s_t)||^2 <= (1 / c) sum_t alpha_t^2 E||g_t||^2.

With unbiased independent noise g_t = grad f(theta_t) + zeta_t, E||g_t||^2 = E||grad f(theta_t)||^2 + E||zeta_t||^2 <= H^2 + sigma^2. Since alpha_t = alpha / sqrt(t), this contributes alpha^2 (H^2 + sigma^2)(1 + log T) / c.

The second term is a telescope because alpha_t decreases and s_t increases:

E sum_t ||alpha_t / sqrt(s_t) - alpha_{t-1} / sqrt(s_{t-1})||_1 = E sum_i [alpha_1 / sqrt(s_{1,i}) - alpha_T / sqrt(s_{T,i})] <= d alpha / sqrt(c).

The third term is bounded by multiplying the absolute difference term by the largest possible coordinate value alpha / sqrt(c), so it is at most d alpha^2 / c. For the lower side, the bounded-gradient assumption gives 1 / sqrt(s_{t,i}) >= 1 / H, so

E sum_t alpha_t <grad f(theta_t), grad f(theta_t) / sqrt(s_t)> >= (alpha sqrt(T) / H) min_{t <= T} E||grad f(theta_t)||^2.

Combining upper and lower bounds yields

min_{t <= T} E||grad f(theta_t)||^2 <= H / (sqrt(T) alpha) [C_1 alpha^2 (H^2 + sigma^2)(1 + log T) / c + C_2 d alpha / sqrt(c) + C_3 d alpha^2 / c + C_4],

which is (Q_1 + Q_2 log T) / sqrt(T), with Q_1 = H/alpha [C_1 alpha^2(H^2 + sigma^2)/c + C_2 d alpha/sqrt(c) + C_3 d alpha^2/c + C_4] and Q_2 = H C_1 alpha(H^2 + sigma^2)/c. The centered statistic is not just an intuitive swap; with the positive lower bound and the monotone variant, it fits the Adam-type convergence template.

The Bayesian interpretation also checks out if I write it explicitly. Suppose the true gradient has prior g_tilde ~ N(0, sigma^2 I), and the observed minibatch gradient is g ~ N(g_tilde, C). The posterior precision is sigma^{-2} I + C^{-1}, so the posterior covariance is (sigma^{-2} I + C^{-1})^{-1}. The posterior mean is (sigma^{-2} I + C^{-1})^{-1} C^{-1} g. Because (sigma^{-2} I + C^{-1}) = C^{-1}(I + C / sigma^2), the mean simplifies to (I + C / sigma^2)^{-1} g = sigma^2 (sigma^2 I + C)^{-1} g, proportional to (sigma^2 I + C)^{-1} g. With epsilon = sigma^2, the ideal reliability-weighted direction is (epsilon I + C)^{-1} g, and in practice I use m_t instead of a single noisy g_t.

Full inverse covariance is too expensive and can be numerically harsh, so I use the diagonal square-root form that adaptive optimizers already use: replace (epsilon I + C)^{-1} m_t by a learning-rate-scaled diagonal (epsilon I + C)^{-1/2} m_t. The only question is how to estimate C. Adam uses the uncentered diagonal E[g_t g_t^T], but C is covariance, so it should be centered. The diagonal centered estimate is EMA[(g_t - E[g_t])(g_t - E[g_t])^T], and m_t is the local estimate of E[g_t]. That brings me right back to s_t = EMA[(g_t - m_t)^2]. In a coordinate where Var(g_t^i) is much smaller than |E[g_t^i]|, the centered C_i is much smaller than Adam's uncentered E[(g_t^i)^2], so the step is larger, exactly the behavior I wanted in the large-gradient, low-curvature case.

The optional engineering branches now make sense as orthogonal pieces rather than part of the core idea. A running maximum of s_t gives the monotonicity used in the convex proof. RAdam-style rectification protects the early steps when the adaptive statistic has too few effective samples; if the estimated degrees of freedom N_sma is below 5, I can fall back to the momentum update, and if it is at least 5, I use the rectified adaptive step. Decoupled weight decay keeps regularization separate from the gradient statistic. These are inherited Adam-family controls, not new denominator logic.

I can now write the optimizer in the same shape as Adam: one buffer for m_t, one buffer for the centered second statistic, optional max-statistic storage, decoupled or coupled weight decay, then either the plain adaptive step or the rectified branch. The crucial implementation detail is that epsilon is added into the stored statistic before taking the square root, so the recurrence really stores the lower-bounded s_t.

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

The chain is now clean: Adam's denominator is sold as curvature scaling but really tracks gradient magnitude; magnitude is fine in flat regions and steep oscillating valleys but wrong on long steady slopes; the residual g_t - m_t measures whether the gradient agrees with its own running prediction, so its squared EMA estimates variance, preserves sign-consistency information, behaves like a diagonal curvature-change signal, and admits the same Adam-family proof scaffolding once I lower-bound and optionally monotone-ize the statistic. The resulting optimizer is still an Adam-shaped first-order method, but its per-coordinate step is controlled by belief in the observed gradient rather than by the raw size of that gradient.
