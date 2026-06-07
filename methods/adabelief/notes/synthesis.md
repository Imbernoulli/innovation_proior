# AdaBelief — synthesis (grounded in arXiv 2010.07468 main_all.tex + juntang-zhuang/Adabelief-Optimizer)

## Verified arXiv id: 2010.07468 (Zhuang et al., NeurIPS 2020)
Canonical impl: github.com/juntang-zhuang/Adabelief-Optimizer (code/adabelief.py downloaded).

## Pain point / research question
Want ONE optimizer with three properties simultaneously: (1) fast convergence like adaptive methods (Adam), (2) good generalization like SGD, (3) training stability (esp. GANs). Adaptive methods converge fast but generalize worse than SGD on CNNs (Wilson 2017); SGD generalizes well but slow early; on GANs adaptive methods are default for stability. Many Adam variants (AMSGrad, Yogi, RAdam, MSVAG, AdaBound, SWATS, Fromage, AdamW) improve some axis but still generalize worse than SGD on ImageNet and/or unstable on GANs.

## The method (exact, Algorithm 2)
Adam: m_t = beta1 m_{t-1} + (1-beta1)g_t; v_t = beta2 v_{t-1} + (1-beta2) g_t^2; mhat=m/(1-beta1^t); vhat=v/(1-beta2^t); theta = theta - alpha mhat/(sqrt(vhat)+eps).
AdaBelief (one-line change, diff in blue): replace v_t (EMA of g_t^2) with
  s_t = beta2 s_{t-1} + (1-beta2)(g_t - m_t)^2 + eps
shat = s_t/(1-beta2^t); theta = theta - alpha mhat/(sqrt(shat)+eps).
m_t same as Adam. Defaults alpha=1e-3, eps=1e-8, beta1=0.9, beta2=0.999 (Adam defaults). NO extra parameters vs Adam.
Note the extra +eps INSIDE s_t update (so s_t bounded below by ~eps; the blue (g_t-m_t)^2+eps), plus the usual +eps in denominator.
Projection Pi_{F,sqrt(vhat)} onto feasible set F for the (constrained) convergence-analysis version; in unconstrained DL it's just the update.

## Core idea / intuition
- m_t = EMA of past gradients = a PREDICTION of the gradient at next step.
- g_t - m_t = prediction error (observed minus predicted).
- s_t = EMA of (g_t - m_t)^2 = belief in the observation. 1/sqrt(s_t) = "belief":
  if g_t deviates much from prediction m_t -> large (g_t-m_t)^2 -> large s_t -> small 1/sqrt(s_t) -> SMALL (cautious) step (weak belief).
  if g_t close to m_t -> small s_t -> large 1/sqrt(s_t) -> LARGE (confident) step (strong belief).
- "Adapt stepsize by the belief in observed gradients."
- Statistically: s_t = EMA[(g_t - m_t)^2] approx E[(g_t - Eg_t)^2] = Var(g_t). So AdaBelief = adaptive-VARIANCE optimizer (Adam = adaptive second-MOMENT). Low-variance (consistent) gradient -> big step; high-variance -> small step.

## Curvature argument (1D, three cases, Fig curvature + Table)
"Stepsize" |Delta theta| should track an IDEAL optimizer: NOT large-where-gradient-large; should follow curvature.
SGD: |Delta theta| ~ alpha|m_t| (prop to gradient). Adam: ~ alpha|m_t/sqrt(v_t)|. AdaBelief: ~ alpha|m_t/sqrt(s_t)|.
Case 1 (flat region, |g|,|g_t-g_{t-1}|,v_t,s_t all small): ideal = LARGE step. SGD small (m small); Adam large (small v denom); AdaBelief large (small s denom). Adam & AdaBelief OK.
Case 2 (steep narrow valley, oscillating, |g| and |g_t-g_{t-1}| large -> v_t and s_t large): ideal = SMALL step. SGD large (bad); Adam small (large v); AdaBelief small (large s). Adam & AdaBelief OK.
Case 3 ("large gradient, small curvature": |g| and v_t large BUT |g_t-g_{t-1}| and s_t small; happens with small alpha): ideal = LARGE step. SGD large (~alpha|g|); Adam SMALL (large v denom -> WRONG); AdaBelief LARGE (small s denom -> RIGHT).
=> Only AdaBelief matches ideal in all 3. s_t ~ change-in-gradient (g_t - m_t ~ g_t - g_{t-1}) ~ related to Hessian/curvature. So AdaBelief scales by the change in gradient = curvature info; Adam scales by magnitude only.
(Hessian intuition: H_{ii} ~ [g(theta+delta)-g(theta)]/delta; identify g(theta)->m_t, g(theta+delta)->g_t, then sqrt(s_t,i) ~ |H_ii| amplitude. ignoring constant.)

## 2D sign example (Fig oscillation)
f(x,y)=|x|+|y|, gradient per axis in {+1,-1}. Start near x-axis, x0<<0, y0~0. Oscillate in y, advance in x.
Long run, bias small: m_{t,x}~E g_x=1, m_{t,y}~E g_y=0; g_x always +1, g_y alternates -1/+1.
Adam: v_x ~ E g_x^2 = 1, v_y ~ E g_y^2 = 1. EQUAL -> same stepsize x and y. BAD: should advance fast in x, damp y.
AdaBelief: s_x ~ Var(g_x) = 0 (g_x constant), s_y ~ Var(g_y) = 1. So 1/sqrt(s_x) >> 1/sqrt(s_y): LARGE step in x, SMALL in y. Matches ideal.
KEY: Adam's v uses |g| only (ignores SIGN); s_t uses g_t - m_t so it captures the sign/consistency. v_x=v_y but s_x<<s_y.

## Why three goals
- Fast convergence: adaptive denominator like Adam (Cases 1,2 same as Adam).
- Good generalization: in Case 3 and the "large gradient small curvature" regions AdaBelief takes large steps where Adam stalls; matching SGD-like behaviour where appropriate -> closes the adaptive/SGD generalization gap. Also s_t small in well-behaved directions -> trusts consistent gradient like SGD.
- Stability (GAN): denominator tracks variance; reacts to deviation -> smoother.

## eps placement nuance
s_t update adds eps: s_t = beta2 s_{t-1} + (1-beta2)(g_t-m_t)^2 + eps. Ensures s_t bounded below (>= ~eps), so when prediction is perfect (s->0) the step doesn't blow up; the denom also adds eps. In code: exp_avg_var.add_(eps).sqrt()/sqrt(bias_correction2) then .add_(eps).

## Load-bearing ancestors
- Adam (Kingma 2014): m, v EMAs, bias correction, m/sqrt(v). AdaBelief = Adam with v -> s (centered second moment).
- AdaGrad (Duchi 2011): sum of g^2 denominator -> per-coord adaptive lr; decays too fast.
- SGD(+momentum): step ~ m_t; generalizes well but no curvature adaptation.
- AMSGrad (Reddi 2018): max of v_t for convergence fix; AdaBelief optionally uses max(s_t) (amsgrad flag).
- RAdam (Liu 2019): rectify variance of adaptive lr early; AdaBelief optionally forks rectified update.
- MSVAG (Balles 2018): dissect Adam into sign + variance-adaptive magnitude; close in spirit (variance).
- AdamW (Loshchilov 2017): decoupled weight decay; AdaBelief supports weight_decouple.
- Bayesian filtering / belief: m_t = prediction, g_t = measurement, deviation -> trust. Kalman-like.

## Design decisions -> why
- s_t = EMA[(g_t - m_t)^2] not EMA[g_t^2]: it's the variance/prediction-error, gives curvature info & captures sign/consistency. Centered, so constant-gradient directions get tiny denom -> big step (the Adam fix in Case 3 / 2D-x).
- Use m_t (not g_{t-1}) as the prediction center: m_t is the smoothed prediction; g_t - m_t is residual. (g_t - m_t ~ g_t - g_{t-1} for slowly-varying m.)
- +eps inside s_t: lower-bound the denom for stability when belief is near-perfect.
- bias correction on both m and s (Adam-style): correct the zero-init bias of EMAs.
- No new hyperparameters: drop-in for Adam, reuse alpha=1e-3, beta=(0.9,0.999), eps=1e-8.
- optional amsgrad (max s), rectify (RAdam warmup), decoupled weight decay: orthogonal robustness knobs.

## Code (juntang-zhuang) core
state: exp_avg (m), exp_avg_var (s).
exp_avg.mul_(beta1).add_(grad, 1-beta1)            # m_t
grad_residual = grad - exp_avg                      # g_t - m_t
exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1-beta2)   # s_t (eps added below)
denom = (exp_avg_var.add_(eps).sqrt() / sqrt(bias_correction2)).add_(eps)
step_size = lr / bias_correction1
p.data.addcdiv_(exp_avg, denom, value=-step_size)   # theta -= lr * mhat/(sqrt(shat)+eps)
decoupled wd: p.data.mul_(1 - lr*wd) before. amsgrad: max(s). rectify: RAdam-style.
