# Lookahead — synthesis (grounded in arXiv 1907.08610 source + michaelrzhang/lookahead repo)

## Verified arXiv id: 1907.08610 (Zhang, Lucas, Hinton, Ba, NeurIPS 2019)
Canonical impl: https://github.com/michaelrzhang/lookahead — lookahead_pytorch.py (downloaded verbatim to code/)

## Pain point / research question
SGD-variants (adaptive: AdaGrad/Adam; accelerated: heavy-ball/Nesterov) dominate NN training but need costly hyperparameter tuning to avoid oscillation/slow convergence. Both adaptive and accelerated approaches accumulate past gradients. Want a method ORTHOGONAL to both that improves stability and reduces variance with negligible compute/memory cost and minimal tuning.

## The method (exact, from main.tex)
Maintain slow weights phi and fast weights theta.
Inner loop: theta_{t,0} <- phi_{t-1}; for i=1..k: sample minibatch d~D, theta_{t,i} <- theta_{t,i-1} + A(L, theta_{t,i-1}, d) where A = any optimizer.
Outer update: phi_t <- phi_{t-1} + alpha (theta_{t,k} - phi_{t-1}).
Then reset fast weights to slow weights. Defaults in repo: la_steps k=5, la_alpha alpha=0.8.
Note alpha=1.0 recovers inner optimizer exactly.

## EMA view (main.tex eq, derived)
phi_{t+1} = phi_t + alpha(theta_{t,k} - phi_t)
          = alpha[theta_{t,k} + (1-alpha)theta_{t-1,k} + ... + (1-alpha)^{t-1} theta_{0,k}] + (1-alpha)^t phi_0
So slow weights = EMA of the FINAL fast weight of each inner loop. Weights recent proposals heavily but keeps influence of older ones. Decay factor (1-alpha).

## Optimal slow-weight step size (Prop 1, proof in app)
Quadratic L(x) = 1/2 x^T A x - b^T x, optimum theta* = A^{-1} b.
alpha* = argmin_alpha L(theta_{t,0} + alpha(theta_{t,k}-theta_{t,0}))
       = [(theta_{t,0}-theta*)^T A (theta_{t,0}-theta_{t,k})] / [(theta_{t,0}-theta_{t,k})^T A (theta_{t,0}-theta_{t,k})]
PROOF: d/dalpha L = (theta_{t,k}-theta_{t,0})^T A(theta_{t,0}+alpha(theta_{t,k}-theta_{t,0})) - (theta_{t,k}-theta_{t,0})^T b. Set=0, sub b=A theta*:
alpha[(theta_{t,k}-theta_{t,0})^T A (theta_{t,k}-theta_{t,0})] = (theta_{t,k}-theta_{t,0})^T A(theta*-theta_{t,0}). Rearrange -> alpha*.
Practical adaptive: replace theta* with theta_{t,k} - Ahat^{-1} grad L(theta_{t,k}) (Ahat = empirical-Fisher diagonal, e.g. Adam's v), clip to [alpha_low, 1]. But fixed alpha works as well + generalizes better + avoids storing Fisher for SGD -> use fixed alpha.

## Noisy quadratic analysis (Section + Appendix, the variance-reduction heart)
Model (Schaul 2013, Wu 2018): Lhat(x) = 1/2 (x-c)^T A(x-c), c ~ N(x*=0, Sigma), A and Sigma diagonal, a_i, sigma_i^2.
Expected loss L(theta^t) = 1/2 sum_i a_i (E[theta_i]^2 + V[theta_i] + sigma_i^2).
SGD dynamics (lr gamma): E[x^{t+1}] = (I-gamma A) E[x^t]; V[x^{t+1}] = (I-gamma A)^2 V[x^t] + gamma^2 A^2 Sigma.
SGD variance fixed point: V*_SGD = gamma^2 A^2 Sigma / (I - (I-gamma A)^2).
Lookahead slow-weight dynamics (Lemma, proof in app):
E[phi_{t+1}] = [1-alpha + alpha(I-gamma A)^k] E[phi_t].
V[phi_{t+1}] = [1-alpha+alpha(I-gamma A)^k]^2 V[phi_t] + alpha^2 sum_{i=0}^{k-1}(I-gamma A)^{2i} gamma^2 A^2 Sigma.
Key cov step: cov(theta_{t,k-1},theta_{t,k}) = (1-gamma a)V[theta_{t,k-1}]; generally cov(phi_t,theta_{t,k}) = (I-gamma A)^k V[phi_t].
V[phi] = (1-alpha)^2 V[phi] + alpha^2 V[theta_{t,k}] + 2alpha(1-alpha)cov(phi,theta_{t,k}).
Solve fixed point:
V*_LA = alpha^2 sum_{i=0}^{k-1}(I-gamma A)^{2i} / (I - [(1-alpha)I + alpha(I-gamma A)^k]^2) * gamma^2 A^2 Sigma.
Use sum_{i=0}^{k-1} a^i = (1-a^k)/(1-a) with a=(I-gamma A)^2:
V*_LA = [alpha^2(I-(I-gamma A)^{2k})] / [alpha^2(I-(I-gamma A)^{2k}) + 2alpha(1-alpha)(I-(I-gamma A)^k)] * V*_SGD.
First factor < 1 for alpha in (0,1) => Lookahead steady-state variance strictly smaller than SGD at SAME lr.
Tradeoff: expectation contraction is slower: compare 1-alpha+alpha(I-gamma A)^k vs (I-gamma A)^k; latter smaller for alpha<1. But for high-lr NN regime (variance-dominated, short-horizon bias, Wu 2018) variance term dominates -> faster effective convergence.

## Deterministic underdamped quadratic (app): linear dynamical system
f(x)=1/2 x^T A x. Stack fast weights, write transition = L B^{(k-1)} T with L=interpolation matrix, B=CM update, T=realign matrix. Eigenvalues bound rate rho; take k-th root because one slow step = k inner steps. Lookahead improves rate in the under-damped (oscillating) regime by "skipping" across oscillations; slightly worse over-damped.

## Inner-optimizer state choice (app): maintain / interpolate / reset momentum
All three beat SGD. CIFAR: maintain 95.15, interpolate 95.16, reset 94.91. Repo supports pullback_momentum in {none, reset, pullback}. "pullback" = interpolate momentum buffer same way as params. Default "none" (maintain).

## Load-bearing ancestors
- Polyak-Ruppert averaging (Ruppert 1988, Polyak-Juditsky 1992): average iterates for optimal asymptotic variance / statistical efficiency. But that's TAIL averaging at the END, arithmetic mean. Lookahead averages DURING training, EMA (recent-weighted). Martens 2014 noted EMA "works much better in practice" than Polyak average.
- SWA (Izmailov 2018, arXiv 1803.05407): equal-weight average of weights along SGD trajectory with cyclical/constant lr -> flatter wider optima, better generalization. Needs a START-averaging decision (too early/late hurts); arithmetic not EMA; produces a final model rather than driving optimization. Lookahead used from init, EMA, feeds back into the trajectory.
- Reptile (Nichol 2018): outer loop samples tasks, inner loop optimizes each, init updated toward new weights. Same outer/inner shape but for meta-learning across tasks, not single-task convergence.
- Katyusha (Allen-Zhu 2017) / SVRG (Johnson-Zhang 2013): outer/inner with pullback toward checkpoint EVERY inner step + SVRG variance-correction; convex guarantees but SVRG poor for NN (1812.04529). Lookahead pulls back only at END of inner loop, no SVRG correction.
- Anderson acceleration (Anderson 1965) / nonlinear extrapolation (Scieur 2018): keep ALL inner iterates, find best linear combination toward fixed point. Memory ~k x, and finding the combination is hard. Lookahead = simplest case: only first & last iterate, fixed alpha.
- hinton1987using: "fast weights" terminology origin.

## Design decisions -> why
- Two weight sets (fast/slow): orthogonal to adaptive+accelerated; lets any optimizer A be wrapped (one line of code).
- Interpolate toward final fast weight (not average all iterates): cheap (1 extra param copy, O((k+1)/k) ops), vs Anderson's O(k) memory.
- EMA (via repeated interpolation) not arithmetic: emphasizes recent proposals (Martens 2014), no start-decision (vs SWA).
- Pull back only at end of inner loop (not every step like Katyusha): lets fast weights actually explore/oscillate so the interpolation skips across oscillation.
- Reset fast weights to slow after each cycle: the slow weight is the committed state.
- alpha=0.8, k=5 defaults: robust; alpha=1 recovers inner optimizer; smaller alpha = more variance reduction but slower expectation.
- Maintain inner momentum by default: simplest, all variants beat SGD.

## Code structure (repo) -> scaffold mapping
class Lookahead(Optimizer): wraps inner optimizer.
__init__: store optimizer, la_step counter=0, la_alpha, total_la_steps; cache cached_params copy of each p; if pullback cache cached_mom.
step(closure): loss = optimizer.step(closure); la_step+=1; if la_step>=total: la_step=0; for each p: p.data.mul_(alpha).add_(cached_params, alpha=1-alpha) [crucial interpolation line]; cached_params.copy_(p.data); pullback/reset momentum handling.
_backup_and_load_cache / _clear_and_load_backup: swap to slow weights for eval.
