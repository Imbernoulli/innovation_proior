G-PBGD broke the unrolling ceiling exactly where I expected and broke something else exactly where I warned it would. On the hidden MLP it jumped to $92.38$ test accuracy — eight points above RHG's $84.79$ — confirming the bottleneck was never the differentiation but the crude $500$-step inner solve; linear accuracy rose to $89.84$. But the cleaner f1 on the linear model *collapsed* to $80.63$ mean, with an enormous seed spread ($76.67/81.69/83.52$) against the unrolling family's stable $\sim 89.5$, and the cause is visible in the profile: precision held at $0.839$ but recall fell to $0.776$, so the aggressive $\gamma_{\max}=37$ penalty threw away too many genuinely clean examples to keep its lower-stationarity small. The toy also regressed as predicted, to $303.7$ mean steps at $0.081$ residual, because it dispatched to the gradient-norm penalty's stiffer landscape. The diagnosis is the one I rehearsed in the counterexample: the gradient-norm penalty's local stationarity only controls $\nabla_{yy} g\cdot\nabla_y g$, so it is hostage to the lower-level curvature $\nabla_{yy} g$ — which is what makes both its f1 unstable and its toy residual coarse. The fix is the robust member of the penalty family, whose local stationarity controls $\nabla_y g$ directly with no curvature factor in the way.

I propose **V-PBGD, value-function penalty-based bilevel gradient descent**: penalize the lower-level value gap $p = g(x, y) - v(x)$ with $v(x) = \min_y g(x, y)$, and descend $F_\gamma = f + \gamma p$. The single structural difference from the gradient norm is where the Hessian appears in the local relation. At a local solution of $F_\gamma$, stationarity in $y$ is
$$\nabla_y f + \gamma\,\nabla_y g = 0,$$
because $v$ carries no $y$-dependence at all — so $\|\nabla_y g\| \le L/\gamma$ *directly*, no $\nabla_{yy} g$, no singular-value condition. Under the lower level being $(1/\mu)$-PL, the error bound $g - v \ge (1/\mu)\,d^2$ then gives $p \le \mu L^2/\gamma^2 = O(1/\gamma^2)$ with one PL chain, so the value gap is a $\mu$-squared-distance bound under *weaker* assumptions than the gradient norm needed ($1/\sqrt\mu$-PL). This is the whole story of step 3's failure inverted: the gradient norm's bound only controls $\|\nabla_y g\|$ after dividing by the smallest singular value of $\nabla_{yy} g$, which vanishes wherever the lower Hessian degenerates — and the linear hyper-cleaner's cross-entropy has exactly such flat directions, which is why its f1 swung seven points across seeds. The value gap has no such division, so it does not get whipsawed by lower-level curvature, and I expect its cleaner f1 to be both higher and far more stable.

The value gap is a faithful penalty at the global level too, by the squared-distance-bound argument. Take any feasible $(x, y)$, project $y$ to $y_x \in S(x)$ so $d = \|y_x - y\|$, and use $L$-Lipschitz $f$: $f(x, y) - f(x, y_x) \ge -L d$. Add $\gamma p$ with $p \ge d^2/\mu$ and the whole expression is at least $-L d + (\gamma/\mu)d^2$, a scalar quadratic in $d \ge 0$ whose minimum over $d$ is $-L^2\mu/(4\gamma)$. Setting $\gamma^* = L^2\mu/(4\varepsilon_1)$ floors it at $-\varepsilon_1$, so any bilevel global solution (where $p=0$) is an $\varepsilon_1$-global-min of the penalized problem; conversely an $\varepsilon_2$-global-min with $\gamma > \gamma^*$ has $p \le (\varepsilon_1+\varepsilon_2)/(\gamma-\gamma^*)$ — residual $O(1/\gamma)$ globally, $O(1/\gamma^2)$ locally. The lever is the quadratic $(\gamma/\mu)d^2$, present only because $p$ *dominates* $d^2$, overpowering the linear Lipschitz slack $-Ld$ once $\gamma$ clears the threshold. And $\gamma = \Theta(\delta^{-1/2})$ is genuinely tight: on $f = y$, $g = y^2$ the penalized minimizer is $y = -1/(2\gamma)$ with gap $1/(4\gamma^2)$, so forcing the gap below $\delta$ requires $\gamma \ge 1/(2\sqrt\delta)$. A *finite* $\gamma$ suffices for any finite accuracy — which is exactly why this rung can use $\gamma_{\max} = 0.1$–$0.2$ rather than the huge value the gradient norm needed, and that gentleness is what protects the cleaner's recall.

The one subtlety the value gap introduces is computing $\nabla v(x)$, and the PL Danskin lemma dissolves it. Naively $\nabla v$ needs a lower solution $y^*(x)$ and looks like the implicit machinery I have been avoiding: $v(x) = g(x, y^*(x))$, so $\nabla v = \nabla_x g(x, y^*) + (dy^*/dx)^\top\nabla_y g(x, y^*)$. But at an unconstrained lower minimum $\nabla_y g(x, y^*) = 0$, so the scary implicit term multiplies by zero and $\nabla v(x) = \nabla_x g(x, y^*)$ for *any* $y^* \in S(x)$ — no Hessian, no implicit-function theorem, no strong convexity. So I only need an *approximate* lower minimizer, which I estimate with a short inner loop, warm-started so the inner gap stays tied to outer progress. The warm start is load-bearing for a reason the analysis pins down: inner gradient descent contracts the value gap geometrically under PL, $g(x, \omega_{t+1}) - v \le (1 - \beta/(2\mu))(g(x, \omega_t) - v)$, but that bound carries the *initial* gap $g(x_k, \omega_1) - v(x_k)$, and $x_k$ drifts across outer iterations — a cold start at an unrelated point could let the initial gap grow without bound, so no fixed inner-step count would control the estimation error. Warm-starting $\omega_1 = y_k$ ties the initial inner gap to the outer step size through PL ($g(x_k, y_k) - v \le (1/\mu)\|\nabla_y g\|^2$, and $\nabla_y g$ is controlled by the outer move), so a logarithmically short inner loop keeps the estimation error summable and the outer projected descent telescopes to a $1/K$ stationarity rate. This is why a *single* warm-started inner step ($\text{inner\_itr} = 1$) is enough here: because the inner network persists and is never reset, each outer iteration only has to close the small incremental gap the last outer move opened. The value-gap direction is then
$$h = \nabla g(x, y) - \big(\nabla_x g(x, \hat y),\,0\big),$$
the subtraction living only in the $x$-coordinates, because $v$ depends only on $x$ — so in $y$ the value-gap gradient is just $\nabla_y g(x, y)$.

This is where V-PBGD is concrete and where I match the harness, not the generic method. The scaffold's `run_v_pbgd` keeps a *persistent auxiliary* inner network `net_inner`, initialized from the main model and updated across outer iterations by `inner_itr` SGD steps on the $\sigma(x)$-weighted training loss at frozen weights $\text{sigx} = \sigma(x).\text{detach}()$. It then forms $f_y = \text{CE}(\text{val}, \text{clean})$, $g_{xy} = (\sigma(x)\cdot\text{CE}_{\text{train}}).\text{mean}()$ through the *main* model, and $v_x = (\sigma(x)\cdot\text{CE}_{\text{inner}}).\text{mean}()$ through `net_inner` with the inner outputs *detached* — so the gradient through $v_x$ flows only via $\sigma(x)$, giving exactly $\nabla_x g(x, \hat y)$ and nothing through $\hat y$. The objective is $\min(1/(\gamma+\varepsilon), 1)\cdot(f_y + \gamma(g_{xy} - v_x))$, backpropped to step both $x_{\text{opt}}$ and $y_{\text{opt}}$, with $\gamma$ ramped linearly from $0$ to `gamma_max`. So the value-gap subtraction $g_{xy} - v_x$ is realized by the detach pattern, and the $\min(1/\gamma, 1)$ rescale is the same stabilizer as G-PBGD. My edit points `algorithm` at `run_v_pbgd` and sets the schedule. Note this rung *does* run a real inner loop — unlike G-PBGD, which had none — but it is a single warm-started step per outer iteration, not the $500$-step-from-scratch unroll the RHG family paid for; that is the distinction between approximating $\nabla v$ cheaply and differentiating a full inner solve. As in every prior rung, it ignores the exposed `inner_grad/outer_grad/inner_val` callables, building the gradients directly from the two networks and $\sigma(x)$.

The hyperparameters are the gentle schedule the reference uses, and the contrast with G-PBGD's $\gamma_{\max}=37$ is itself diagnostic: linear `lrx=lry=0.1, lr_inner=0.01, inner_itr=1, gamma_max=0.2, gamma_argmax_step=30000, outer_itr=40000`; MLP `lrx=0.1, lry=0.01, lr_inner=0.01, inner_itr=1, gamma_max=0.1, gamma_argmax_step=10000, outer_itr=80000`. The tiny $\gamma_{\max}$ is possible *because* the value gap is the more faithful surrogate — its $O(1/\gamma^2)$ local residual is achieved at modest $\gamma$, so it need not be slammed to a huge strength to pin $y$, and a gentle penalty is exactly what keeps the cleaner's recall from collapsing the way G-PBGD's did; the longer MLP horizon (80k outer steps) lets the gentle penalty converge fully. The toy, finally, returns to the well-behaved regime: `run_v_pbgd` in toy mode dispatches to `_toy_pbgd_step("v_pbgd")`, which uses $\nabla g$ (the value-gap gradient, since $S(x)=\{-x\}$ makes $v(x)=0$ and $p=g$), not the gradient-norm penalty, so V-PBGD descends the same smooth landscape the unrolling rungs did and the toy line should snap back to $260.7$ mean steps at $0.030$ residual.

So my falsifiable expectations against the measured numbers, since this is meant to be the strongest rung. On the toy I expect V-PBGD to recover the unrolling family's $260.7$ steps and $0.030$ residual, beating G-PBGD's $303.7/0.081$ and fixing the toy regression. On hyper-cleaning I expect it to keep the penalty family's accuracy win over the unrolling family while *fixing the f1 collapse*: linear accuracy around $90$ (above the unrolling $84.6$, comparable to G-PBGD), but with linear f1 back up around $91.7$ — eleven points above G-PBGD's fragile $80.63$ — and a stable cleaner recall near $0.95$ rather than $0.78$. On the hidden MLP I expect $\sim 91.5$, just below G-PBGD's $92.38$: the gentle value-gap penalty trades roughly a point of raw MLP accuracy for a far more faithful, lower-variance cleaner and a recovered toy convergence. That is the trade I am willing to make, and it is why the value gap is the robust default — it wins the toy primary, the linear primary, and both f1 scores decisively, losing only the single hidden MLP accuracy by about a point. If the value gap instead matched G-PBGD's broken f1, the detach pattern would not be realizing the $g_{xy} - v_x$ value gap and I would have wired the wrong subtraction.

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.1,
        "lry": 0.1,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.2,
        "gamma_argmax_step": 30_000,
        "outer_itr": 40_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.1,
        "lry": 0.01,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.1,
        "gamma_argmax_step": 10_000,
        "outer_itr": 80_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_v_pbgd(state, hparams, grad_fns)
```
