Full RHG confirmed what I suspected at the end of step 1: it did not separate from the truncated unroll. RHG landed at $84.63$ mean linear test accuracy and $84.79$ on the MLP, against T-RHG's $84.61$ and $84.79$ — identical within seed noise, same high-recall ($0.969$), middling-precision ($0.832$) cleaner profile, linear f1 $89.55$ — so the deep trajectory terms I spent five times the memory to include moved nothing. The cap is the message: in the converged limit RHG is implicit differentiation summed as a Neumann series, so its hypergradient is only as good as the inner solve underneath it, and that solve is $500$ gradient-descent steps run fresh from a random init every outer step. The $\sim 85\%$ ceiling is the price of solving the inner problem badly, not of differentiating it imperfectly. The move that could actually break it is to stop chasing the inner solution at all and instead fold lower-level optimality directly into one joint objective over $(x, y)$ that I descend with coupled first-order steps — never resetting, never unrolling.

I propose **G-PBGD, gradient-norm penalty-based bilevel gradient descent**, the cheapest member of the penalty family: the one with an exactly computable penalty gradient and no inner loop. The bilevel constraint is $y \in S(x) = \arg\min_y g(x, y)$, and the squared lower-level gradient norm $\|\nabla_y g(x, y)\|^2$ measures it — zero exactly on the argmin. It is irresistible here because its gradient is exactly computable with first-order autodiff: differentiating $\|\nabla_y g\|^2 = \nabla_y g\cdot\nabla_y g$ gives $2\,\nabla_{yy} g\cdot\nabla_y g$ in $y$ and $2\,\nabla_{xy} g\cdot\nabla_y g$ in $x$, both Hessian-vector products of the Hessian of $g$ against the already-computed vector $\nabla_y g$. I never materialize the Hessian: I compute $\nabla_y g$ keeping the graph (`create_graph=True`), form the scalar $f + \tfrac{\gamma}{2}\|\nabla_y g\|^2$, and one more backward hands me exactly $\gamma\,\nabla_{yy} g\cdot\nabla_y g$ and $\gamma\,\nabla_{xy} g\cdot\nabla_y g$ — the $\tfrac12$ cancelling the $2$ from the square. So the method is plain coupled gradient descent on
$$F_\gamma(x, y) = f(x, y) + \tfrac{\gamma}{2}\,\|\nabla_y g(x, y)\|^2,$$
the penalty gradient available exactly, no inner optimizer anywhere. Compare the value gap $g(x,y)-v(x)$, whose gradient needs $\nabla v(x)$ and hence knowledge of a lower solution $y^*(x)$ — exactly the object the unrolling rungs strained to approximate. After watching that family bottleneck on its inner solve, "no inner solve at all" is precisely the structural escape I want.

What makes the penalty principled is the squared-distance-bound framework, and it also exposes G-PBGD's signature fragility. Write the constraint as $d^2_{S(x)}(y) = 0$. A penalty $p$ is a $\rho$-squared-distance bound if $p\ge 0$, $\rho p \ge d^2_{S(x)}(y)$, and $p=0$ exactly on $S(x)$; under an $L$-Lipschitz $f$, the calmness inequality $\min_{z\ge 0}(-Lz + (\gamma/\rho)z^2) = -L^2\rho/(4\gamma)$ then makes globals of $f+\gamma p$ have penalty residual $O(1/\gamma)$ and locals $O(1/\gamma^2)$ — at a *finite* $\gamma = \Theta(\delta^{-1/2})$ for target residual $\delta$, not $\gamma\to\infty$. But the local bound for the gradient norm needs the singular values of $\nabla_{yy} g$ bounded below by some $\sigma>0$, because $\|\nabla_{yy} g\cdot\nabla_y g\| \ge \sigma\|\nabla_y g\|$ is what converts the controlled penalty gradient back into control on $\|\nabla_y g\|$. Wherever $\nabla_{yy} g$ degenerates, that conversion fails. The cleanest counterexample is one-dimensional: $\min f = \sin^2(y - 2\pi/3)$ subject to $y\in\arg\min g = y^2 + 2\sin^2 y$, whose only solution is $y=0$. At $y=2\pi/3$ the penalty gradient $2(y+\sin 2y)(1 + 2\cos 2y)$ vanishes for *every* $\gamma$ — $1 + 2\cos(4\pi/3) = 0$ — and $f$ is flat there, so $y=2\pi/3$ is a spurious stationary point of $F_\gamma$ at any strength. The decisive fact is that there $\nabla_{yy} g = 2 + 4\cos 2y = 0$: the degenerate Hessian kills the penalty gradient $\nabla_{yy} g\cdot\nabla_y g$ even though $\nabla_y g$ itself is nonzero. The gradient-norm penalty can stall wherever the lower Hessian goes singular — which is exactly why the value-gap penalty is the more robust sibling, since its local stationarity $\nabla_y f + \gamma\nabla_y g = 0$ controls $\|\nabla_y g\|\le L/\gamma$ directly with no $\nabla_{yy} g$ in the way. (In the SDB language this is one PL step weaker: the gradient norm needs $(1/\sqrt\mu)$-PL to chain twice, $\|\nabla_y g\|^2 \ge (1/\mu)(g-v)\ge (1/\mu^2)d^2$, where the value gap needs only $(1/\mu)$-PL.)

Two practical knobs fall out of the analysis and have to be wired into the helper, not invented. $F_\gamma$ is $L_\gamma$-smooth with $L_\gamma$ growing linearly in $\gamma$, so a stable step needs $\alpha \lesssim 1/L_\gamma$; a huge $\gamma$ from the start is a stiff landscape that forces a tiny step before $y$ reaches the lower valley. So I ramp $\gamma$ from $0$ to a finite cap over a fixed number of steps — essentially pure validation-loss descent at first, then phasing in the penalty. And once $\gamma > 1$ the penalty term dominates the gradient, so I rescale the whole joint step by $\min(1/\gamma, 1)$, keeping the effective penalty step order-one rather than order-$\gamma$. The scaffold's `run_g_pbgd` implements exactly this: it forms $g_{xy} = (\sigma(x)\cdot\text{CE}_{\text{train}}).\text{mean}()$, takes $d_{gdy} = \text{autograd.grad}(g_{xy}, \text{params}, \texttt{create\_graph=True})$, sets $\text{objective} = \min(1/(\gamma+\varepsilon), 1)\cdot(f_y + 0.5\,\gamma\,\|d_{gdy}\|^2)$, backprops, and steps $x_{\text{opt}}$ (lr `lrx`) and $y_{\text{opt}}$ (lr `lry`) with $\gamma$ annealed linearly from `gamma_init` to `gamma_max` over `gamma_argmax_step`. So my edit points `algorithm` at `run_g_pbgd` and sets the schedule. Critically — and this is where this task's G-PBGD departs from a generic gradient-norm penalty — the helper does not use the exposed `inner_grad/outer_grad` callables either; it builds the penalty gradient directly from the model and the $\sigma(x)$ weights with a retained graph, so the HVP is exact.

I take the aggressive penalty schedule the reference uses: linear `lrx=0.3, lry=0.5, gamma_max=37.0, gamma_argmax_step=5000, outer_itr=40000`; MLP `lrx=0.5, lry=0.5, gamma_max=37.0, gamma_argmax_step=30000, outer_itr=50000`. The large `gamma_max=37` is the $\Theta(\delta^{-1/2})$ regime pushed hard, so the gradient-norm penalty pins $y$ tightly onto the lower-stationarity manifold; the $\min(1/\gamma,1)$ rescale is what keeps that stable. There is no `lr_inner`, no `T`, no `K` — no inner loop at all, which is the whole point of escaping the unrolling ceiling.

The toy is where the fragility becomes measurable, and I predict it honestly. Unlike RHG and T-RHG, whose toy mode dispatched to the value-gap projected step, `run_g_pbgd` in toy mode dispatches to `_toy_pbgd_step("g_pbgd")`, which uses `toy_gpbgd_penalty_grad` — the gradient of $\|\nabla_y g\|^2$ — instead of $\nabla g$. So G-PBGD descends a different, stiffer toy objective: near $S(x)=\{-x\}$ the gradient-norm landscape is sharper and its curvature factor $\nabla_{yy} g$ modulates the step, so I expect more steps to hit the stationarity tolerance — the count rising from the shared $260.7$ to roughly $300$, with a larger residual ($\sim 0.06$–$0.10$ rather than $0.030$). This is not a regression I am introducing by mistake; it is the same $\nabla_{yy} g$-sensitivity that made $y=2\pi/3$ spurious. On hyper-cleaning I expect G-PBGD to break the unrolling ceiling decisively — MLP accuracy from RHG's $84.79$ into the low 90s (around $92$), linear from $84.63$ to about $90$ — but at a cost on the cleaner f1, because the aggressive penalty drives precision up at the expense of recall, so I expect linear f1 to drop below the unrolling rungs (from $\sim 89.5$ toward $\sim 80$) with cleaner recall near $0.78$. If G-PBGD instead matched the unrolling toy numbers the toy dispatch would not be using the gradient-norm penalty; if its MLP accuracy did not break $90$, the penalty escape would have failed and the bottleneck would not have been the inner solve after all.

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.3,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 5_000,
        "outer_itr": 40_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.5,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 30_000,
        "outer_itr": 50_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_g_pbgd(state, hparams, grad_fns)
```
