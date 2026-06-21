The truncated unroll told me something by *not* failing. T-RHG, back-propagating through only the last $K=100$ of $T=500$ inner steps, landed at $84.61$ mean linear test accuracy ($84.10/84.82/84.92$ across seeds) and $84.79$ on the hidden MLP, with linear f1 $89.06$ and a high cleaner recall of $0.964$ against a middling precision of $0.828$; on the toy it sat at exactly $260.7$ mean convergence steps at full success and $0.030$ residual, the shared projected step. So the truncation bias I worried about did not show up as a deficit. The one thing those numbers leave open is *why*: either $K=100$ is already deep enough that the dropped $400$ transitions were geometrically negligible, or the inner problem is well-enough conditioned that even a short adjoint is a good descent direction — and the only way to settle it, and to check whether the unrolling family can do better at all by spending the memory I saved, is to remove the truncation and back-propagate through the whole inner trajectory.

I propose **RHG, the full reverse-mode hyper-gradient** — the exact gradient of the $T$-step inner run, which is the $K=T$ special case of exactly the method I just ran. The exact hypergradient is the full trajectory sum
$$d_x F = \nabla_x f + \sum_{t=0}^{T} B_t\, A_{t+1}\cdots A_T\, \nabla_y f,\qquad A_t = I - \eta_{\text{in}}\nabla_{yy} g,\quad B_t = -\eta_{\text{in}}\nabla_{xy} g,$$
with $A_t$ the state Jacobian of the inner gradient-descent map and $B_t$ its direct dependence on $x$. Truncation kept only the last $K$ of those $T+1$ terms; full RHG keeps all of them, and the terms it adds back are precisely the deep ones, indexed $t \le T-K$, each carrying a product of more than $K$ contraction matrices $A_{t+1}\cdots A_T$ and so bounded in size by $(1-\eta_{\text{in}}\alpha)^{T-t}\le(1-\eta_{\text{in}}\alpha)^{K}$. In other words, the extra terms full RHG sums are exactly the ones T-RHG's bias bound called negligible: if the inner map genuinely contracts, adding them should move the hypergradient, and therefore the accuracy, almost not at all. The $84.6/84.8$ result is my baseline expectation for what RHG reproduces.

To be sure full unrolling really is the truncated sweep with $K=T$, the Lagrangian view makes the adjoint clean. Minimize $f(x, y_T)$ subject to the $T$ constraints $y_t = \Phi_t(y_{t-1}, x)$, attach a row multiplier $\alpha_t$ to each, and stationarity gives the terminal condition $\alpha_T = \nabla_y f$, the backward recursion $\alpha_{t-1} = \alpha_t\cdot A_t$, and the gradient $d_x F = \nabla_x f + \sum_t \alpha_t\cdot B_t$. The multiplier $\alpha_t$ is the adjoint state — the sensitivity of the final validation loss to a perturbation of the inner iterate at time $t$ — carried from the end of training to the beginning, every operation a transposed-Jacobian-vector product that forms no matrix. Setting $K=T$ means the backward loop runs from $T$ down to $1$ over the *entire* stored trajectory rather than the last hundred states; the recursion, the accumulation, and the per-step cost are identical — only the window length, and therefore the memory, change. This is why I can say with confidence that RHG and T-RHG are the same algorithm differing in one integer.

The harness makes that literally true: `run_rhg_family` reads $K$ from the hyperparameters and runs the forward inner loop keeping the suffix of $K+1$ states, then calls the fixed `hg.reverse(..., [fp_map] * K, ...)` adjoint sweep. To go from truncated to full I set $K = T = 500$; the helper then stores all $500$ transitions and back-propagates through every one. Everything else — $T=500$, $\text{lr}=0.001$ outer, $\eta_{\text{in}}=0.1$ linear and $0.4$ MLP, $\text{outer\_itr}=100$, $\text{eval\_interval}=1$ — I keep identical to step 1, because the only variable I am changing is whether the deep trajectory terms are included. As before, this rung ignores the exposed `outer_grad/inner_grad/inner_val` callables: the adjoint needs the inner step built with `create_graph=True` as a graph node, which the helper constructs internally in its own `fp_map`.

I should be honest about the cost I am paying for exactness, because it is the structural weakness this whole family inherits and the reason the next rung looks elsewhere. Full RHG stores the entire inner trajectory, $O(T\cdot\dim(y))$ memory — for the MLP, $500$ copies of the parameter vector, which is why the truncated variant existed in the first place. I spend that memory here only to measure whether it was worth spending, and step 1 already strongly suggests it was not. The deeper reconciliation tells me where this family's ceiling sits: in the converged limit the full backward sum is the Neumann series of the inverse Hessian, so RHG with $K=T$ is implicit differentiation computed by summation, and its hypergradient is only ever as good as the inner solve underneath it. Here that inner solve is $500$ gradient-descent steps at a fixed $\eta_{\text{in}}$, run fresh from a random init every outer step — a crude inner optimizer, differentiated before it has really converged. So I expect the unrolling family, full or truncated, to be capped well below the penalty methods not because the differentiation is wrong but because the inner problem is barely solved before I differentiate it.

The toy is decoupled from all of this, as in step 1: `run_rhg_family` in toy mode dispatches to the same `_toy_pbgd_step` as the value-gap method — $S(x)=\{-x\}$, $v(x)=0$, nothing to unroll, one projected penalized step on $f + \gamma\,\nabla g$ with `gams=(10.0,)`, `alpha0=0.1`. So RHG's toy line should be identical to T-RHG's: $260.7$ mean steps, success $1.0$, residual $0.030$; any difference there would be a wiring bug, not a property of unrolling. On hyper-cleaning I expect RHG to land at or just above T-RHG — linear accuracy around $84.6$, MLP around $84.8$, linear f1 near $89.5$, the same high-recall, middling-precision profile. If it matches within seed noise, the truncation was free and this family's ceiling on this task is $\sim 85\%$, set by the $500$-step inner solve; if it instead jumps several points, the deep terms mattered and the inner problem is poorly conditioned. Either way, the gap to the penalty methods is the real story — the gap between differentiating a crude inner solve and folding lower-level optimality into the objective and descending it directly.

```python
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lr": 0.001,
        "lr_inner": 0.1,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
    "mlp": {
        "lr": 0.001,
        "lr_inner": 0.4,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_rhg_family(state, hparams, grad_fns)
```
