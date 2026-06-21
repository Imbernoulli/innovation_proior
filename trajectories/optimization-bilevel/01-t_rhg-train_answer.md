The starting point is the scaffold default: one function, `algorithm(state, hparams, grad_fns)`, has to serve both the toy convergence problem and the MNIST data hyper-cleaning problem at once, and I want the first rung to be the cheapest update that is *unambiguously correct* as a bilevel solver, so its measured behavior sets the floor the harder rungs must beat. The most conservative thing I can write down makes the fewest assumptions about the inner objective: differentiate the actual inner training run rather than assume it is strongly convex or invert any Hessian. That is the reverse-mode hypergradient, and its truncated form is the natural first rung, because it is the same algorithm at a smaller, cheaper memory footprint.

I propose **T-RHG, the truncated reverse-mode hyper-gradient**. The object I actually want to descend is $F(x) = f(x, y_T(x))$, where in hyper-cleaning $x$ is a vector of per-example logits (the weight on training point $i$ is $\sigma(x_i) \in (0,1)$), $y$ is the classifier's parameters, and $y_T(x)$ is not a formula but the endpoint of $T$ steps of inner gradient descent on the $\sigma(x)$-weighted training loss. Its total derivative is
$$d_x F = \nabla_x f + (d y_T/d x)^\top \nabla_y f,$$
and the two partials are one backward pass each; the entire difficulty is the middle factor $d y_T/d x$, the sensitivity of the inner solution to the per-example weights. To compute it I make the inner optimizer explicit as a dynamical system, $y_{t+1} = \Phi_{t+1}(y_t, x) = y_t - \eta_{\text{in}}\,\nabla_y g(y_t, x)$, so that $y_T$ is a composition of $T$ such maps. Differentiating the recursion totally in $x$ gives the hypergradient as a sum over the whole trajectory,
$$d_x F = \nabla_x f + \sum_t B_t\, A_{t+1} A_{t+2}\cdots A_T\, \nabla_y f,\qquad A_t = I - \eta_{\text{in}}\nabla_{yy} g,\quad B_t = -\eta_{\text{in}}\nabla_{xy} g,$$
where $A_t$ is how step $t$ reacts to a perturbation of the *state* and $B_t$ how it reacts to a perturbation of $x$ directly.

What makes this affordable is the order in which the chain-rule product is contracted. The $A_t$ are parameter-by-parameter matrices — for the MLP, hundreds of thousands of dimensions square — so I never materialize them; I only ever need their action on a vector. If I push the row $\nabla_y f$ leftward through the chain, I carry a single adjoint vector the size of $y$, and each $\alpha\cdot A_t$ and $\alpha\cdot B_t$ is one transposed-Jacobian-vector product that autograd gives for the price of one inner step. This is the adjoint recursion — set $\alpha_T = \nabla_y f$, sweep $t$ downward, accumulate $h \mathrel{+}= \alpha_t\cdot B_t$ and propagate $\alpha_{t-1} = \alpha_t\cdot A_t$ — i.e. back-propagation through time over the optimizer's own steps, costing $O(T)$ time *independent of* $\dim(x)$. That independence is decisive here, because $x$ has one coordinate per training example: the complementary forward contraction would carry a $\dim(y)\times\dim(x)$ sensitivity matrix and run $\dim(x)$ times slower, which with $x$ this wide is hopeless. Reverse mode is the only sane direction.

The one cost the reverse contraction cannot escape, and the reason I fill the *truncated* variant rather than the full unroll, is memory. To evaluate $A_t$ and $B_t$ on the way back I need the state $y_{t-1}$ I was at when I took step $t$; the backward sweep visits states in reverse, so naively I must hold the entire trajectory, $O(T\cdot\dim(y))$. The way out keeps the estimator exact in spirit and optimizer-agnostic — no fragile reversal of the dynamics, no information-buffer tricks. When the last inner iterates sit where $g(\cdot, x)$ is well-conditioned (locally strongly convex with $\eta_{\text{in}}$ below the smoothness reciprocal), each $A_t$ is a contraction, $\|A_t\| \le 1 - \eta_{\text{in}}\alpha < 1$, so the term at depth $T-t$ carries a factor bounded by $(1-\eta_{\text{in}}\alpha)^{T-t}$ that decays geometrically the further back I go. The deep terms contribute almost nothing. So I run the inner optimizer forward for the full $T$ steps — the final weights must be properly converged — but in the backward pass I walk back only $K$ transitions and stop. The truncated estimate
$$h_{T-K} = \nabla_x f + \sum_{t=T-K+1}^{T} B_t\, A_{t+1}\cdots A_T\, \nabla_y f$$
has bias on the order of $(1-\eta_{\text{in}}\alpha)^K$ — geometrically small in $K$ — at memory $O(K\cdot\dim(y))$ instead of $O(T\cdot\dim(y))$. Full reverse-mode unrolling is exactly the $K=T$ special case; truncation is the same algorithm at a fraction of the memory.

There is a satisfying consistency check that this truncation is not a crude hack. In the converged limit the per-step Jacobians settle to constants $A_\infty = I - \eta_{\text{in}}\nabla_{yy} g(y^*)$ and $B_\infty = -\eta_{\text{in}}\nabla_{xy} g(y^*)$, and the full backward accumulation becomes $\nabla_y f\cdot\big(\sum_k A_\infty^k\big)\cdot B_\infty$. But $\sum_k A_\infty^k$ is the Neumann series of $(I-A_\infty)^{-1} = (\eta_{\text{in}}\nabla_{yy} g)^{-1}$, so the infinite backward sum equals the implicit-function hypergradient $-\nabla_y f\,(\nabla_{yy} g)^{-1}\,\nabla_{xy} g$. The $K$-step truncation is precisely keeping the first $K$ terms of that Neumann series, so $K$ is the order of the inverse-Hessian approximation: this rung is quietly implicit differentiation computed by summing instead of solving, and truncating trades a controllable bias for memory.

Mapping this onto the harness pins down what the code can do. The scaffold's `run_rhg_family` already implements the entire machinery: in hyper-cleaning mode it runs the forward inner loop of $T$ steps keeping the last $K+1$ states, calls the fixed `hg.reverse(...)` adjoint sweep over `[fp_map] * K`, and steps the upper optimizer; the truncation knob is literally the hyperparameter `K`. So filling this rung is choosing $K$, $T$, $\text{lr}$, $\eta_{\text{in}}$ and pointing `algorithm` at the helper. One harness fact I must respect: although `grad_fns` also exposes `outer_grad`, `inner_grad`, `inner_val`, the reverse-mode family does *not* consume them — the adjoint needs the inner step to be a graph node, which it builds itself with `create_graph=True` inside the helper, while the pre-detached `inner_grad` callable is not designed to give that. Those callables are affordances for the penalty rungs to come. A second harness fact fixes toy behavior: there `run_rhg_family` dispatches to the same projected penalized step as the value-gap method, because $S(x)=\{-x\}$ makes $v(x)=0$ and there is nothing to unroll — the "inner solve" is trivial — so my `TOY_HPARAMS` are the shared `gams=(10.0,), alpha0=0.1` and I expect the toy numbers (around $260$ mean steps at full success, residual near $0.03$) to coincide with the other rungs. T-RHG can only distinguish itself on hyper-cleaning, and only through $K$.

That leaves the choice of $K$, which is this rung's whole identity. I take $T=500$ inner steps, $K=100$ (back-propagate through the last fifth of the trajectory), $\text{lr}=0.001$ outer, $\eta_{\text{in}}=0.1$ linear and $0.4$ MLP, with $\text{outer\_itr}=100$ and $\text{eval\_interval}=1$. The contraction argument bounds the bias from dropping the first $400$ transitions by $(1-\eta_{\text{in}}\alpha)^{100}$ — negligible if the inner problem is even mildly well-conditioned — while memory drops fivefold against the full unroll. The non-interference structure sharpens it: the validation loss depends on $x$ only through $y$, so $\nabla_x f = 0$ and the truncated adjoint is, for the right problem class, essentially as good a descent direction as the full one. I expect $K=100$ to track full unrolling closely; if it did not, the contraction assumption would be failing and the geometry would be telling me the inner problem is not well-conditioned. On hyper-cleaning I expect this differentiate-a-500-step-run family to land in the mid-80s rather than the 90s — a crude inner solve, not a wrong derivative — with high cleaner recall and only middling precision. This rung's job is to set that floor; the next removes the truncation and asks whether the missing 400 transitions were worth their memory.

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
        "K": 100,
        "reg": 0.0,
        "eval_interval": 1,
    },
    "mlp": {
        "lr": 0.001,
        "lr_inner": 0.4,
        "outer_itr": 100,
        "T": 500,
        "K": 100,
        "reg": 0.0,
        "eval_interval": 1,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_rhg_family(state, hparams, grad_fns)
```
