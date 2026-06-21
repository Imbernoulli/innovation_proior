We want to solve the smooth stochastic minimax problem $\min_x \max_y f(x,y)$, with $f(x,y) = \mathbb{E}_\xi[f(x,y;\xi)]$ accessed only through an unbiased first-order oracle of bounded variance $\sigma^2$, and we want a near-stationary point: a $z = (x,y)$ at which the gradient operator $F(z) = [\nabla_x f ; -\nabla_y f]$ has small norm, $\|F(z)\| \le \epsilon$, found in as few oracle calls as possible. The target is deliberately the gradient norm and not the duality gap, because the gap is a trap — on the canonical bilinear instance $f(x,y) = x^\top y$ the inner $\max_{y'} f(x,y')$ blows up the moment $x \neq 0$, so the gap is $+\infty$ everywhere except the saddle and cannot even rank two bad points, whereas $\|F(z)\|$ is always finite, always defined, and directly measurable. The only structure I can lean on is operator monotonicity: smooth convex-concave (CC) makes $F$ monotone, strongly-convex-strongly-concave (SCSC) with parameter $\lambda$ makes $F$ $\lambda$-strongly monotone, and smoothness makes $F$ $L$-Lipschitz. The complexity I am chasing is one in which the statistical part driven by $\sigma^2$ and the optimization part driven by $L$, the condition number $\kappa$, and the initial distance $D = \|z_0 - z^*\|$ appear *additively* rather than multiplied together, i.e. $\tilde O(\sigma^2 \epsilon^{-2} + \kappa)$ in the SCSC case and $\tilde O(\sigma^2 \epsilon^{-2} + LD\epsilon^{-1})$ in the CC case.

The existing options each fall short in the same revealing way. Naive simultaneous stochastic gradient descent-ascent simply diverges on $x^\top y$, spiraling outward; the standard cure is the extragradient look-ahead, taking $z_{t+1/2} = z_t - \eta F(z_t;\xi)$ then $z_{t+1} = z_t - \eta F(z_{t+1/2};\xi)$, which cancels the rotational part and converges. But stochastic extragradient (SEG) with a fixed stepsize satisfies, for SCSC $f$ with $\eta < 1/(4L)$, the averaged bound $\mathbb{E}\|\bar z - z^*\|^2 \le \|z_0 - z^*\|^2/(\lambda\eta T) + 16\eta\sigma^2/\lambda$: the optimization term decays like $1/T$ but the noise term is a constant floor $16\eta\sigma^2/\lambda$, so SEG marches to a $\sigma^2$-sized ball and stalls. Shrinking $\eta$ shrinks the ball but cripples the $1/(\lambda\eta T)$ term, and for the gradient norm this costs $O(\sigma^2 L^2 \epsilon^{-4} + L^2 D^2 \epsilon^{-2})$ — two factors of $\epsilon$ above the statistical floor $\sigma^2\epsilon^{-2}$. Anchoring (Halpern-style extragradient, pulling each iterate back toward the fixed start $z_0$ with weight $1/(k+2)$) achieves the optimal *deterministic* squared-gradient-norm rate $O(L^2 D^2/k^2)$, but stochastically it still pays $O(\sigma^2 L^2 \epsilon^{-4} + LD\epsilon^{-1})$ — the deterministic side is fixed, the noise side untouched. Regularized SEG, which runs SEG on a fixed surrogate $g = f + \tfrac{\lambda}{2}\|x-x_0\|^2 - \tfrac{\lambda}{2}\|y-y_0\|^2$, gains strong monotonicity but only with $\lambda = O(\epsilon/D)$, too small to help. The recurring diagnosis is the same: the methods that make the gradient small pay $\epsilon^{-4}$ under noise, and the regularized ones that exploit strong monotonicity are held back by a regularization strength that has to stay tiny.

I propose RAIN — Recursive Anchored IteratioN — and the way to see why it works is to first ask precisely *why* anchoring fails on the noise. Adding the penalty $\tfrac{\lambda}{2}\|x-x_0\|^2 - \tfrac{\lambda}{2}\|y-y_0\|^2$ turns the operator into $G(z) = F(z) + \lambda(z - z_0)$, which is $\lambda$-strongly monotone even when $F$ is merely monotone — strong monotonicity bought for free, exactly the property SEG needs to escape the noise floor. The problem is the *bias*. Let $z_g^*$ be the root of $G$. Then $\|F(z_g^*)\| = \|G(z_g^*) - \lambda(z_g^* - z_0)\| = \lambda\|z_g^* - z_0\|$, so even the exact anchored solution has residual gradient $\lambda\|z_g^* - z_0\|$, forcing $\lambda = O(\epsilon/D)$. More generally, for any candidate $\tilde z$ I can bound $\|F(\tilde z)\| \le \|G(\tilde z)\| + \lambda\|\tilde z - z_0\|$, and using $\lambda$-strong monotonicity of $G$ (which gives $\lambda\|\tilde z - z_g^*\| \le \|G(\tilde z)\|$) plus the triangle inequality through $z_g^*$ yields the **anchoring lemma**

$$\|F(\tilde z)\| \le 2\|G(\tilde z)\| + \lambda\|z_0 - z^*\|.$$

The price of anchoring at $z_0$ is exactly $\lambda\|z_0 - z^*\|$: strength and bias are coupled through the distance from the anchor to $z^*$, and since $z_0$ is generically far from $z^*$, $\lambda$ must stay tiny and the strong monotonicity I bought is worth almost nothing.

The escape is a fact I can prove for free. Since $G(z_g^*) = 0$, strong monotonicity between $z_g^*$ and $z^*$ gives $\lambda\|z_g^* - z^*\|^2 \le (G(z_g^*) - G(z^*))^\top(z_g^* - z^*) = G(z^*)^\top(z^* - z_g^*)$, and because $F(z^*) = 0$ we have $G(z^*) = \lambda(z^* - z_0)$, so $\|z_g^* - z^*\|^2 \le (z^* - z_0)^\top(z^* - z_g^*)$. Expanding the inner product by polarization, $(z^*-z_0)^\top(z^*-z_g^*) = \tfrac12(\|z_g^*-z^*\|^2 + \|z^*-z_0\|^2 - \|z_g^*-z_0\|^2)$, and rearranging gives the **non-expansiveness lemma**: $\|z_g^* - z_0\| \le \|z^* - z_0\|$ and, crucially, $\|z_g^* - z^*\| \le \|z^* - z_0\|$. The anchored solution lands *closer* to $z^*$ than the anchor was. So I should not anchor once. I should solve the anchored problem approximately, get a point closer to $z^*$, re-anchor *there*, and — because the new anchor is closer — raise $\lambda$, which makes the next subproblem more strongly monotone, better conditioned, and cheaper to solve under noise. A chain of warm restarts, each with a closer anchor and a larger penalty. This same escape exists in convex single-player minimization, but that argument rests on $\min_x f(x) \le f(x)$, an artifact of scalar convexity with no saddle counterpart; so I redo the entire recursion using only operator monotonicity, never touching function values.

Concretely, define the recursively regularized operator after $s$ anchors as $F^{(s)}(z) = F(z) + \sum_{i=1}^s \lambda_i (z - z_i)$, with anchors $z_i$ re-set to the approximate subproblem solutions and strengths growing geometrically: $\lambda_0 = \lambda\gamma$ and $\lambda_{s+1} = (1+\gamma)\lambda_s$. Taking $\gamma = 1$ for the analysis, the added penalties are $\lambda_i = \lambda 2^i$, so $F^{(s)}(z) = F(z) + \lambda\sum_{i=1}^s 2^i (z - z_i)$, run for $S = \lfloor\log_2(L/\lambda)\rfloor$ rounds. Then $F^{(s)}$ is at least $2^s\lambda$-strongly monotone (the added strengths sum past $\lambda 2^s$) and at most $2L$-Lipschitz (the accumulated penalties cap at $\lambda 2^S \le L$), so every subproblem stays $O(L)$-smooth while its condition number $2L/(2^s\lambda)$ shrinks toward $O(1)$ — later subproblems get *cheaper* to solve accurately, not harder. The central guarantee is the **recursive anchoring lemma**: writing $z_s^*$ for the exact root of subproblem $s$, peeling the last penalty off $F(z_S) = F^{(S-1)}(z_S) - \lambda\sum_{i=1}^{S-1}2^i(z_S - z_i)$, absorbing the look-ahead distance via the $2L$-Lipschitz and strong-monotonicity bounds, and telescoping the inter-solution distances with the non-expansiveness lemma $\|z_j^* - z_{j-1}^*\| \le \|z_{j-1}^* - z_j\|$, all the geometric coefficients fit under one envelope:

$$\|F(z_S)\| \le 16\lambda \sum_{s=1}^S 2^{s-1}\,\|z_{s-1}^* - z_s\|.$$

The final gradient norm is a geometrically-weighted sum of per-round subroutine errors; the weight $2^{s-1}$ grows, but so does the strong monotonicity $2^s\lambda$ of subproblem $s$, so each round can be driven just accurately enough — it suffices that $256\,\lambda_s^2 S^2\,\mathbb{E}\|z_{s+1} - z_s^*\|^2 \le \epsilon^2$ — and the tightening accuracy and improving conditioning fight to a draw.

Each subproblem is solved by **Epoch-SEG**, a two-phase stochastic extragradient that separates the two error sources, which want opposite stepsizes (optimization error wants a big $\eta$ and few steps; statistical error wants a small $\eta$ and many). Phase one runs SEG with fixed $\eta = 1/(4L)$ and epoch length $T = 8L/\lambda$ for $N$ epochs, restarting each from the previous output; the contraction factor is $1/(\lambda\eta T) = 1/2$, so each epoch halves the squared distance and adds a fixed noise term, telescoping to $\mathbb{E}\|z_N - z^*\|^2 \le 2^{-N}\mathbb{E}\|z_0 - z^*\|^2 + 8\sigma^2/(\lambda L)$ — optimization error killed fast, parked at a noise floor. Phase two eats that floor by shrinking $\eta$ and growing $T$ together, $\eta_{N+r} = 1/(2^{r+3}L)$ and $T_{N+r} = 2^{r+5}L/\lambda$, which keeps the per-epoch contraction at $1/4$ while halving the noise term each epoch; inducting over $K$ epochs gives $\mathbb{E}\|z_{N+K} - z^*\|^2 \le 2^{-(N+2K)}\mathbb{E}\|z_0 - z^*\|^2 + 8\sigma^2/(2^K\lambda L)$ at a cost of $\le 16\kappa N + 2^{K+6}\kappa$ SFO calls, with $N$ and $K$ as independent knobs for optimization vs. statistical error. Stitching this into the recursion, only the cold first round needs a real phase one ($N_0 = \lceil\log_2(512\lambda^2 S^2 D^2/\epsilon^2)\rceil$); thereafter the non-expansiveness lemma makes each warm start so good that $N_s = 3$ suffices, and the phase-two length is set by the statistical requirement $K_s = \lceil\log_2(2048\lambda_s S^2 \sigma^2/(L\epsilon^2))\rceil$. Summing $\sum_s (2L/\lambda_s)(16 N_s + 64\cdot 2^{K_s})$, the $\lambda_s$ cancels against the $2^{K_s}$ requirement in the statistical part and the total lands at $\tilde O(\sigma^2\epsilon^{-2} + \kappa)$ for SCSC — the floor, with no spurious $L^2$ and no $\epsilon^{-4}$. For general CC $f$, one cold anchor with $\lambda = \min\{\epsilon/D, L\}$ reduces it to an SCSC problem via the anchoring lemma, the recursion runs on that, and the output is a $3\epsilon$-stationary point of $f$ at total $\tilde O(\sigma^2\epsilon^{-2} + LD\epsilon^{-1})$ — both matching their lower bounds.

For something I would actually run, I collapse the triple-nested loop (outer recursion over $s$, epochs over $k$, SEG iterations over $t$) by setting $N_s = 1$, $K_s = 0$, so each round is a single SEG step and the round index and the SEG step index merge. The accumulated penalty then becomes, at iteration $t$, a single anchor pulling toward all stored past iterates with geometric weights, and one extragradient step reads

$$z_{t+1/2} = z_t - \eta\Big(F(z_t;\xi) + \lambda\gamma\textstyle\sum_{j=0}^{t-1}(1+\gamma)^j (z_t - z_j)\Big),\qquad z_{t+1} = z_t - \eta\Big(F(z_{t+1/2};\xi) + \lambda\gamma\textstyle\sum_{j=0}^{t-1}(1+\gamma)^j (z_{t+1/2} - z_j)\Big).$$

The anchor sum is $\lambda\big[(\sum_j w_j)\,z - \sum_j w_j z_j\big]$ with $w_j = \gamma(1+\gamma)^j$ — a pull from the current point toward the geometrically-weighted average of past iterates, favoring the recent past, with $\gamma$ controlling how fast the weighting grows. Naively this costs $O(t\cdot d)$ per step, but I never need the iterates individually: keeping a scalar $\texttt{weight\_sum} = \sum_j w_j$ and a vector $\texttt{weighted\_flow\_sum} = \sum_j w_j z_j$, the contribution is $\lambda(\texttt{weighted\_flow\_sum} - \texttt{weight\_sum}\cdot z)$ in $O(d)$ time and memory, updated each step by inserting the new iterate $z_{t+1}$ with weight $\gamma(1+\gamma)^{\text{step\_index}+1}$ (the stored-iterate convention is one-based even though the displayed recurrence is zero-based). Folding $\eta$ into the step factor $\tau$, both the look-ahead and the update add this running anchor plus a fresh oracle noise draw, two oracle evaluations per iteration. It keeps the essential mechanism — a moving anchor with geometrically growing strength, converging toward the saddle — without the nested epochs, and $\gamma$ is kept small so $(1+\gamma)^t$ cannot overflow while the order of the method is unchanged.

```python
import numpy as np
from typing import Any


def init_state(problem, initial_z, seed, hyperparameters):
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "step_index": 0,
        "weight_sum": 0.0,                          # running sum of geometric weights
        "weighted_flow_sum": np.zeros_like(z0),     # running weighted sum of stored iterates
    }


def step(state, oracle, problem, hyperparameters, max_sfo_calls):
    tau = float(hyperparameters["tau"])             # extragradient stepsize eta
    lam = float(hyperparameters["lambda"])          # base regularization strength lambda
    gamma = float(hyperparameters["gamma"])         # geometric growth of anchor weights
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    weighted_flow_sum = as_vector(
        state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim
    )

    # extragradient look-ahead with the moving anchor (pull toward weighted past iterates)
    g = oracle.grad(z)
    anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    w = z - tau * g + anchor_z + oracle.noise()

    # extragradient update, anchor re-evaluated at the look-ahead w
    gw = oracle.grad(w)
    anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    z_next = z - tau * gw + anchor_w + oracle.noise()

    # new iterate becomes the newest, most heavily weighted anchor
    current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    next_state = {
        "z": z_next,
        "step_index": step_index + 1,
        "weight_sum": weight_sum + current_weight,
        "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    }
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)


def get_hyperparameters(problem_name, sigma):
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    raise KeyError(f"Unknown problem: {problem_name}")
```
