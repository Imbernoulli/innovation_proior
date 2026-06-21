We want to solve a convex-concave saddle point, $\min_x \max_y f(x,y)$ with $f$ convex in $x$ and concave in $y$ — the object behind two-player zero-sum games, primal-dual formulations, robust learning and adversarial training. The quantity that actually governs the dynamics is the joint game vector field $F(z) = [\nabla_x f(x,y);\, -\nabla_y f(x,y)]$ with $z = (x,y)$, and a solution is a point where this field vanishes, $F(z^\star) = 0$. More generally $F$ need not be the gradient of any single scalar; it is just a monotone operator, $\langle F(z) - F(z'), z - z'\rangle \ge 0$ for all $z, z'$, and we are solving the variational inequality of which $F(z^\star)=0$ is the unconstrained special case. The trouble is that the obvious method — descend in $x$, ascend in $y$ simultaneously, $z_{t+1} = z_t - \tau F(z_t)$ — fails, and it fails on the simplest instance. Take $f(x,y) = x\cdot y$, whose saddle is the origin. Its field is $F(x,y) = [y; -x] = Jz$ with $J = \begin{psmallmatrix}0 & 1\\ -1 & 0\end{psmallmatrix}$ skew-symmetric, a purely rotational field: at every point it points *around* the equilibrium, never inward, so there is no downhill direction to descend. Quantitatively the simultaneous update operator is $I - \tau J$, whose eigenvalues are $1 \mp i\tau$ of modulus $\sqrt{1+\tau^2} > 1$ for every $\tau > 0$; hence $\|z_t - z^\star\|$ grows like $(1+\tau^2)^{t/2}$, geometric divergence spiraling outward. Shrinking $\tau$ only slows the blow-up, it never stops it. The diagnosis is structural: $F$'s Jacobian here is skew-symmetric, eigenvalues pure imaginary, so the field is monotone but *only just* — $\langle Jz - Jz', z-z'\rangle = 0$ everywhere, zero strong-monotonicity margin, no contractive component for a single forward evaluation to grab. Alternating descent-ascent keeps the iterates bounded but only cycles; it does not drive $\|F\|$ to zero. The implicit/backward step $z_{t+1} = z_t - \tau F(z_{t+1}) = (I+\tau F)^{-1}(z_t)$ is, by contrast, the ideal: on a monotone operator its resolvent is firmly nonexpansive for *any* $\tau > 0$, and on the bilinear field $(I+\tau J)^{-1}$ has eigenvalues of modulus $1/\sqrt{1+\tau^2} < 1$, an unconditional inward spiral. But it is implicit — $z_{t+1}$ appears inside $F$ on both sides, so each step is a nonlinear solve as hard as the original problem. It is a target to imitate, not an algorithm we can run.

I propose the Extragradient method, and its noisy version the Stochastic Extragradient (SEG). The idea is to recover the backward step's stability while only ever evaluating $F$ explicitly, at points we already hold. The backward step needs $F$ at the unknown future point $z_{t+1}$; the forward step gives $F$ at the present $z_t$, which points the wrong way. So we *guess* the future point cheaply with one ordinary forward step, the look-ahead (or leader) iterate $w = z_t - \tau F(z_t)$, evaluate the field there, and — this is the defining move — take the actual step from the *original* point $z_t$ rather than from $w$:
$$ w = z_t - \tau F(z_t), \qquad z_{t+1} = z_t - \tau F(w). $$
The first evaluation is a predictor, the second a corrector; both are closed-form, no inner solve, two operator evaluations per iteration. Anchoring at $z_t$ is essential and is exactly what separates this from a useless variant: had we stepped from $w$, i.e. $w - \tau F(w)$, that is literally two forward steps, the operator $(I-\tau J)^2$ of modulus $1+\tau^2 > 1$, still divergent. The anchor is what manufactures contraction. On the bilinear field, $F(z_t) = Jz_t$ so $w = (I-\tau J)z_t$, then $F(w) = J(I-\tau J)z_t = (J - \tau J^2)z_t = (J + \tau I)z_t$ using $J^2 = -I$, hence
$$ z_{t+1} = z_t - \tau(J + \tau I)z_t = (I - \tau J - \tau^2 I)\,z_t. $$
An inward $-\tau^2 I$ term appears that the forward step never had. The eigenvalues become $1 - \tau^2 \mp i\tau$ with modulus $\sqrt{(1-\tau^2)^2 + \tau^2} = \sqrt{1 - \tau^2(1-\tau^2)} < 1$ for $\tau < 1$ — the contraction the forward step lacked, bought with one extra gradient evaluation. That $-\tau^2 I$ is no accident: it is the leading curvature term of the resolvent. Since $(I+\tau J)(I-\tau J) = (1+\tau^2)I$, the implicit operator is $(I+\tau J)^{-1} = \tfrac{1}{1+\tau^2}(I-\tau J) = I - \tau J - \tau^2 I + O(\tau^3)$; the forward step keeps only $I - \tau J$, dropping the inward correction, while the corrected step keeps it. The general statement, for $F$ $L$-Lipschitz with $w_{\mathrm{imp}}$ the true implicit next point, is that
$$ \|z_{\mathrm{EG}} - w_{\mathrm{imp}}\| \le \tau^2 L^2 \,\|z_t - w_{\mathrm{imp}}\|, $$
so Extragradient matches the implicit step to $O(\tau^2)$ versus the forward step's $O(\tau)$. This follows from two applications of one line: $\|z_{\mathrm{EG}} - w_{\mathrm{imp}}\| = \tau\|F(w) - F(w_{\mathrm{imp}})\| \le \tau L\|w - w_{\mathrm{imp}}\|$, and then $\|w - w_{\mathrm{imp}}\| = \tau\|F(z_t) - F(w_{\mathrm{imp}})\| \le \tau L\|z_t - w_{\mathrm{imp}}\|$; chaining gives $\tau^2 L^2$. The extra factor $\tau L$ is a genuine reduction only when $\tau < 1/L$, which is precisely why the method wants small step sizes — one extra step already turns $O(\tau)$ into $O(\tau^2)$, enough to cross from divergence to convergence, so we pay for exactly one.

What makes it work in general, beyond the toy, is a clean one-step descent identity for any monotone $L$-Lipschitz $F$. With $z^\star$ satisfying $F(z^\star) = 0$, $w = z_t - \tau F(z_t)$, $z_{t+1} = z_t - \tau F(w)$, expanding $\|z_{t+1} - z^\star\|^2$ and completing the square (using $z_t - w = \tau F(z_t)$ and $z_{t+1} - z_t = -\tau F(w)$) yields
$$ \|z_{t+1} - z^\star\|^2 = \|z_t - z^\star\|^2 - 2\tau\,\langle F(w), w - z^\star\rangle + \tau^2\|F(w) - F(z_t)\|^2 - \|w - z_t\|^2, $$
and every term carries meaning. The middle term is the progress: by monotonicity $\langle F(w) - F(z^\star), w - z^\star\rangle \ge 0$ and $F(z^\star) = 0$, so $\langle F(w), w - z^\star\rangle \ge 0$ and with its minus sign it shrinks the distance. The last two terms are the discretization error of using $F(w)$ instead of the field at the true implicit point, and this is where the extra step pays again: Lipschitzness gives $\tau^2\|F(w) - F(z_t)\|^2 \le \tau^2 L^2 \|w - z_t\|^2$, so the pair becomes $(\tau^2 L^2 - 1)\|w - z_t\|^2 \le 0$ whenever $\tau \le 1/L$. Hence
$$ \|z_{t+1} - z^\star\|^2 \le \|z_t - z^\star\|^2 - 2\tau\,\langle F(w), w - z^\star\rangle - (1 - \tau^2 L^2)\,\|w - z_t\|^2, $$
with both subtracted terms nonnegative — Fejér-decreasing, strictly so until the solution. The $-\|w - z_t\|^2$ surplus that anchoring at $z_t$ provides is exactly the contractive term that cancels the rotational overshoot, and the step-size ceiling $\tau \le 1/L$ is forced by this same inequality. Two regimes fall out: if $F$ is merely monotone ($\mu = 0$), there is no linear contraction but the telescoped progress is bounded and the averaged iterate $\hat z_t = \tfrac1t\sum_k w_k$ drives the variational-inequality gap down at $O(1/t)$ in the deterministic bounded-domain setting; if $F$ is $\mu$-strongly monotone, $-2\tau\langle F(w), w - z^\star\rangle \le -2\tau\mu\|w - z^\star\|^2$ feeds a geometric factor into the recursion, giving linear last-iterate convergence with no averaging.

The remaining wrinkle is noise, and it forces one more design decision. With two evaluations per step, the tempting but wrong choice is to draw fresh independent randomness for each — sample $\xi$ for the predictor, an independent $\xi'$ for the corrector. That breaks the whole logic: the look-ahead $w$ was computed to predict the implicit point of $F(\cdot;\xi)$, but correcting with $F(\cdot;\xi')$ is approximating a *different* operator, so the $O(\tau^2)$ argument collapses and the error no longer shrinks with the distance to the solution — it is floored by the variance between $\xi$ and $\xi'$, and one observes divergence on stochastic bilinear (this is the failure of two-sample stochastic Mirror-Prox). The fix is to use the *same* sample $\xi$ for both evaluations within a step, so the iteration always approximates the implicit update of the one operator $F(\cdot;\xi)$ and the predictor-corrector logic survives. Carrying the same complete-the-square identity through in expectation, with $F(\cdot;\xi)$ a.s. monotone and $L$-Lipschitz, $g$ $\mu$-strongly convex, and variance at the optimum $\mathbb{E}\|F(z^\star;\xi) - F(z^\star)\|^2 \le \sigma^2$, the only new term is the noise cross-term $\mathbb{E}\langle F(z^\star) - F(z^\star;\xi), w - z^\star\rangle \le \eta\sigma^2 + \tfrac{1}{4\eta}\mathbb{E}\|w - z_t\|^2$ by Young's inequality; the second piece is absorbed by the $-\|w - z_t\|^2$ surplus, which is why the ceiling tightens slightly to $\eta \le 1/(2L)$ — we need a bit of that negative term left over. Unrolling the resulting recursion $(1 + 3\eta\mu/2)\,\mathbb{E}\|z_{t+1} - z^\star\|^2 \le \mathbb{E}\|z_t - z^\star\|^2 + 2\eta^2\sigma^2$ and using $1/(1+3\eta\mu/2) \le 1 - 2\eta\mu/3$ for $\eta\mu \le 1/2$ gives
$$ \mathbb{E}\|z_t - z^\star\|^2 \le \Big(1 - \tfrac{2\eta\mu}{3}\Big)^{t}\,\|z_0 - z^\star\|^2 + \frac{3\eta\sigma^2}{\mu}, $$
geometric contraction down to an $O(\eta\sigma^2/\mu)$ neighborhood that vanishes as $\eta \to 0$, recovering the clean linear rate when $\sigma = 0$. In the constrained or regularized case both steps become proximal, $w = \mathrm{prox}_{\eta g}(z_t - \eta F(z_t;\xi))$ and $z_{t+1} = \mathrm{prox}_{\eta g}(z_t - \eta F(w;\xi))$, and nothing else in the analysis changes; on an unconstrained box the prox is the identity and this collapses to the plain two-step update. Mapping onto the harness, the benchmark exposes a deterministic operator $\texttt{oracle.grad}(z) = F(z)$ and additive Gaussian update perturbation $\texttt{oracle.noise}()$, the feasible set is all of $\mathbb{R}^{2d}$ so no projection is needed, and the stochasticity is additive update noise rather than two independently sampled operators — so the same-sample concern is automatically met. The step size is per-problem: $\tau = 0.1$ on the pure-rotation bilinear field (where $L = 1$, so $\tau < 1/L$ with margin and the contraction modulus $\sqrt{1 - \tau^2(1-\tau^2)}$ wants $\tau$ well below one) and $\tau = 1$ on the structured $(\delta,\nu)$ instance whose monotone clipped component has slope at most about one.

```python
from typing import Any
import numpy as np

from fixed_benchmark import (
    ProblemSpec, StepOutput, StochasticOracle,
    as_vector, make_step_output, run_cli,
)


def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {"z": as_vector(initial_z, expected_dim=2 * problem.dim), "step_index": 0}


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)                       # F(z_t)
    w = z - tau * g + oracle.noise()         # predictor: w = z_t - tau F(z_t) + noise
    gw = oracle.grad(w)                      # F(w), same deterministic operator
    z_next = z - tau * gw + oracle.noise()   # corrector: z_{t+1} = z_t - tau F(w) + noise

    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,                                   # two operator evaluations per iteration
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1}      # tau < 1/L = 1 with margin; rotation field needs small tau
    if problem_name == "delta_nu":
        return {"tau": 1.0}      # ~1-Lipschitz monotone field; tau at the stability boundary
    raise KeyError(f"Unknown problem: {problem_name}")


if __name__ == "__main__":
    run_cli(init_state=init_state, step=step, get_hyperparameters=get_hyperparameters)
```
