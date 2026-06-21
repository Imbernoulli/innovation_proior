We want to solve a smooth convex-concave minimax problem $\min_x \max_y L(x,y)$, where $L$ is convex in $x$ for each fixed $y$ and concave in $y$ for each fixed $x$. Bundling the variables into $z=(x,y)$ and the gradients into the saddle operator $G(z) = [\nabla_x L;\, -\nabla_y L]$ — with the sign flip on the $y$-block built in so that a saddle point is exactly a zero of $G$ — convex-concavity makes $G$ monotone, $\langle G(z_1)-G(z_2), z_1-z_2\rangle \ge 0$, and I assume it is $R$-Lipschitz. The quantity I want to drive to zero is the squared gradient norm $\|\nabla L(z)\|^2 = \|G(z)\|^2$ (the sign flip does not change the norm), because that is what actually vanishes at a solution and stays meaningful even outside the convex-concave world, in the GAN-like games I really care about; the classical duality gap needs bounded domains even to be finite and is awkward to measure, so I commit to the gradient norm on the unconstrained $\mathbb{R}^n\times\mathbb{R}^m$, and I want it small in the *last* iterate, not an average and not the best-so-far.

The trouble is that the obvious method already fails on the easiest instance. Simultaneous gradient descent-ascent $z^{k+1} = z^k - \alpha G(z^k)$ on $L(x,y)=xy$ has $G(z) = [y,-x]$, a $90^\circ$ rotation, so each step pushes perpendicular to the current position vector. In continuous time $\dot z = -G(z)$ conserves $\|z\|^2$ and circles the saddle forever; the explicit-Euler step grows it, $\|z^{k+1}\|^2 = \|z^k\|^2(1+\alpha^2)$ because $\langle z^k, G(z^k)\rangle = 0$, so the iterate spirals strictly outward and diverges. Extragradient cures the rotation by not trusting the gradient where it stands: take a look-ahead step $w = z - \alpha G(z)$, evaluate $G$ there, and apply that corrected direction back at $z$, $z^+ = z - \alpha G(w)$. Pushing both distances through the common point $w$ and using monotonicity ($\langle G(w), w-z^*\rangle \ge 0$) and Lipschitzness ($\|z^+ - w\|^2 = \alpha^2\|G(z)-G(w)\|^2 \le \alpha^2 R^2 \|z-w\|^2$) gives the one-step decrease $\|z-z^*\|^2 - \|z^+-z^*\|^2 \ge (1-\alpha^2 R^2)\alpha^2\|G(z)\|^2$, strictly positive for $\alpha < 1/R$. Summing telescopes the left side to $\|z^0-z^*\|^2$ and yields $\min_{i\le k}\|G(z^i)\|^2 = O(R^2/k)$ — but only $O(1/k)$, and only *best-iterate*, with a hard last-iterate ceiling for the stationary algorithm class containing extragradient. Optimism (Popov) saves a gradient evaluation but lives in the same class. Separately, the Halpern iteration $u_{k+1} = \lambda_{k+1} u^0 + (1-\lambda_{k+1}) T(u_k)$ for a nonexpansive $T$ makes the *last* iterate converge — to the fixed point nearest the anchor $u^0$, an implicitly regularized selection — with residual $\|T(u_k)-u_k\| = 2\|u^0-u^*\|/(k+1)$ for $\lambda_k = 1/(k+1)$. Transplanting that anchor onto plain gradient steps (SimGD-A) gives a last-iterate gradient-norm rate $O(1/k^{2-2p})$, but the step size is forced to *diminish* like $(1-p)/(k+1)^p$ with $p>1/2$, so it crawls and never reaches $O(1/k^2)$. The two failures are exactly complementary: extragradient allows a *constant* step but is best-iterate and stuck at $O(1/k)$; anchoring is last-iterate and selects the solution but is shackled to a *shrinking* step. Each supplies precisely the ingredient the other lacks.

So I propose the Extra Anchored Gradient method, EAG: run the extragradient predictor-corrector but plant the anchor pull inside *both* of its half-steps, and keep the step size constant,
$$z^{k+1/2} = z^k + \beta_k(z^0 - z^k) - \alpha\, G(z^k),$$
$$z^{k+1} = z^k + \beta_k(z^0 - z^k) - \alpha\, G(z^{k+1/2}).$$
The anchor offset $\beta_k(z^0-z^k)$ is relative to the current point $z^k$ in *both* lines — I anchor where I am, not at the half-step $w$ — and $\beta_k = 0$ recovers plain extragradient, so anchoring is the only new ingredient. The extragradient look-ahead earns the per-step decrease at a *fixed* step that SimGD-A had to buy with a vanishing step; the anchor makes the *last* iterate, not just the best, go to zero and quietly selects the nearest solution. The schedule is not arbitrary: it comes from the anchored flow $\dot z = -G(z) - \beta(t)(z-z^0)$, where the spring has two competing speeds — a *contracting* speed that stabilizes the flow and kills cycling, and a *vanishing* speed, since the spring must die or I converge to $z^0$ rather than to a zero of $G$. With $\beta(t) = \gamma/t^p$, $p>1$ kills the spring too early (flow barely contracted, slow) and $p<1$ too late (spring keeps dragging, slow); $p=1$, i.e. $\beta(t)=1/t$, balances them. On $L=xy$ the anchored flow solves in closed form, $x(t) = (y^0\cos t + x^0\sin t - y^0)/t$ and $y(t) = (y^0\sin t - x^0\cos t + x^0)/t$, decaying like $1/t$ so that $\|G\|^2 \sim 1/t^2$ — versus the EG-flavored Moreau–Yosida flow's feeble $\exp(-\lambda t/(1+\lambda^2))$. The discrete shadow of $\beta(t)=1/t$ is $\beta_k = 1/(k+2)$, the $+2$ just keeping $\beta_0 = 1/2 < 1$ at the start.

What makes it work is a Lyapunov function tuned so that the anchor's $1/(k+2)$ schedule produces quadratic weighting on the gradient norm. Take
$$V_k = A_k\,\|G(z^k)\|^2 + B_k\,\langle G(z^k),\, z^k - z^0\rangle,$$
the first term being the very quantity I want to bound (weighted up so that bounded $V$ forces $\|G\|^2$ small), the second the anchoring inner product of the gradient with the displacement from the anchor. The EAG update gives three structural identities — $z^k - z^{k+1} = \beta_k(z^k-z^0) + \alpha_k G(z^{k+1/2})$, $z^{k+1/2}-z^{k+1} = \alpha_k(G(z^{k+1/2})-G(z^k))$, and $z^0-z^{k+1} = (1-\beta_k)(z^0-z^k) + \alpha_k G(z^{k+1/2})$ — and from $V_k - V_{k+1}$ I subtract a nonnegative monotonicity term $\langle z^k-z^{k+1}, G(z^k)-G(z^{k+1})\rangle$ with weight $B_k/\beta_k$ and a nonnegative Lipschitz term with weight $A_k/(\alpha_k^2 R^2)$. Imposing $B_{k+1} = B_k/(1-\beta_k)$ collects the cross terms cleanly, $A_k = \alpha_k B_k/(2\beta_k)$ annihilates the $\langle G(z^k), G(z^{k+1/2})\rangle$ term, and the recurrence
$$A_{k+1} = A_k\,\frac{1 - \alpha_k^2 R^2 - \beta_k^2}{(1-\alpha_k^2 R^2)(1-\beta_k)^2}$$
leaves $V_k - V_{k+1} \ge a\|G(z^{k+1/2})\|^2 + b\|G(z^{k+1})\|^2 - 2c\langle G(z^{k+1/2}), G(z^{k+1})\rangle$ with $c^2 = ab$ — a perfect signed square, so $V_k$ is nonincreasing. The look-ahead point $z^{k+1/2}$ is exactly what extragradient bought, and it is what supplies the $\|G(z^{k+1/2})\|^2$ term that lets everything close into a sum of squares. Now the payoff of $\beta_k = 1/(k+2)$: $B_{k+1} = B_k(k+2)/(k+1)$ telescopes from $B_0 = 1$ to $B_k = k+1$, *linear*, and $A_k = \alpha_k(k+1)(k+2)/2$ is *quadratic*. Linear $B$ and quadratic $A$ is the structural reason an $O(1/k^2)$ rate is even possible. Bounding $V_k \le V_0 = \alpha_0\|G(z^0)\|^2 \le \alpha_0 R^2\|z^0-z^*\|^2$ from above and, via monotonicity plus Young's inequality, $V_k \ge (\alpha_\infty/4)(k+1)(k+2)\|G(z^k)\|^2 - (1/\alpha_\infty)\|z^0-z^*\|^2$ from below, combining gives
$$\|\nabla L(z^k)\|^2 = \|G(z^k)\|^2 \le \frac{4(1 + \alpha_0\alpha_\infty R^2)}{\alpha_\infty^2}\cdot\frac{\|z^0-z^*\|^2}{(k+1)(k+2)},$$
the optimal $O(R^2\|z^0-z^*\|^2/k^2)$ on the *last* iterate, with no averaging and no best-so-far tracking. This is the varying-step form EAG-V, where $\alpha_k$ follows $\alpha_{k+1} = \alpha_k(1 - \frac{1}{(k+1)(k+3)}\cdot\frac{\alpha_k^2 R^2}{1-\alpha_k^2 R^2})$, which is strictly decreasing yet bounded away from zero (e.g. $\alpha_0 = 0.618/R \Rightarrow \alpha_\infty \approx 0.437/R$, giving constant $\approx 27$). The genuinely *constant*-step form EAG-C — one fixed $\alpha$ throughout, nothing to tune across steps — keeps the same $V_k$, $B_k = k+1$, $\beta_k = 1/(k+2)$, but now combines the monotonicity inequality (weight $(k+1)(k+2)$) and the Lipschitz inequality (weight $\tau_k \ge 0$) into a single trace $V_k - V_{k+1} \ge \mathrm{Tr}(M_k S_k M_k^\top)$ with $M_k = [G(z^k)\;G(z^{k+1/2})\;G(z^{k+1})]$ and a tridiagonal $S_k$; keeping $S_k \succeq 0$ while forcing $A_k$ to grow quadratically (the performance-estimation idea) goes through for $\alpha R$ small, under $1 - 3\alpha R - \alpha^2 R^2 - \alpha^3 R^3 \ge 0$ and $1 - 8\alpha R + \alpha^2 R^2 - 2\alpha^3 R^3 \ge 0$ (which hold for $\alpha \in (0, 1/(8R)]$), with rate $\|\nabla L(z^k)\|^2 \le \frac{4(1+\alpha R + \alpha^2 R^2)}{\alpha^2(1+\alpha R)}\cdot\frac{\|z^0-z^*\|^2}{(k+1)^2}$, constant $260$ at $\alpha = 1/(8R)$. The fixed step is simpler to run; the varying step makes the proof clean.

The rate is optimal, not merely good. Through biaffine problems $L(x,y) = \langle Ax-b, y-c\rangle$ — the simplest convex-concave family, with $G$ being $\|A\|$-Lipschitz — any first-order method whose iterates stay in the span of queried gradients keeps $x^k, y^k$ in the order-$(k-1)$ Krylov subspace of $A$ built from $b$, reducing the minimax problem to solving $Ax=b$ by matrix-vector products. Nemirovsky's matrix-Chebyshev lower bound then gives $\|Ax-b\|^2 \ge R^2\|x^*\|^2/(2\lfloor k/2\rfloor+1)^2$ per block, so $\|\nabla L(z^k)\|^2 \ge R^2\|z^0-z^*\|^2/(2\lfloor k/2\rfloor+1)^2 = \Omega(R^2/k^2)$, matching the upper bound in order. EAG breaks the $O(1/k)$ last-iterate ceiling that pins extragradient precisely because that lower bound is for *stationary* algorithms with fixed coefficients, and $\beta_k = 1/(k+2)$ is non-stationary — the anchoring schedule is what escapes the class. The acceleration is genuinely not Nesterov in disguise: momentum *adds* inertia along the recent direction, while the anchor $\beta_k(z^0-z^k)$ *pulls back* toward the start, damping the oscillation; two opposite mechanisms, both reaching $O(1/k^2)$, here combined with the extragradient look-ahead that handles the monotone rotation a pure gradient step cannot. One honest caveat: the whole acceleration rests on $V_k$ decreasing every step, which used monotonicity and Lipschitz identities for the *exact* operator. With noisy gradients $G(z)+\xi$, or equivalently an additive update perturbation $-\alpha G(z) + \eta$ (the two coincide for fixed $\alpha$ with $\eta = -\alpha\xi$), the same $A_k \sim k^2$ weighting that produces the acceleration also *amplifies* the injected noise, so the stochastic instantiation SEAG keeps EAG's fast $O(1/k^2)$ transient but levels off at a floor set by $\sigma$; stability needs the oracle variance controlled on the order of $1/k$, and with fixed variance the error accumulates — exactly like stochastic Nesterov for convex minimization. The update itself is unchanged; the noise simply enters around the operator calls, and reducing the floor would need a separate variance-control mechanism layered on top.

```python
import numpy as np


def init_state(problem, initial_z, hyperparameters):
    z0 = np.asarray(initial_z, dtype=float).reshape(2 * problem.dim)
    return {
        "z": z0,                 # current iterate z^k
        "anchor_z": z0.copy(),   # the anchor z^0 (fixed)
        "step_index": 0,         # k
    }


def step(state, problem, hyperparameters):
    tau = float(hyperparameters["tau"])           # constant step α
    z = state["z"]
    anchor_z = state["anchor_z"]
    k = int(state["step_index"])

    # anchoring coefficient β_k = 1/(k+2): the 1/t anchored-flow schedule
    beta = 1.0 / (k + 2.0)

    # predictor: z^{k+1/2} = z^k + β_k (z^0 - z^k) - α G(z^k) + η
    g = problem.grad(z)
    w = z + beta * (anchor_z - z) - tau * g + problem.noise()

    # corrector: z^{k+1} = z^k + β_k (z^0 - z^k) - α G(z^{k+1/2}) + η'
    gw = problem.grad(w)
    z_next = z + beta * (anchor_z - z) - tau * gw + problem.noise()

    return {"z": z_next, "anchor_z": anchor_z, "step_index": k + 1}, z_next


def get_hyperparameters(problem_name):
    # constant step τ = α
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```
