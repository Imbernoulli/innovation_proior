The substrate I am handed is already a working method: plain stochastic extragradient, the scaffold default. That default exists because the naive thing — simultaneous descent-ascent $z_{t+1} = z_t - \tau F(z_t)$ — does not even survive the easiest convex-concave instance. On $f=xy$ the operator $F(z)=[y,-x]=Jz$ is a $90°$ rotation, the step is orthogonal to the pull toward the saddle, and $\|z_{t+1}\|^2 = (1+\tau^2)\|z_t\|^2$ spirals strictly outward for every $\tau>0$. Extragradient's look-ahead repairs that and contracts the rotation. But the quantity I am graded on is the *gradient norm under noise*, and here the honest situation is worse than "converges": writing the per-step contraction inequality for SEG,

$$\lambda\,\mathbb{E}\|z_{1/2}-z^\*\|^2 \le \tfrac{1}{\eta}\,\mathbb{E}\big[\|z_t-z^\*\|^2 - \|z_{t+1}-z^\*\|^2\big] + c\,\eta\sigma^2,$$

every useful term carries the strong-monotonicity constant $\lambda$, and the noise enters as a floor $c\eta\sigma^2/\lambda$ that does not vanish with more iterations. My operators are *merely* convex-concave, so $\lambda=0$: the bilinear field is a pure rotation with zero margin, the $(\delta,\nu)$ field's clipped component is monotone but flat. At $\lambda=0$ the inequality says nothing — no contraction, only the $O(1/k)$ ergodic gap, and no last-iterate gradient-norm guarantee at all. The property that would make SEG fast is exactly the one this problem denies me.

So the first move I make is to *manufacture* the missing strong monotonicity, because it is suspiciously cheap and cheap is worth measuring before anything elaborate. I propose **R-SEG**, Tikhonov-anchored stochastic extragradient: add to $F$ a quadratic pull toward a fixed anchor point $a$,

$$G(z) = F(z) + \lambda(z - a),$$

and run the extragradient step I already trust on $G$ instead of $F$. For any $z,z'$, $\langle G(z)-G(z'),\,z-z'\rangle = \langle F(z)-F(z'),\,z-z'\rangle + \lambda\|z-z'\|^2 \ge \lambda\|z-z'\|^2$, so $G$ is $\lambda$-strongly monotone *by construction* regardless of $F$ being merely monotone, while remaining $(L+\lambda)$-Lipschitz — essentially as smooth as $F$. $G$ is the gradient operator of the Tikhonov-regularized saddle objective $f(x,y) + \tfrac{\lambda}{2}\|x-a_x\|^2 - \tfrac{\lambda}{2}\|y-a_y\|^2$: a strongly-convex penalty on $x$, a strongly-concave one on $y$. Running SEG on $G$ recovers the contraction the bare problem could never give: the iterate contracts geometrically toward $G$'s zero $w^\*$ with a genuine noise floor $\sim\eta\sigma^2/\lambda$ at $\lambda>0$.

The cost — and it is the entire content of this rung — is that I have solved a *different* problem. Running on $G$ drives me toward $w^\*$, the zero of $G$, which is not $z^\*$, the zero of $F$, and the metric is $\|F\|$, not $\|G\|$. From the definition, $F(w^\*) = G(w^\*) - \lambda(w^\*-a) = -\lambda(w^\*-a)$, so even the *exact* regularized solution carries residual $\|F(w^\*)\| = \lambda\|w^\*-a\|$: an irreducible bias the reported gradient norm can never fall below. I want a clean transfer bound. For any candidate $\tilde z$,

$$\|F(\tilde z)\| \le \|G(\tilde z)\| + \lambda\|\tilde z - a\| \le \|G(\tilde z)\| + \lambda\|\tilde z - w^\*\| + \lambda\|w^\* - a\|,$$

and strong monotonicity of $G$ (with $G(w^\*)=0$) gives $\lambda\|\tilde z - w^\*\| \le \|G(\tilde z)\|$, so $\|F(\tilde z)\| \le 2\|G(\tilde z)\| + \lambda\|w^\*-a\|$. The first term I crush by running SEG on $G$; the second, $\lambda\|w^\*-a\|$, is the price of regularizing.

Where to put the anchor? The only special point I have at the start is $z_0$; I cannot anchor at the unknown $z^\*$. So I anchor at the initial point, $a=z_0$, fixed forever. This is geometrically safe: strong monotonicity of $G$ between $w^\*$ and $z^\*$ together with $F(z^\*)=0$ gives $\lambda\|w^\*-z^\*\|^2 \le \lambda(z^\*-z_0)^\top(z^\*-w^\*)$, and polarizing yields *both* $\|w^\*-z_0\| \le \|z^\*-z_0\|$ and $\|w^\*-z^\*\| \le \|z^\*-z_0\|$ — the regularized solution is no farther from the anchor than the true solution is. The transfer bound collapses to

$$\|F(\tilde z)\| \le 2\|G(\tilde z)\| + \lambda\|z_0 - z^\*\|,$$

so I keep no extra anchor state beyond $z_0$ and the only residual lever is the single distance $\|z_0-z^\*\|$. The choice of $\lambda$ is then a genuine tension: large $\lambda$ makes $G$ strongly monotone, so the contraction is fast and the noise floor $\eta\sigma^2/\lambda$ small, but the bias $\lambda\|z_0-z^\*\|$ is large; small $\lambda$ kills the bias but barely regularizes, so the conditioning $L/\lambda$ and the noise floor blow up. The balance is $\lambda \sim \varepsilon/D$ with $D$ a bound on $\|z_0-z^\*\|$; the harness fixes $\tau=0.1,\lambda=0.1$ on bilinear and $\tau=1.0,\lambda=0.01$ on $(\delta,\nu)$.

The implementation is one extragradient step on $G$: each half-step adds a fixed pull $\tau\lambda(z_0-\cdot)$ toward the anchor plus the oracle noise — the predictor pulling from the current $z$, the corrector re-evaluating the operator at the look-ahead $w$ while still pulling toward $z_0$. Two operator evaluations, two noise draws, the fixed $z_0$ anchor that never moves.

I should be precise about where this will hurt, because it is the falsifiable prediction. The irreducible bias $\lambda\|z_0-z^\*\|$ is large precisely where the start is far from the solution — and on bilinear, $z_0=[10,10]^\top$ is *very* far from the saddle at the origin: $\|z_0-z^\*\| = \sqrt{200} \approx 14.14$, so with $\lambda=0.1$ the residual floor is $\approx 1.41$. Since $\|F(z)\|=\|z\|$ on this rotation field, the best bilinear gradient norm this rung can reach is about $1.4$ no matter how many of the 900 iterations I spend — the fixed pull toward $[10,10]$ literally drags the iterate back toward the worst possible point. The $(\delta,\nu)$ case is the opposite: there $z_0\sim N(0,I)$ already starts near the solution and $\lambda=0.01$ is tiny, so the bias is negligible and the mild regularization only stabilizes; I expect a small $\delta\nu$ norm, an order or two below the bilinear value. The mean, the average of the two, will be dragged up almost entirely by the bilinear half — the worst of any rung on this ladder. The diagnosis goes in the record: the fixed-$z_0$ Tikhonov anchor buys strong monotonicity and noise robustness but pays an irreducible $\lambda\|z_0-z^\*\|$ bias, and on bilinear the start is so far from the solution that the bias *is* the score. The next rung's job will be to keep extragradient's anti-rotation contraction without paying that anchor bias at all.

```python
def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    z0 = as_vector(initial_z, expected_dim=2 * problem.dim)
    return {
        "z": z0,
        "anchor_z": z0.copy(),
        "step_index": 0,
    }


def step(
    state: dict[str, Any],
    oracle: StochasticOracle,
    problem: ProblemSpec,
    hyperparameters: dict[str, Any],
    max_sfo_calls: int,
) -> StepOutput:
    tau = float(hyperparameters["tau"])
    lam = float(hyperparameters["lambda"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)
    w = z - tau * g + tau * lam * (anchor_z - z) + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + tau * lam * (anchor_z - w) + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01}
    raise KeyError(f"Unknown problem: {problem_name}")
```
