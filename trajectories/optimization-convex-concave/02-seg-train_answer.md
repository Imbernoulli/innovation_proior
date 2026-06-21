R-SEG's mean $\texttt{final\_gradient\_norm}$ landed at $0.751257$, and the split confirms the diagnosis exactly: $\texttt{bilinear\_fgn} = 1.409909$, $\texttt{delta\_nu\_fgn} = 0.092606$. The bilinear half is pinned almost precisely at the irreducible-bias floor $\lambda\|z_0-z^\*\| \approx 0.1\times\sqrt{200} \approx 1.41$ — the anchored extragradient converged fine, but it converged to the regularized solution $w^\*$, which on this rotation field sits at distance $\|F(w^\*)\|/\lambda$ from the origin, and since $\|F(z)\|=\|z\|$ here that residual *is* the $1.41$ on the board. The high-noise column barely moves from default ($0.774997$, bilinear $1.421081$), which is itself the tell: the score is set by deterministic anchor bias, not by variance, so turning the noise up changes almost nothing. The fixed pull toward $z_0=[10,10]^\top$ did its job too well. The cure is not to add something — it is to *remove* the anchor.

So I propose **SEG**, plain stochastic extragradient: run the bare look-ahead step, whose contraction comes from the operator's own curvature rather than from any external pull. Strip the bilinear instance to the bone, $f(x,y)=xy$, the one that produced the $1.41$. The joint field is $F(z)=[y,-x]=Jz$ with $J=\begin{bmatrix}0&1\\-1&0\end{bmatrix}$, the skew-symmetric rotation generator; every evaluation points *around* the origin, never toward it. A plain forward step is the operator $M=I-\tau J$ with eigenvalues $1\mp i\tau$, modulus $\sqrt{1+\tau^2}>1$ — geometric divergence for every $\tau$, because $J$ skew means pure-imaginary eigenvalues: monotone ($z^\top Jz=0$) but with strictly *zero* margin, no contractive component for a single forward evaluation to grab. This is exactly why a plain gradient step cannot touch bilinear, and why R-SEG's contraction came entirely from the artificial $\lambda$ it injected at the cost of the bias.

The fix that needs no artificial $\lambda$ is to stop trusting $F$ at where I am. The implicit step $z_{t+1}=z_t-\tau F(z_{t+1})=(I+\tau F)^{-1}(z_t)$ is the resolvent; on the bilinear field $(I+\tau J)^{-1}$ has eigenvalues $1/(1\mp i\tau)$, modulus $1/\sqrt{1+\tau^2}<1$ for *every* $\tau$ — it spirals inward unconditionally, no step-size restriction, no $\lambda$. That is the ideal, but it is implicit: $z_{t+1}$ sits inside $F$ on both sides, a nonlinear solve per step. So I imitate it explicitly. Guess the future point with one cheap forward step, the look-ahead $w = z_t - \tau F(z_t)$, then take the actual step from the *original* $z_t$ using the field at $w$:

$$z_{t+1} = z_t - \tau F(w).$$

Anchor at $z_t$, aim with $F(w)$ — two operator evaluations, a predictor at $z_t$ and a corrector at $w$, both fully explicit. The critical check is that I have not smuggled in two forward steps, which would still diverge. Grind it out on the rotation: $F(z_t)=Jz_t$, so $w=(I-\tau J)z_t$; then $F(w)=Jw=(J-\tau J^2)z_t=(J+\tau I)z_t$ since $J^2=-I$. Therefore

$$z_{t+1} = z_t - \tau(J+\tau I)z_t = (I - \tau J - \tau^2 I)z_t.$$

A $-\tau^2 I$ term has appeared that was not in the forward step. The eigenvalues are now $1-\tau^2\mp i\tau$, modulus $\sqrt{(1-\tau^2)^2+\tau^2}=\sqrt{1-\tau^2(1-\tau^2)}<1$ for $\tau<1$; at $\tau=0.1$ that is $\approx 0.99504$, below one — the spiral turns inward. The extra evaluation manufactured, for free, the contractive $-\tau^2 I$ the forward step lacked, and it did so *without any anchor bias*: the inward force is the operator's own curvature, not a pull toward an external point. That is the deep contrast with R-SEG, whose inward force was the biased spring $\lambda(z_0-z)$; here on bilinear the iterate contracts toward the *true* origin, so the $1.41$ floor simply vanishes. That $-\tau^2 I$ is no accident of the toy: the backward step expands as $(I+\tau J)^{-1} = I - \tau J - \tau^2 I + O(\tau^3)$, so extragradient keeps the inward curvature term the forward step drops — it is the $O(\tau^2)$ explicit approximation of the implicit step where the forward step is only $O(\tau)$.

The general guarantee I lean on for the $(\delta,\nu)$ half uses only monotonicity and Lipschitzness. With $F(z^\*)=0$, completing the square gives the one-step identity $\|z_{t+1}-z^\*\|^2 = \|z_t-z^\*\|^2 - 2\tau\langle F(w),w-z^\*\rangle + \tau^2\|F(w)-F(z_t)\|^2 - \|w-z_t\|^2$. The middle term is $\le 0$ by monotonicity — that is the progress; the last two are the discretization error of using $F(w)$ for the implicit point, bounded by $(\tau^2 L^2 - 1)\|w-z_t\|^2$, strictly negative for $\tau<1/L$. So the distance is Fejér-decreasing and the step ceiling $\tau\le 1/L$ is forced by exactly this inequality. What this does *not* give, when the margin $\mu=0$, is a rate on the last iterate or the gradient norm — only the ergodic $O(1/k)$ gap, and under additive update noise convergence only to an $O(\tau\sigma^2)$ ball. So I am trading R-SEG's artificial contraction-plus-bias for honest no-bias-but-slow. On bilinear that is a clear win because the bias was the whole problem; on $(\delta,\nu)$ it is a risk, since dropping R-SEG's stabilizing $\lambda=0.01$ removes the only thing that was contracting that flat merely-monotone field.

One design rule governs the noise: both evaluations within a step must use the *same* operator. The predictor-corrector logic is that $F(w)$ stands in for $F(z_{t+1})$ of one operator; independently sampled operators across the two half-steps would collapse the $O(\tau^2)$ approximation and reintroduce divergence. Here the harness's stochasticity is additive *update* noise around a single deterministic $F$, not resampled operators, so this is automatic: $\texttt{oracle.grad}$ is the same field at both evaluations and $\texttt{oracle.noise()}$ is injected after each half-step. The step uses $\tau=0.1$ on bilinear ($L=1$, with margin) and $\tau=1.0$ on $(\delta,\nu)$ (the clipped-monotone field has slope at most about one). Two operator evaluations, two noise draws, and no anchor state at all — this is the literal scaffold default.

The falsifiable expectations against the R-SEG numbers: the bilinear half should collapse from $1.41$ to a small noise-limited value, well under $0.2$, since nothing now drags the iterate toward $[10,10]$ and the modulus-$0.995$ contraction runs for 900 steps; that alone slashes the mean. The $(\delta,\nu)$ half is the risk — removing the stabilizing $\lambda$ could let it drift up from $0.0926$ toward $\sim 0.2$, because the flat field has no contraction and lands wherever the ergodic behavior and noise leave it. I predict the mean drops sharply from $0.751$ to roughly $0.15$–$0.20$, the two halves now comparable in magnitude. If $\delta\nu$ instead blows up past bilinear, that tells me the next rung must restore a *non-biased* form of anchoring — one that contracts the merely-monotone field without the fixed-$z_0$ bias that sank R-SEG.

```python
def init_state(
    problem: ProblemSpec,
    initial_z: np.ndarray,
    seed: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "z": as_vector(initial_z, expected_dim=2 * problem.dim),
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
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))

    g = oracle.grad(z)
    w = z - tau * g + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "step_index": step_index + 1},
        metric_iterate,
        2,
    )


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```
