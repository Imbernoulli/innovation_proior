Dropping the anchor paid off and exposed the next constraint cleanly. SEG's mean $\texttt{final\_gradient\_norm}$ fell from R-SEG's $0.751257$ to $0.182141$, driven by the bilinear half collapsing from $1.409909$ to $0.173788$ — with no pull toward $[10,10]^\top$, the look-ahead's modulus-$0.995$ contraction drove the iterate to the true origin and the bias floor vanished exactly as the $-\tau^2 I$ story promised. But the other half of the prediction came true too, and it is now binding: $\texttt{delta\_nu\_fgn}$ *rose* from $0.092606$ to $0.190493$, and at high noise it blew up to $0.936626$. Removing R-SEG's stabilizing $\lambda=0.01$ cost me on the clipped, flat $(\delta,\nu)$ field, which is merely monotone with no margin, so bare extragradient has nothing to contract there and lands wherever the $O(1/k)$ ergodic behavior and the noise leave it. The two failures are now symmetric and inverted: R-SEG's anchor contracted everything but biased bilinear into the floor; SEG dropped the anchor, fixed bilinear, and lost contraction on $(\delta,\nu)$.

What I want is the thing that sat between them: a form of anchoring that *contracts* the merely-monotone field, including under noise, without paying R-SEG's fixed-$z_0$ bias. The defect in R-SEG was never anchoring per se — it was that the anchor weight stayed *constant* forever, so the pull toward $z_0$ never died and the bias $\lambda\|z_0-z^\*\|$ was permanent. So I propose **SEAG**, stochastic extra-anchored gradient: plant a Halpern-style anchor toward $z_0$ inside the extragradient step I now trust, but with a *time-decaying* weight — strong enough early to kill the rotation and stabilize the flat field, vanishing late so the iterate is not dragged toward $z_0$ at the end. Then the bilinear collapse SEG demonstrated survives while the early contraction tames $(\delta,\nu)$.

The rate of decay is the whole design, so I derive the schedule rather than guess it. The cleanest place to find it is the continuous anchored flow $\dot z(t) = -F(z(t)) - \beta(t)(z(t)-z_0)$: the operator drives toward a zero of $F$, a decaying spring pulls back toward the start. Two speeds compete in $\beta(t)$. The contracting speed — the spring alone, $\dot z=-\beta(z-z_0)$ — is what kills the rotation and stabilizes the flat $(\delta,\nu)$ field. The vanishing speed — I want a zero of $F$, not $z_0$, so the spring must eventually die, which is precisely what R-SEG's constant weight failed to do. Parametrize $\beta(t)=\gamma/t^p$: with $p>1$ the spring dies too early and the flow is barely contracted; with $p<1$ it dies too late and keeps dragging toward $z_0$ (R-SEG's disease in the limit $p\to 0$). The sweet spot is $p=1$, $\beta(t)=1/t$, where contracting and vanishing speeds match.

I check that $\beta(t)=1/t$ actually accelerates on the rotation, since that is where R-SEG died. For $f=xy$ the anchored flow is $\dot x=-y+(1/t)(x_0-x)$, $\dot y=x+(1/t)(y_0-y)$. Multiplying by $t$ gives $\frac{d}{dt}(tx)=-ty+x_0$ and $\frac{d}{dt}(ty)=tx+y_0$; differentiating again yields forced harmonic oscillators in $tx,ty$ whose solution is $x(t)=(y_0\cos t + x_0\sin t - y_0)/t$ and $y(t)=(y_0\sin t - x_0\cos t + x_0)/t$. So the iterate decays like $1/t$, $\|z(t)\|^2 \sim 1/t^2$ — the $1/t$-weighted anchor converts the rotation's neutral circling into a clean polynomial $1/t^2$ decay, *and* the $z_0$ dependence sits inside bounded oscillating numerators divided by $t$, so its influence vanishes: no permanent bias toward $[10,10]$. That is the property I need — R-SEG's contraction, but with the bias decaying away rather than fixed. The discrete shadow of $\beta(t)=1/t$ is an anchoring coefficient decaying like $1/k$.

Concretely, I plant the decaying anchor inside both half-steps, with the offset relative to the *current* point $z$ in both lines and a constant gradient step size:

$$w = z - \tau F(z) + c_k(z_0 - z) + \text{noise}, \qquad z_{\text{next}} = z - \tau F(w) + c_k(z_0 - z) + \text{noise},$$

with $c_k = 1/(k+3)$ and $k$ the zero-based step index. When $c_k=0$ this is exactly the SEG of the previous rung; the decaying anchor is the only new thing. The $+3$ offset keeps the first coefficient sensible ($c_0=1/3<1$, a contraction not an overshoot) and avoids the $k=0$ singularity of a bare $1/k$, while the tail still decays as $1/k$ so the $1/t^2$ acceleration is intact. The decisive design choice is that this is a *pure* convex pull $c_k(z_0-z)$ — no separate $\tau\lambda$ factor, unlike R-SEG's $\tau\lambda(z_0-z)$. That matters: R-SEG's pull was $O(\tau\lambda)$ and constant; here it is $O(1/k)$ and decaying, so it can be order-one early (at $c_0=1/3$ it moves a third of the way toward $z_0$, a real contraction) yet die to nothing by iteration 900 or 6000. The strength lives in the schedule, not in a fixed $\lambda$.

That the schedule gives the right $1/k^2$ rate is structural, and it is also why the anchor must sit *inside* extragradient rather than alongside it. The anchoring extragradient with $c_k=1/(k+3)$ admits a Lyapunov function $V_k = A_k\|F(z_k)\|^2 + B_k\langle F(z_k),\,z_k-z_0\rangle$, and the schedule forces the recurrence $B_{k+1}=B_k/(1-c_k)$, which telescopes to $B_k$ growing *linearly* in $k$ and $A_k\propto c_k^{-1}B_k$ growing *quadratically*. A quadratically growing weight on $\|F(z_k)\|^2$ is exactly what makes a bounded $V_k$ force $\|F(z_k)\|^2=O(1/k^2)$ on the *last* iterate — no averaging, no best-so-far tracking. The look-ahead gradient earns a sum-of-squares in the Lyapunov decrease that a plain anchored gradient step cannot produce, and monotonicity plus Lipschitzness weighted by these coefficients give $V_{k+1}\le V_k$. Deterministically this is the optimal $O(L^2\|z_0-z^\*\|^2/k^2)$ last-iterate gradient-norm rate — faster than SEG's $O(1/k)$, with no fixed bias.

I have to be honest about noise, because the high-noise $(\delta,\nu)$ blow-up I just saw is the warning. The $1/k^2$ rests on $V_k$ decreasing every step, and that decrease was driven by identities for the *exact* operator. Additive update noise injects an error into those identities each step, and because the $\|F\|^2$ term carries $A_k\sim k^2$, the accumulated noise is *amplified* by exactly the quadratic factor that produced the acceleration — precisely stochastic Nesterov in the convex case. So I expect a fast transient followed by a noise-dominated floor: the gradient norm drops quickly while the deterministic dynamics dominate, then flattens once the $k^2$-amplified noise catches up. The decaying anchor still beats bare SEG on $(\delta,\nu)$ — it supplies the early contraction SEG lacked — but it is not a variance-control mechanism, so its floor is still set by $\sigma$.

The implementation carries $z$, the fixed anchor $z_0$, and the step index (the coefficient needs $k$); each step does two operator evaluations — predictor at $z$, corrector at the look-ahead $w$ — two noise draws, and the same offset $c_k(z_0-z)$ in both lines with $c_k=1/(\texttt{step\_index}+3)$. The step size is constant ($\tau=0.1$ bilinear, $\tau=1.0$ $(\delta,\nu)$); only the anchor decays. The falsifiable expectations against SEG: both halves should improve, since the decaying anchor adds early contraction without permanent bias — $\texttt{bilinear\_fgn}$ down from $0.173788$ toward $\sim 0.16$ as the $1/t^2$ decay runs on the near-noiseless rotation, $\texttt{delta\_nu\_fgn}$ down most from $0.190493$ toward $\sim 0.10$–$0.12$ as the early contraction attacks the flatness, with the noise floor preventing anything smaller. The mean should drop from $0.182141$ to roughly $0.13$–$0.14$, $(\delta,\nu)$ no longer the larger half. The cleanest signature is the AUC: the $1/t^2$ transient should make $\texttt{auc\_log\_iteration\_log\_grad}$ markedly more negative than SEG's $-0.346938$ even while the final norm improves only moderately — that gap between a strongly negative AUC and a not-dramatically-smaller final value is the fingerprint of "fast transient, then noise floor." If the high-noise $(\delta,\nu)$ still blows up, the next rung must add genuine variance control — re-anchoring with growing strength toward the *moving* trajectory rather than a single fixed $z_0$.

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
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    anchor_z = as_vector(state["anchor_z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    coeff = 1.0 / (step_index + 3.0)

    g = oracle.grad(z)
    w = z - tau * g + coeff * (anchor_z - z) + oracle.noise()
    gw = oracle.grad(w)
    z_next = z - tau * gw + coeff * (anchor_z - z) + oracle.noise()
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(
        {"z": z_next, "anchor_z": anchor_z, "step_index": step_index + 1},
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
