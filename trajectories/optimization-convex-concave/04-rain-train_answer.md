SEAG is the best so far and it points at the exact ceiling I now have to break. The mean $\texttt{final\_gradient\_norm}$ dropped from SEG's $0.182141$ to $0.135449$, both halves improving as predicted ($\texttt{bilinear\_fgn}$ $0.173788\to0.160590$, $\texttt{delta\_nu\_fgn}$ $0.190493\to0.110307$), and the signature confirmed the mechanism: $\texttt{auc\_log\_iteration\_log\_grad}$ plunged from $-0.346938$ to $-1.107236$ — a far steeper log-log descent — while the *final* norm improved only moderately. That gap is the fingerprint of a fast $1/k^2$ transient hitting a noise-set floor, and the high-noise column proves the floor is the limit: $\texttt{delta\_nu\_fgn}$ still blows up to $0.582557$ there. So the binding defect is now *statistical*, not optimization. SEAG's single fixed anchor at $z_0$ must keep its regularization strength small — it decays toward zero — and small strength is weak strong-monotonicity, which is exactly the property that leaves a large noise floor $\sim\eta\sigma^2/\lambda$. A *large* fixed $\lambda$ toward $z_0$ would crush the noise but reintroduce R-SEG's bias $\lambda\|z_0-z^\*\|$. Strength (which fights noise) and bias (from a far fixed anchor) are coupled through the anchor-to-$z^\*$ distance, and that coupling is the trap.

The way out is the non-expansiveness fact I proved back at the R-SEG rung: the anchored solution $w^\*$ is *closer* to $z^\*$ than $z_0$ was, $\|w^\*-z^\*\| \le \|z_0-z^\*\|$. So I propose **RAIN**, recursively/running-anchored stochastic extragradient: do not anchor once at the far $z_0$. Solve the anchored problem approximately, get a point closer to $z^\*$, *re-anchor there*, and crank $\lambda$ up — because the new anchor is closer, the bias $\lambda\cdot(\text{distance})$ tolerates a bigger $\lambda$, and a bigger $\lambda$ makes the next subproblem more strongly monotone, better conditioned, cheaper to solve under noise. A chain of warm restarts, each anchor closer and each penalty larger, breaks the coupling: the bias stays bounded because every anchor is close, while $\lambda$ climbs geometrically and the noise floor shrinks geometrically.

I redo the recursion at the operator level, never touching function values, because the single-player convex version of this trick rests on $\min f \le f$, an inequality with no saddle-point analogue (neither $\min\max f \le f$ nor $\ge f$ holds). Define a recursively regularized sequence $f^{(s+1)} = f^{(s)} + \tfrac{\lambda_{s+1}}{2}\|x-x_{s+1}\|^2 - \tfrac{\lambda_{s+1}}{2}\|y-y_{s+1}\|^2$, where $(x_{s+1},y_{s+1})$ is the approximate solution from round $s$ and the strengths grow geometrically, $\lambda_{s+1}=(1+\gamma)\lambda_s$. In operator form, after $s$ anchors the gradient operator is

$$F^{(s)}(z) = F(z) + \sum_{i=1}^{s}\lambda_i\,(z - z_i),$$

the penalties accumulating, each a fresh anchor $z_i$ with its own strength. Running $S=\lfloor\log(L/\lambda)\rfloor$ rounds drives the accumulated strength from $\lambda$ up to about $L$, so $F^{(s)}$ is at least $\sim\lambda(1+\gamma)^s$-strongly monotone while staying $\le 2L$-Lipschitz — its condition number $2L/(\lambda(1+\gamma)^s)$ *shrinks* toward $O(1)$. The later subproblems are well-conditioned, hence cheap to solve accurately even under noise, which is the whole payoff. The central question is whether driving each subproblem's residual small makes the final $\|F(z_S)\|$ small. Peeling off the penalties, using strong monotonicity of $F^{(S-1)}$ to convert distances back into residuals, and chaining the exact solutions through $\|z^\*_j-z^\*_{j-1}\| \le \|z^\*_{j-1}-z_j\|$, the double sum collapses to a clean recursive anchoring lemma:

$$\|F(z_S)\| \le 16\lambda \sum_{s=1}^{S} (1+\gamma)^{s-1}\,\|z^\*_{s-1} - z_s\|.$$

The final gradient norm is a *geometrically-weighted sum of per-round subroutine errors*. The weight $(1+\gamma)^{s-1}$ grows, but so does the strong monotonicity $\lambda(1+\gamma)^s$ of subproblem $s$, so I can drive the per-round error down at the same geometric rate and keep the product small — a fair fight. Solving each subproblem with a two-phase epoch-SEG (a fixed step to kill optimization error, then a shrinking step to eat the statistical floor) lands the total oracle cost at the additive statistical floor $\tilde O(\sigma^2\varepsilon^{-2}+\kappa)$, with no spurious $L^2$ and no $\varepsilon^{-4}$ — the recursion buys back the two factors of $\varepsilon$ a single far anchor cannot. The merely-monotone case reduces to this via one cold anchor at $z_0$ with $\lambda=\min(\varepsilon/D,L)$.

The theory is a triple-nested loop, and I want it as the one loop the harness runs, so I collapse it: set the inner counts to one ($N_s=1$, $K_s=0$, a single SEG iteration per round), so "round $s$" and "SEG step $t$" become the same index and the accumulated penalty $\sum_i\lambda_i(z-z_i)$ becomes, at iteration $t$, an anchor pulling toward the stored past iterates with geometric weights. One extragradient step with that running anchor:

$$w = z - \tau F(z) + \tau\lambda\textstyle\sum_j w_j(z_j - z) + \text{noise}, \qquad z_{\text{next}} = z - \tau F(w) + \tau\lambda\textstyle\sum_j w_j(z_j - w) + \text{noise},$$

the regularizer being a geometrically-weighted running average of the trajectory already written into state. Two facts make this runnable in $O(d)$. First, I never need the iterates individually: $\lambda\sum_j w_j(z_{\text{current}}-z_j) = \lambda[(\sum_j w_j)z_{\text{current}} - \sum_j w_j z_j]$, so I keep two running buffers — a scalar $\texttt{weight\_sum}=\sum_j w_j$ and a vector $\texttt{weighted\_flow\_sum}=\sum_j w_j z_j$ — and the anchor contribution is $\tau\lambda(\texttt{weighted\_flow\_sum} - \texttt{weight\_sum}\cdot z_{\text{current}})$, an $O(d)$ evaluation. Second, the sign: that expression is a pull from the current point *toward* the weighted average of past points — the moving anchor, re-anchoring toward where the trajectory has been, weighted to favor the recent past, with $\gamma$ setting how fast the weighting grows. Each new iterate $z_{\text{next}}$ is inserted with weight $\texttt{current\_weight}=\gamma(1+\gamma)^{\texttt{step\_index}+1}$ (the stored-iterate convention is one-based even though the displayed recurrence is zero-based, because $z_{\text{next}}$ is written *after* the step), updating both buffers.

The structural difference from SEAG is the direction of the strength. SEAG's regularization *decayed* toward zero, so its strong-monotonicity vanished and its noise floor was fixed. Here the effective regularization toward the recent trajectory *grows*: the geometric weights $(1+\gamma)^j$ accumulate, so late iterations are increasingly strongly monotone around where the trajectory has settled — which is increasingly close to $z^\*$. The contraction keeps tightening instead of stalling, so the noise floor keeps shrinking, while the bias stays bounded because the anchor tracks the moving trajectory rather than the far fixed $z_0$. The harness fixes $\tau=0.1,\lambda=0.1,\gamma=0.001$ on bilinear and $\tau=1.0,\lambda=0.01,\gamma=0.0001$ on $(\delta,\nu)$; $\gamma$ is deliberately tiny so $(1+\gamma)^t$ cannot overflow over the 900/6000 iterations while the order of the method is unchanged — the weighting is gentle but its *accumulation* over thousands of steps is what grows the effective strength. The state carries $z$, the step index, and the two buffers; the first step has empty buffers ($\texttt{weight\_sum}=0$), so it is a plain extragradient step, and the anchor switches on as the trajectory accumulates. Two operator evaluations, two noise draws, $O(d)$ buffer updates.

The falsifiable bar against SEAG: the growing moving anchor attacks the binding constraint — the $\sigma$-set floor — not the transient, so I expect a *large* drop in the mean from $0.135449$, both halves falling substantially. $\texttt{bilinear\_fgn}$ from $0.160590$ toward the $0.02$ range, since the near-noiseless rotation ($\sigma=0.001$) lets the growing contraction run almost unimpeded, and $\texttt{delta\_nu\_fgn}$ from $0.110307$ toward $0.02$ as the growing strength around the settled trajectory crushes the noise that left SEAG stranded — the mean roughly an order of magnitude below SEAG, in the low-$0.0X$ range. The single decisive test is the high-noise $(\delta,\nu)$: SEAG blew up to $0.582557$ there because its floor scaled with $\sigma$; if the growing anchor genuinely controls variance, this rung's high-noise $(\delta,\nu)$ should be *far* smaller — if it instead still blows up to several tenths, the moving anchor is not actually crushing the floor and the recursion's promise has not survived the collapse to a single loop.

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
        "step_index": 0,
        "weight_sum": 0.0,
        "weighted_flow_sum": np.zeros_like(z0),
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
    gamma = float(hyperparameters["gamma"])
    z = as_vector(state["z"], expected_dim=2 * problem.dim)
    step_index = int(state.get("step_index", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    weighted_flow_sum = as_vector(state.get("weighted_flow_sum", np.zeros_like(z)), expected_dim=2 * problem.dim)

    g = oracle.grad(z)
    anchor_z = tau * lam * (weighted_flow_sum - weight_sum * z)
    w = z - tau * g + anchor_z + oracle.noise()
    gw = oracle.grad(w)
    anchor_w = tau * lam * (weighted_flow_sum - weight_sum * w)
    z_next = z - tau * gw + anchor_w + oracle.noise()

    current_weight = gamma * (1.0 + gamma) ** (step_index + 1)
    next_state = {
        "z": z_next,
        "step_index": step_index + 1,
        "weight_sum": weight_sum + current_weight,
        "weighted_flow_sum": weighted_flow_sum + current_weight * z_next,
    }
    metric_iterate = z_next if problem.name == "bilinear" else z
    return make_step_output(next_state, metric_iterate, 2)


def get_hyperparameters(problem_name: str, sigma: float) -> dict[str, Any]:
    if problem_name == "bilinear":
        return {"tau": 0.1, "lambda": 0.1, "gamma": 0.001}
    if problem_name == "delta_nu":
        return {"tau": 1.0, "lambda": 0.01, "gamma": 0.0001}
    raise KeyError(f"Unknown problem: {problem_name}")
```
