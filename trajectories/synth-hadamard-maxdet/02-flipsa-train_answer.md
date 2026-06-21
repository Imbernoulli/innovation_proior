The Jacobsthal design measured out at multiplier exactly `49`, and the feedback added one sharp fact: of the `841` single-entry sign flips available from $Q + I$, not one increases $|\det|$. So $Q + I$ is a *strict* local maximum under the most basic move I have. Any greedy hill-climb from this seed terminates on its first step, at `49`, because there is no uphill direction at all. The symmetry of the construction has parked me at the bottom of a basin that looks like a peak to a greedy eye, while the record at `320` lives somewhere else entirely on the landscape. The only way out is to be willing to walk *downhill* — to accept moves that make the determinant temporarily worse — so that the search can cross the ridge bounding this basin and reach a region greedy could never enter.

I propose **simulated annealing on $\log|\det|$ with single-entry sign flips**. The move set is unchanged from greedy — propose flipping one random entry — but the acceptance rule changes. If the flip improves $|\det|$, take it; if it worsens $|\det|$, take it anyway with the Metropolis probability $\exp(\Delta\log|\det| / T)$, where $T$ is a temperature I cool over the run. Early, when $T$ is high, almost any downhill move is accepted and the search wanders freely, shaking loose from the Jacobsthal basin; late, when $T$ is small, only improving moves survive and the search settles into whatever basin it has wandered into. The whole bet is that with enough wandering it finds a basin deeper than `49` before it freezes.

Two design decisions are forced by the geometry of *this* objective, and they are what make the method work rather than just restating textbook annealing. The first is **what quantity the acceptance rule should compare**. The raw determinant here is astronomically large — multiplier times $2^{28}\cdot 7^{12}$, a twenty-one-digit integer — and a single flip can change it by an enormous absolute amount while changing it only modestly in relative terms. If I annealed on the raw difference $|\det'| - |\det|$, the temperature would have to span twenty orders of magnitude and the schedule would be untunable. The natural scale is multiplicative: a flip multiplies the determinant by some ratio, and what matters is whether that ratio sits above or below $1$. So I anneal on $\log|\det|$, not $|\det|$. A flip from $m=49$ to $m=48$ is a log step of about $-0.02$; one that doubles the determinant is $+0.69$. On the log scale the steps are $O(0.01)$ to $O(1)$, a temperature of order $0.01$–$0.1$ governs the dynamics, and the schedule is sane. The reframing is the key idea of the rung: *maximize $\log|\det|$ by annealing, and read the exact integer determinant only at the end.*

The second decision is **how to evaluate a candidate flip cheaply enough to afford many of them**. The honest, exact thing is the Bareiss integer determinant, but that is $O(n^3)$ of big-integer arithmetic per candidate, and I will propose tens of thousands of candidates. I do not need exactness *during* the search — I only need a faithful ranking of which configurations have larger $|\det|$ — and the final answer's determinant I compute exactly anyway. So inside the loop I score with floating-point $\log|\det|$ via `slogdet`: one LU factorization, $O(n^3)$ in floats but fast and, at order `29`, accurate enough to compare configurations. Each accepted flip simply recomputes `slogdet` from scratch. It is not clever, but it is correct, and at `n = 29` a float factorization is cheap enough to give tens of thousands of evaluated flips per second.

One more choice carries real weight: the **seed**. Starting from $Q + I$ rather than from a random sign matrix begins the search in a structured, already-good region ($m = 49$) instead of the random-matrix swamp, where the typical determinant multiplier is a fraction of one. The annealing then spends its budget improving a good configuration rather than first clawing up out of noise. Concretely the schedule is a warm start log-temperature `0.06`, geometric decay to a floor `2e-4` over the step budget, `40k` flips from the $Q + I$ seed, a fixed RNG seed for reproducibility, and the best matrix ever seen retained and re-scored exactly with Bareiss at the end.

I am honest about the ceiling of this move set, because it sets up the next rung. A single-entry flip is a local move, and each candidate costs a full $O(n^3)$ factorization, so the affordable budget is only tens of thousands of flips from one seed — enough to clear the strict-local-maximum wall and climb into the low hundreds of multiplier, but it will plateau short of the frontier. The remaining gains require long, coordinated multi-entry sequences that need far more flips and many restarts than full-recompute scoring can pay for. So I expect this rung to *prove the principle* — that accepting downhill moves buys a large jump off the symmetric baseline — and then to stall, limited not by the idea but by per-flip cost. That stall is precisely the opening for what comes next: if the only thing holding me back is the cost of scoring a flip, making each flip dramatically cheaper lets the same annealing run far longer and climb the part of the curve this rung can only begin.

```python
import numpy as np

def construct(iters=40000, seed=11):
    q = 29
    qr = set((i * i) % q for i in range(1, q))
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], dtype=int)
    M = (np.eye(q, dtype=int) + Q).astype(float)          # Jacobsthal seed, m = 49

    rng = np.random.default_rng(seed)
    logdet = lambda A: np.linalg.slogdet(A)[1]
    cur = logdet(M); best = M.copy(); best_ld = cur
    logT = 0.06; decay = (2e-4 / 0.06) ** (1.0 / iters)
    for _ in range(iters):
        i, j = rng.integers(q), rng.integers(q)
        M[i, j] *= -1                                       # propose single-entry flip
        nl = logdet(M); d = nl - cur                        # anneal on log|det|
        if d >= 0 or rng.random() < np.exp(d / logT):
            cur = nl
        else:
            M[i, j] *= -1                                   # reject: undo
        if cur > best_ld:
            best_ld = cur; best = M.copy()
        logT = max(logT * decay, 2e-4)                      # geometric cooling
    return np.rint(best).astype(int)
```
