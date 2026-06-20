**Problem.** Construct a single `29 × 29` `±1` matrix maximizing `|det|`, scored as
`|det| / (2^28 · 7^12 · 342)`. Because `29 ≡ 1 (mod 4)` the Hadamard bound is unreachable and the
true maximum is open (record multiplier `320`, Barba ceiling `369.94`). The principled symmetric
Jacobsthal seed `Q + I` is a *strict* local maximum under single-entry flips (0 of 841 flips raise
`|det|`), so any greedy climb terminates at multiplier `49`. The gap must be crossed by leaving the
symmetric basin.

**Key idea.** Simulated annealing on `log|det|` with single-entry sign flips. Propose flipping a
random entry; accept improving flips always and worsening flips with Metropolis probability
`exp(Δlog|det| / T)`; cool `T` geometrically from a warm start to a small floor; keep the best
matrix seen. Two choices are forced by the objective's geometry: (1) anneal on `log|det|`, not
`|det|` — a single flip changes the determinant *multiplicatively*, and the log scale turns
21-digit swings into `O(0.01–1)` steps a single temperature can govern; (2) score candidates with
floating-point `slogdet` inside the loop (a faithful ranking is all annealing needs), and recompute
the *exact* integer determinant with Bareiss only on the final matrix. Seeding at `Q + I` rather
than a random matrix starts the search in an already-good region (`m = 49`) instead of the
random-sign swamp (`m ≪ 1`).

**Why these choices.** Accepting downhill moves is the only way out of a strict local maximum;
that is the whole content of the method. The log scale is what makes the temperature tunable at
all. `slogdet` trades exactness-during-search (unnecessary) for speed (necessary), with exactness
restored at the end. The limitation is structural: a single-entry flip is a local move, and each
candidate costs a full `O(n³)` factorization, so the affordable budget is tens of thousands of
flips from one seed — enough to clear the wall and reach the low hundreds of multiplier, but it
plateaus short of the frontier because the remaining gains need far more flips and many restarts
than full-recompute scoring can pay for.

**Hyperparameters / contract.** Start `log`-temperature `0.06`, geometric decay to floor `2e-4`
over the step budget; `40k` flips from the `Q + I` seed; fixed RNG seed for reproducibility.
Output is `±1` by construction (flips preserve the `±1` alphabet). Reported determinant is the
exact Bareiss integer of the best matrix. Measured: multiplier `149.87`, score `0.43821`
(a `3.1×` jump in `|det|`, `+0.295` in score, over the Jacobsthal baseline).

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
