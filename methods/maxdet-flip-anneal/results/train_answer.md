The natural place to start the `n = 29` maximal-determinant problem is the symmetric Jacobsthal
design `Q + I`, which realizes the almost-orthogonal Gram structure parity allows and yields
multiplier `49`. But measurement reveals a hard wall: not one of its `841` single-entry sign flips
raises `|det|`. The seed is a *strict* local maximum under the basic flip move, so any greedy
hill-climb terminates immediately, on its first step, at `49`. The symmetry that makes the design
elegant has parked the search at the bottom of a basin that looks like a peak to a greedy eye,
while the record at multiplier `320` lives elsewhere on the landscape. The only way out is to be
willing to walk *downhill* — to accept moves that make the determinant temporarily worse so the
search can cross the ridge out of the symmetric basin.

The method is simulated annealing on `log|det|` with single-entry sign flips, seeded at the
Jacobsthal matrix. It keeps the flip as the move and changes only the acceptance rule: propose
flipping a random entry, take it if it improves the determinant, and take it anyway with
Metropolis probability `exp(Δlog|det| / T)` if it worsens, cooling the temperature `T`
geometrically from a warm start to a small floor while remembering the best matrix ever seen. Two
choices are forced by the objective's geometry. First, the search anneals on `log|det|` rather than
`|det|`, because a single flip changes the determinant *multiplicatively*; the raw determinant is a
twenty-one-digit integer whose changes would force a temperature spanning twenty orders of
magnitude, whereas on the log scale a flip is an `O(0.01–1)` step that a single temperature near
`0.06` can govern. Second, candidates are scored inside the loop with floating-point `slogdet` —
one LU factorization — because annealing needs only a faithful *ranking* of configurations, not
exactness during the search; the exact integer determinant is recomputed with Bareiss only on the
final best matrix. Seeding at `Q + I` rather than a random sign matrix starts the search in an
already-good region (`m = 49`) instead of the random-sign swamp where the typical multiplier is a
fraction of one, so the budget is spent improving a good configuration rather than clawing up from
noise.

This works exactly as intended: annealing clears the strict local maximum immediately and climbs
`49 → 149.87`, a `3.1×` jump in `|det|` and `+0.295` in score (to `0.43821`), confirming that the
baseline's symmetry, not the move set, was the wall. The multiplier is no longer an integer, which
is expected — `7^12` divides the determinant only for specially structured matrices, so a searched,
symmetry-broken configuration has fractional `m`, and the score is defined on `|det|` directly. The
method also exposes its own ceiling honestly. A single-entry flip is a local move, and each
candidate costs a full `O(n³)` factorization, so a single-seed run of forty thousand flips is about
what is affordable; it plateaus in the low-`150`s while the frontier sits near `197` and the record
at `320`. The binding constraint is per-flip cost — the remaining climb needs far more flips and
many restarts than full-recompute scoring can pay for — which is precisely the opening for making
each flip cheaper.

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
