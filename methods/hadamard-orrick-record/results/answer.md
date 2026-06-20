**Problem.** Same `n = 29` maximal-determinant construction, scored by `|det| / (2^28 · 7^12 · 342)`.
Single-entry flip annealing plateaus near multiplier `184.6` (score `0.540`), in the band of the
best reported program-evolution results — but the record sits far above, at multiplier `320`, and no
local-move search reaches it. The reason is structural: entry-flip annealing optimizes `|det|` in the
space of `±1` matrices, where the record is not a short coordinated walk from any structured seed.

**Key idea.** Solve the problem where it is actually solved — in **Gram space** — and reproduce the
result rather than re-search it. For a `29 × 29` `±1` matrix `R`, the record is characterized by
`G = R Rᵀ`: a symmetric design with `29` on the diagonal, restricted off-diagonal inner products, and
`det(R)² = det(G)`. The dedicated maximal-determinant search (Bruce Solomon, `6 July 2002`; tabulated
by Will Orrick, published with explicit `±1` solutions by Richard Brent) found the conjectured-optimal
Gram matrix `G` with `det(G) = (2^28 · 7^12 · 320)²`, then decomposed it into `±1` factors `R`. This
rung imports one published representative `R` — class `1`, automorphism group size `18`, from Brent's
order-29 tabulation of Orrick's database (`s29allsofar.txt`) — and verifies it exactly. The matrix is
stored verbatim as `record_matrix.json` (a `29 × 29` array of `±1`); the constructor loads it.

**Why these choices.** A local-move constructor optimizes the right quantity (`|det|`) in the wrong
space; the record is decided in the far smaller, rigid space of admissible Gram matrices and only then
factored back to `±1`. So the honest top-of-ladder move is not to pretend annealing reached `320` but
to reproduce the dedicated construction from its primary source and check it with the same exact
arithmetic the evaluator uses. Verification is float-free in the verdict: structural checks
(`entries ∈ {±1}`; `G = R Rᵀ` has diagonal `29` and only the permitted off-diagonal values; the normal
condition `R Rᵀ = Rᵀ R`) confirm it is a genuine Solomon solution, and the exact fraction-free Bareiss
determinant — not `slogdet` — decides the multiplier.

**Honest ceiling.** This is the published record, reproduced and verified, not a search result that
out-climbed the previous rung; local annealing genuinely plateaus near `184.6`. Multiplier `320` is
*conjectured* optimal, not proven — the Barba ceiling for `n ≡ 1 (mod 4)` is multiplier `369.94`, still
unmet — so even the record leaves an open gap above it. The distance from `184.6` (annealing) to `320`
(Gram-space record) is the still-open content the ladder set out to measure.

**Hyperparameters / contract.** No search; deterministic load of the published matrix from
`record_matrix.json`. Output `±1` by construction, shape `(29, 29)`. Verified: exact
`|det| = 2^28 · 7^12 · 320 = 1188957517256767569920`, multiplier exactly `320.000`, score
`320/342 = 0.935673`. Source: Bruce Solomon (2002), Will Orrick's maximal-determinant database, R. P.
Brent's order-29 tabulation (`maths-people.anu.edu.au/~brent/maxdet/order29/`).

```python
import json, os
import numpy as np

BASE = (2**28) * (7**12)      # structural factor of the n=29 record
NORM = BASE * 342             # score == 1.0 target (above record, below Barba)

def bareiss_det(M):
    """Exact integer determinant via fraction-free Gaussian elimination."""
    A = [[int(x) for x in row] for row in M]
    n = len(A); sign = 1; prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]; sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]

# ---- EDITABLE: the constructor. Load the published record matrix. ----
def construct():
    # Solomon (2002) / Orrick maximal-determinant database; Brent order-29 tabulation,
    # class 1 (automorphism group size 18) of R with R R^T = G, det(G) = (2^28*7^12*320)^2.
    path = os.path.join(os.path.dirname(__file__), "record_matrix.json")
    R = np.array(json.load(open(path)), dtype=int)
    return R

if __name__ == "__main__":
    R = construct()
    assert R.shape == (29, 29)
    assert set(np.unique(R).tolist()) <= {-1, 1}

    # Structural checks: genuine Solomon Gram solution (no float in the verdict).
    G = R @ R.T
    assert np.array_equal(np.diag(G), np.full(29, 29))
    assert set(G[~np.eye(29, dtype=bool)].tolist()) <= {-3, 1, 5}
    assert np.array_equal(R @ R.T, R.T @ R)     # normal: R R^T = R^T R

    # Exact integer determinant (Bareiss) — the reported number.
    d = abs(bareiss_det(R))
    m = d / BASE
    print("det        =", d)
    print("expected   =", BASE * 320)
    print("multiplier =", m)                    # -> 320.0
    print("score      =", d / NORM)             # -> 0.935672...
    assert d == BASE * 320                       # |det| = 2^28 * 7^12 * 320
    assert d % BASE == 0 and d // BASE == 320
```
