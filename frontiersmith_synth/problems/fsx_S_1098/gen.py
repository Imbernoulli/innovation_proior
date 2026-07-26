#!/usr/bin/env python3
"""gen.py <testId> -- emit one cheap-spectral-projector instance to stdout.

Construction: A = diag(D) + E where D plants two eigenvalue clusters around
theta=0.5 with a guaranteed gap (via a Weyl/Gershgorin row-sum bound on the
banded perturbation E), so the checker can certify the revealed gap WITHOUT
needing an eigendecomposition at generation time. All randomness is seeded
deterministically by testId.
"""
import sys
import numpy as np

# ladder: (n, gap) -- gap widens for easy cases, shrinks (trap) for hard ones
LADDER = {
    1: (30, 0.35), 2: (40, 0.25), 3: (50, 0.16), 4: (60, 0.11), 5: (70, 0.075),
    6: (90, 0.05), 7: (110, 0.032), 8: (120, 0.022), 9: (140, 0.015), 10: (150, 0.010),
}
EPS = 5e-3
B_BW = 3      # matrix bandwidth
K_PROBES = 4
THETA = 0.5


def main():
    testId = int(sys.argv[1])
    n, gap = LADDER[testId]
    rng = np.random.default_rng(900000 + 97 * testId)

    n_low = n // 2
    n_high = n - n_low
    delta = 0.1 * gap  # perturbation-norm budget (Gershgorin/Weyl bound)
    # planted diagonal ranges leave a full extra `delta` margin at BOTH the
    # gap side and the [0,1] domain edges, so that after the worst-case
    # perturbation shift (<= delta, by Weyl) every eigenvalue is guaranteed
    # to stay strictly inside (0, 1) -- required so u = 2*lambda-1 stays
    # inside [-1, 1] for the Chebyshev-basis filters used by the solutions.
    low_lo = 2.0 * delta
    low_hi = THETA - gap - delta
    high_lo = THETA + gap + delta
    high_hi = 1.0 - 2.0 * delta
    assert 0.0 < low_lo < low_hi < THETA < high_lo < high_hi < 1.0, \
        "gap/delta budget infeasible for this (n, gap)"

    d_low = rng.uniform(low_lo, low_hi, size=n_low)
    d_high = rng.uniform(high_lo, high_hi, size=n_high)
    diag = np.concatenate([d_low, d_high])
    rng.shuffle(diag)

    # banded symmetric perturbation E, row abs-sum bounded by delta (Gershgorin)
    raw = {}
    for i in range(n):
        for o in range(1, B_BW + 1):
            j = i + o
            if j < n:
                raw[(i, j)] = rng.uniform(-1.0, 1.0)
    rowsum = np.zeros(n)
    for (i, j), v in raw.items():
        rowsum[i] += abs(v)
        rowsum[j] += abs(v)
    Mraw = rowsum.max() if raw else 1.0
    scale = (delta / Mraw) if Mraw > 0 else 0.0

    entries = []  # (i, j, val), i<=j, includes diagonal
    for i in range(n):
        entries.append((i, i, float(diag[i])))
    for (i, j), v in raw.items():
        entries.append((i, j, float(v * scale)))
    entries.sort(key=lambda t: (t[0], t[1]))

    probes = rng.normal(size=(K_PROBES, n))
    probes = probes / np.linalg.norm(probes, axis=1, keepdims=True)

    out = []
    out.append(str(n))
    out.append(f"{THETA:.10g} {EPS:.10g} {B_BW}")
    out.append(f"{gap:.10g}")
    out.append(str(len(entries)))
    for i, j, v in entries:
        out.append(f"{i} {j} {v:.12g}")
    out.append(str(K_PROBES))
    for p in probes:
        out.append(" ".join(f"{x:.12g}" for x in p))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
