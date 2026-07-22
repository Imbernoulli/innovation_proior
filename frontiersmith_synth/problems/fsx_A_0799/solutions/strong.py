# TIER: strong
"""The insight: you cannot outrun a comb by shifting the spectrum -- you must
tear holes in it.  Any allocation that stays "close to uniform" (a whole-
spectrum shift/shave, or greedy's localized defect patches) keeps the same
dense mode ladder the comb was planted on and can only relocate a few
eigenfrequencies at a time.

Instead, IMPOSE PERIODIC STRUCTURE on the stiffness allocation: alternate the
budget between "heavy" and "light" segments with period P.  Breaking the
translational symmetry from period-1 (uniform) to period-P folds the
dispersion relation and opens a genuine BAND GAP in the eigenfrequency
spectrum at the zone boundary -- removing a whole *band* of modes at once,
exactly where the drum corps' comb concentrates its heaviest weight (the
generator plants the heavy band in the middle third of the mode index range,
which is where a period-2/3/4 modulation's gap lands).

We search a small, principled family (period in {2..7} x modulation
amplitude), actually evaluating the resulting spectrum against the comb, and
keep the best -- a direct construction, not a local patch-and-hope search."""
import sys
import numpy as np


def eigenfrequencies(ks):
    S = len(ks)
    N = S - 1
    if N <= 0:
        return np.array([])
    K = np.zeros((N, N), dtype=float)
    for i in range(N):
        K[i, i] = ks[i] + ks[i + 1]
    for i in range(N - 1):
        K[i, i + 1] = -ks[i + 1]
        K[i + 1, i] = -ks[i + 1]
    ev = np.clip(np.linalg.eigvalsh(K), 0.0, None)
    return np.sqrt(ev)


def resonance_score(omegas, lines, eps):
    total = 0.0
    for f, w in lines:
        d = float(np.min(np.abs(omegas - f)))
        total += w / (d + eps)
    return total


def periodic_alloc(S, BUDGET, period, amp):
    base = BUDGET // S
    ks = [base] * S
    for i in range(S):
        if (i % period) < (period // 2):
            ks[i] += amp
        else:
            ks[i] -= amp
    ks = [max(1, k) for k in ks]
    # repair to hit the exact budget deterministically (round-robin, low index first)
    diff = BUDGET - sum(ks)
    i = 0
    guard = 0
    while diff != 0 and guard < 100000:
        j = i % S
        if diff > 0:
            ks[j] += 1; diff -= 1
        elif ks[j] > 1:
            ks[j] -= 1; diff += 1
        i += 1
        guard += 1
    return ks


def main():
    data = sys.stdin.read().split()
    S = int(data[0]); BUDGET = int(data[1])
    eps = float(data[2])
    F = int(data[3])
    rest = data[4:]
    lines = [(float(rest[2 * i]), int(rest[2 * i + 1])) for i in range(F)]

    base = BUDGET // S
    max_amp = max(1, base)  # keep every segment >= 1 without needing the repair loop to fight hard

    best_ks, best_score = None, None
    for period in range(2, 8):
        if period > S:
            continue
        for amp in range(1, max_amp):
            ks = periodic_alloc(S, BUDGET, period, amp)
            if sum(ks) != BUDGET or min(ks) < 1:
                continue
            s = resonance_score(eigenfrequencies(ks), lines, eps)
            if best_score is None or s < best_score:
                best_score, best_ks = s, ks

    if best_ks is None:  # degenerate fallback (should not trigger given max_amp>=1 loop bounds)
        best_ks = [base + (1 if i < BUDGET - base * S else 0) for i in range(S)]

    print(" ".join(map(str, best_ks)))


if __name__ == "__main__":
    main()
