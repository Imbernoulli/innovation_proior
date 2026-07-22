# TIER: greedy
"""The "obvious" textbook approach: a marginal-gain greedy FILL.  Start every
segment at the minimum stiffness (1), then repeatedly hand the next available
budget unit to whichever single segment currently yields the biggest drop in
the resonance score, re-evaluating the full spectrum after each unit.  This is
exactly what an average strong coder writes first for a "budget allocation"
objective -- a knapsack-style marginal-gain construction.

It ONLY ever ADDS stiffness (never re-balances what it already committed), so
it greedily piles stiffness onto whichever few segments look best step by
step.  That produces a localized "defect" stiffening pattern (like a strong
patch on 1-2 segments), which pulls only a HANDFUL of eigenfrequencies out of
the band -- it never discovers a coordinated, alternating (periodic) pattern,
so most of the densely-packed comb still resonates."""
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


def main():
    data = sys.stdin.read().split()
    S = int(data[0]); BUDGET = int(data[1])
    eps = float(data[2])
    F = int(data[3])
    rest = data[4:]
    lines = [(float(rest[2 * i]), int(rest[2 * i + 1])) for i in range(F)]

    ks = [1] * S
    remaining = BUDGET - S
    for _ in range(remaining):
        best_i, best_s = None, None
        for i in range(S):
            ks[i] += 1
            s = resonance_score(eigenfrequencies(ks), lines, eps)
            ks[i] -= 1
            if best_s is None or s < best_s - 1e-12:
                best_s, best_i = s, i
        ks[best_i] += 1

    print(" ".join(map(str, ks)))


if __name__ == "__main__":
    main()
