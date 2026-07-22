# TIER: greedy
# The "obvious" approach: model this as a continuous-time quantum walk and tune a single
# GLOBAL coupling strength (uniform profile) by a 1-D grid + local search on the true
# fidelity.  It never adapts the coupling SHAPE around the defects/frozen edges, so on
# instances where a frozen edge is pinned far from the shape the whole profile is off.
import sys
import numpy as np


def read_instance():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    T = float(next(it))
    J_LO = float(next(it)); J_HI = float(next(it))
    D = int(next(it))
    defects = {}
    for _ in range(D):
        s = int(next(it)); v = float(next(it))
        defects[s] = v
    K = int(next(it))
    frozen = {}
    for _ in range(K):
        e = int(next(it)); v = float(next(it))
        frozen[e] = v
    free_edges = sorted(e for e in range(1, N) if e not in frozen)
    return N, T, J_LO, J_HI, defects, frozen, free_edges


def fidelity(N, defects, couplings, T):
    A = np.zeros((N, N), dtype=np.float64)
    for s, v in defects.items():
        A[s - 1, s - 1] = v
    for e, v in couplings.items():
        A[e - 1, e] = v
        A[e, e - 1] = v
    w, V = np.linalg.eigh(A)
    amp = np.sum(V[0, :] * V[N - 1, :] * np.exp(-1j * w * T))
    return abs(complex(amp))


def main():
    N, T, J_LO, J_HI, defects, frozen, free_edges = read_instance()

    def full(g):
        c = dict(frozen)
        for e in free_edges:
            c[e] = g
        return c

    grid = np.geomspace(max(J_LO, 1e-6), J_HI, 60)
    best_f, best_g = -1.0, J_LO
    for g in grid:
        f = fidelity(N, defects, full(g), T)
        if f > best_f:
            best_f, best_g = f, g

    step = (J_HI - J_LO) / 60.0
    g = best_g
    for _ in range(40):
        improved = False
        for cand in (g + step, g - step):
            cand = min(J_HI, max(J_LO, cand))
            f = fidelity(N, defects, full(cand), T)
            if f > best_f:
                best_f, g = f, cand
                improved = True
        step *= 0.8 if improved else 0.6

    print(len(free_edges))
    print(" ".join("%.6f" % g for _ in free_edges))


if __name__ == "__main__":
    main()
