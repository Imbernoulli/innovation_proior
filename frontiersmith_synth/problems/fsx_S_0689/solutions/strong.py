# TIER: strong
# The insight: fidelity is spectral arithmetic, not propagation. A mirror-symmetric chain
# with couplings J_k = c*sqrt(k(N-k)) has an EXACTLY linear eigenvalue spectrum (gap 2c),
# and perfect end-to-end transfer occurs at T = pi/(2c) -- independent of N. So instead of
# searching coupling SPACE, invert the given T directly for the target gap c = pi/(2T),
# lay down that spectral shape on the free edges, then locally refine per-edge to absorb
# the frozen edges / on-site defects that break exact mirror symmetry.
import sys, math, random
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


def refine(N, defects, frozen, free_edges, T, cur, iters, seed, step0, J_LO, J_HI):
    def full(c):
        d = dict(frozen); d.update(c)
        return d

    best = dict(cur)
    best_f = fidelity(N, defects, full(best), T)
    rng = random.Random(seed)
    step = step0
    for _ in range(iters):
        e = rng.choice(free_edges)
        old = best[e]
        improved = False
        for cand in (old + step, old - step):
            cand = clip(cand, J_LO, J_HI)
            trial = dict(best); trial[e] = cand
            f = fidelity(N, defects, full(trial), T)
            if f > best_f:
                best_f, best = f, trial
                improved = True
        step *= 0.985 if improved else 0.9
    return best, best_f


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def main():
    N, T, J_LO, J_HI, defects, frozen, free_edges = read_instance()

    c = math.pi / (2.0 * T)
    shape_init = {e: clip(c * math.sqrt(e * (N - e)), J_LO, J_HI) for e in free_edges}

    best, best_f = refine(N, defects, frozen, free_edges, T, shape_init,
                           iters=260, seed=4321, step0=0.35 * (J_HI - J_LO),
                           J_LO=J_LO, J_HI=J_HI)

    # a couple of cheap extra restarts (perturbed shape) -- keep the best
    for r, seed in enumerate((1717, 9182)):
        rng = random.Random(seed)
        pert = {e: clip(shape_init[e] * (0.7 + 0.6 * rng.random()), J_LO, J_HI) for e in free_edges}
        cand, f = refine(N, defects, frozen, free_edges, T, pert,
                          iters=150, seed=seed + 1, step0=0.25 * (J_HI - J_LO),
                          J_LO=J_LO, J_HI=J_HI)
        if f > best_f:
            best, best_f = cand, f

    print(len(free_edges))
    print(" ".join("%.6f" % best[e] for e in free_edges))


if __name__ == "__main__":
    main()
