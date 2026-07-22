# TIER: strong
"""Co-designs the mass loading and the mode ordering it produces: builds the
same generalized eigenproblem the checker will solve, then runs a bounded,
deterministic local search that scores every CANDIDATE loading by actually
resolving the spectrum and reading off whichever mode lands at rank k -- not
by any static heuristic about "where the targets are". This directly exploits
the problem's insight: since mode k's identity depends on the loading, the
only reliable way to steer it is to simulate the candidate and check what you
actually got, then keep adjusting."""
import sys
import numpy as np

EPS = 1e-6
TAU = 0.2


def build_K(N):
    Nc = N * N
    K = np.zeros((Nc, Nc))
    for r in range(N):
        for c in range(N):
            i = r * N + c
            K[i, i] = 4.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    K[i, rr * N + cc] = -1.0
    return K


def evaluate(K, N, k, loads, targets):
    Nc = N * N
    m = np.array([1.0 + i * EPS + loads[i] for i in range(Nc)])
    s = 1.0 / np.sqrt(m)
    L = (s[:, None] * K) * s[None, :]
    w, U = np.linalg.eigh(L)
    u = U[:, k - 1]
    v = s * u
    maxabs = float(np.max(np.abs(v)))
    if maxabs < 1e-12:
        return -1.0
    vhat = v / maxabs
    F = 0.0
    for c in targets:
        F += max(0.0, 1.0 - abs(vhat[c]) / TAU)
    return F


def clamp_feasible(loads, cap, budget):
    for i in range(len(loads)):
        loads[i] = max(0, min(cap, loads[i]))
    total = sum(loads)
    if total > budget:
        # trim from the largest loads first (deterministic order) until feasible
        order = sorted(range(len(loads)), key=lambda i: (-loads[i], i))
        idx = 0
        while total > budget:
            i = order[idx % len(order)]
            if loads[i] > 0:
                loads[i] -= 1
                total -= 1
            idx += 1
            if idx > 100000:
                break
    return loads


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    k = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1
    budget = int(toks[p]); p += 1
    t = int(toks[p]); p += 1
    targets = [int(toks[p + i]) for i in range(t)]
    p += t

    Nc = N * N
    K = build_K(N)

    # deterministic RNG seeded purely from the instance (no wall-clock)
    seed = (N * 1_000_003 + k * 9973 + budget * 131 + sum(targets) * 17) % (2**31 - 1)
    rng = np.random.RandomState(seed)

    def uniform_loads():
        base = budget // Nc
        rem = budget - base * Nc
        loads = [min(cap, base) for _ in range(Nc)]
        for i in range(rem):
            loads[i] = min(cap, loads[i] + 1)
        return loads

    def target_max_loads():
        loads = [0] * Nc
        remaining = budget
        for c in targets:
            if remaining <= 0:
                break
            add = min(cap, remaining)
            loads[c] = add
            remaining -= add
        return loads

    def target_zero_loads():
        # spend the budget entirely AWAY from the target cells
        loads = [0] * Nc
        others = [i for i in range(Nc) if i not in targets]
        remaining = budget
        oi = 0
        while remaining > 0 and others:
            i = others[oi % len(others)]
            if loads[i] < cap:
                loads[i] += 1
                remaining -= 1
            oi += 1
            if oi > 20 * len(others):
                break
        return loads

    seeds = [
        [0] * Nc,
        uniform_loads(),
        target_max_loads(),
        target_zero_loads(),
    ]
    for _ in range(4):
        loads = [int(rng.randint(0, cap + 1)) for _ in range(Nc)]
        loads = clamp_feasible(loads, cap, budget)
        seeds.append(loads)

    best_loads = None
    best_F = -1.0

    ITERS = 250
    for init in seeds:
        cur = list(init)
        cur = clamp_feasible(cur, cap, budget)
        cur_F = evaluate(K, N, k, cur, targets)
        if cur_F > best_F:
            best_F, best_loads = cur_F, list(cur)

        for it in range(ITERS):
            move_kind = rng.randint(0, 3)
            trial = list(cur)
            if move_kind == 0:
                # increment a random cell if budget allows
                i = int(rng.randint(0, Nc))
                if sum(trial) < budget and trial[i] < cap:
                    trial[i] += 1
            elif move_kind == 1:
                # decrement a random cell
                i = int(rng.randint(0, Nc))
                if trial[i] > 0:
                    trial[i] -= 1
            else:
                # transfer one unit between two random cells
                i = int(rng.randint(0, Nc))
                j = int(rng.randint(0, Nc))
                if trial[i] > 0 and trial[j] < cap and i != j:
                    trial[i] -= 1
                    trial[j] += 1

            if sum(trial) > budget:
                continue
            trial_F = evaluate(K, N, k, trial, targets)

            # simple annealed hill-climb: always accept improvements, accept
            # sideways/slightly-worse moves early to escape local optima
            temp = max(0.0, 1.0 - it / ITERS) * 0.3
            if trial_F >= cur_F - temp * rng.random_sample():
                cur, cur_F = trial, trial_F
            if trial_F > best_F:
                best_F, best_loads = trial_F, list(trial)

    print(" ".join(map(str, best_loads)))


if __name__ == "__main__":
    main()
