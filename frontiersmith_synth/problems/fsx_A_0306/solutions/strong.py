# TIER: strong
# Seeded multi-restart randomized hill-climb (baseline seed + random seeds), bounded evaluation
# budget so it stays well within the time limit while beating the deterministic greedy.
import sys, random

def bareiss_absdet(M):
    n = len(M)
    A = [row[:] for row in M]
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = None
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    piv = r
                    break
            if piv is None:
                return 0
            A[k], A[piv] = A[piv], A[k]
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]; Ai = A[i]; Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return abs(A[n - 1][n - 1])

def hillclimb(N, fixed, start, free, rng, max_passes):
    M = [row[:] for row in start]
    cur = bareiss_absdet(M)
    for _p in range(max_passes):
        improved = False
        order = free[:]
        rng.shuffle(order)
        for (i, j) in order:
            M[i][j] = -M[i][j]
            dd = bareiss_absdet(M)
            if dd > cur:
                cur = dd; improved = True
            else:
                M[i][j] = -M[i][j]
        if not improved:
            break
    return cur, M

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); K = int(next(it))
    fixed = {}
    for _ in range(K):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        fixed[(r, c)] = v
    free = [(i, j) for i in range(N) for j in range(N) if (i, j) not in fixed]
    nfree = max(1, len(free))

    # evaluation budget -> total number of full sweeps allowed (keeps large-N runs fast)
    total_passes = max(4, min(12, 5000 // nfree))
    passes_per_restart = 3
    rng = random.Random(20260701)

    base = [[fixed.get((i, j), 1 if i >= j else -1) for j in range(N)] for i in range(N)]
    best_det, best_M = hillclimb(N, fixed, base, free, rng, min(passes_per_restart, total_passes))
    used = passes_per_restart

    while used < total_passes:
        start = [[fixed.get((i, j), rng.choice([-1, 1])) for j in range(N)] for i in range(N)]
        p = min(passes_per_restart, total_passes - used)
        det, M = hillclimb(N, fixed, start, free, rng, p)
        used += p
        if det > best_det:
            best_det, best_M = det, M

    M = best_M
    sys.stdout.write("\n".join(" ".join(str(M[i][j]) for j in range(N)) for i in range(N)) + "\n")

main()
