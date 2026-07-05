# TIER: greedy
# Seeded multi-restart random search: draw several random permutations and keep
# the one with the smallest linearity (largest nonlinearity).  Reliably beats
# the affine identity baseline, but random sampling stays well short of the
# structured algebraic optimum.
import sys, random

def linearity(S, n):
    N = 1 << n
    pc = [0] * N
    for i in range(1, N):
        pc[i] = pc[i >> 1] + (i & 1)
    Lmax = 0
    for b in range(1, N):
        f = [1 - 2 * (pc[b & S[x]] & 1) for x in range(N)]
        h = 1
        while h < N:
            for i in range(0, N, 2 * h):
                for j in range(i, i + h):
                    a = f[j]; c = f[j + h]
                    f[j] = a + c; f[j + h] = a - c
            h *= 2
        m = max(abs(v) for v in f)
        if m > Lmax:
            Lmax = m
    return Lmax

def main():
    n = int(sys.stdin.read().split()[0])
    N = 1 << n
    rng = random.Random(20260701 + n)
    restarts = 12 if n <= 7 else 6
    best = list(range(N)); best_L = None
    for _ in range(restarts):
        p = list(range(N)); rng.shuffle(p)
        L = linearity(p, n)
        if best_L is None or L < best_L:
            best_L = L; best = p
    sys.stdout.write("\n".join(str(v) for v in best) + "\n")

if __name__ == "__main__":
    main()
