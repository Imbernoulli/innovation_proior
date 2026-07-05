# TIER: greedy
# Seeded multi-restart random +/-1 design; keep the best |det| over a few draws.
import sys, random

def bareiss_det(M):
    n = len(M); M = [row[:] for row in M]; sign = 1; prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    sw = i; break
            if sw == -1:
                return 0
            M[k], M[sw] = M[sw], M[k]; sign = -sign
        akk = M[k][k]
        for i in range(k + 1, n):
            aik = M[i][k]; Mi = M[i]; Mk = M[k]
            for j in range(k + 1, n):
                Mi[j] = (Mi[j] * akk - aik * Mk[j]) // prev
        prev = akk
    return sign * M[n - 1][n - 1]

def main():
    N = int(sys.stdin.read().split()[0])
    best = None; bestd = -1
    for s in range(6):
        rng = random.Random(1000 + N * 7 + s)
        M = [[rng.choice((-1, 1)) for _ in range(N)] for _ in range(N)]
        d = abs(bareiss_det(M))
        if d > bestd:
            bestd = d; best = M
    out = [" ".join(str(x) for x in row) for row in best]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
