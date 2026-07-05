# TIER: strong
# Randomized-restart gap-covering greedy: at every step break ties among the
# equally-best depot slots RANDOMLY (fixed seed -> deterministic), run many
# independent restarts and keep the construction with the longest cleared band.
# Exploring the tie-break tree escapes the single locally-greedy trajectory and
# reliably reaches a longer contiguous reach than the deterministic greedy.
import sys, random

def reach(A, M):
    A = sorted(set(A)); mx = A[-1]
    cov = bytearray(2 * mx + 2)
    for a in A:
        for b in A:
            cov[a + b] = 1
            d = a - b
            if d >= 0:
                cov[d] = 1
    N = 0
    while N < len(cov) and cov[N]:
        N += 1
    return N - 1

def one_run(n, M, rnd):
    A = [0]; covset = {0}; gap = 1; Aset = {0}
    while len(A) < n:
        g = gap
        cands = set()
        for a in A:
            for c in (g - a, g + a, a - g):
                if 1 <= c <= M and c not in Aset:
                    cands.add(c)
        if g % 2 == 0:
            c = g // 2
            if 1 <= c <= M and c not in Aset:
                cands.add(c)
        if not cands:
            for c in range(1, M + 1):
                if c not in Aset:
                    cands.add(c); break
        scored = []
        for c in cands:
            newvals = {2 * c}
            for a in A:
                newvals.add(a + c); newvals.add(abs(a - c))
            gg = g
            while (gg in covset) or (gg in newvals):
                gg += 1
            scored.append((gg - 1, c))
        bestr = max(s[0] for s in scored)
        top = [c for r, c in scored if r == bestr]
        c = rnd.choice(top)
        for a in A:
            covset.add(a + c); covset.add(abs(a - c))
        covset.add(2 * c)
        A.append(c); Aset.add(c)
        while gap in covset:
            gap += 1
    return A

def main():
    data = sys.stdin.read().split()
    n, M = int(data[0]), int(data[1])
    rnd = random.Random(98765)
    RESTARTS = 20
    best = None; bestr = -1
    for _ in range(RESTARTS):
        A = one_run(n, M, rnd)
        r = reach(A, M)
        if r > bestr:
            bestr = r; best = A
    out = [str(len(best))] + [str(x) for x in best]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
