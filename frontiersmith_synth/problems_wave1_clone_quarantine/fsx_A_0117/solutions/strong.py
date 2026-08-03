# TIER: strong
# Marginal-gain greedy seed + multi-restart hill-climbing local search.
# Build a strong constructive layout, then repeatedly try single-slot relocations
# (move an antenna to a currently-empty grid point) accepting any non-worsening
# move. Several deterministic seeds let it escape the greedy layout's collisions.
# All randomness is seeded, so the output is bit-for-bit deterministic.
import sys, random

def channels(A):
    s = set(); d = set()
    for a in A:
        for b in A:
            s.add(a + b); d.add(a - b)
    return len(s) + len(d)

def greedy_marg(n, M):
    A = [0, M] if M >= 1 and n >= 2 else [0]
    cur = set(A)
    while len(A) < n:
        best = None; bv = -1
        for x in range(M + 1):
            if x in cur:
                continue
            v = channels(A + [x])
            if v > bv:
                bv = v; best = x
        A.append(best); cur.add(best)
    return A

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])

    A0 = greedy_marg(n, M)
    best_set = sorted(A0)
    best_val = channels(A0)

    iters = 9000 if n <= 20 else 6000
    for seed in range(5):
        random.seed(seed)
        A = set(A0)
        cur = channels(list(A))
        Alist = list(A)
        for _ in range(iters):
            a = random.choice(Alist)
            x = random.randint(0, M)
            if x in A:
                continue
            B = set(A); B.discard(a); B.add(x)
            v = channels(list(B))
            if v >= cur:
                A = B; cur = v; Alist = list(A)
        if cur > best_val:
            best_val = cur; best_set = sorted(A)

    out = [str(n)] + [str(v) for v in best_set]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
