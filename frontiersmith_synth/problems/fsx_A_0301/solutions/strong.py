# TIER: strong
# Max over several deterministic weight-greedy constructions:
#   - full space (all shelves), and
#   - for each coordinate k, the pool that drops the slab {digit[k] == 2}.
# The greedy tier's construction (drop last-coord slab) is among these, so this
# never scores below greedy; on most instances a different restriction wins.
import sys

def build(n):
    N = 3 ** n
    D = []
    for i in range(N):
        d = []; x = i
        for _ in range(n):
            d.append(x % 3); x //= 3
        D.append(d)
    return N, D

def wgreedy(pool_sorted, D, pow3, n):
    S = []; F = set()
    for p in pool_sorted:
        if p in F:
            continue
        dp = D[p]
        for a in S:
            da = D[a]; c = 0
            for k in range(n):
                c += ((3 - (dp[k] + da[k]) % 3) % 3) * pow3[k]
            F.add(c)
        S.append(p)
    return S

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); N = 3 ** n
    W = [int(x) for x in toks[1:1 + N]]
    pow3 = [3 ** k for k in range(n)]
    _, D = build(n)

    candidates = [list(range(N))]
    for k in range(n):
        candidates.append([i for i in range(N) if D[i][k] != 2])

    best = None; bestw = -1
    for pool in candidates:
        pool.sort(key=lambda i: (-W[i], i))
        S = wgreedy(pool, D, pow3, n)
        w = sum(W[i] for i in S)
        if w > bestw:
            bestw = w; best = S

    sys.stdout.write(str(len(best)) + "\n")
    sys.stdout.write(" ".join(map(str, best)) + "\n")

if __name__ == "__main__":
    main()
