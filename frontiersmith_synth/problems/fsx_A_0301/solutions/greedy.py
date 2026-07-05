# TIER: greedy
# Weight-descending greedy over a RESTRICTED shelf pool: drop the slab where the
# last coordinate equals 2. Weight-aware (beats the sub-cube) but pool-limited.
import sys

def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); N = 3 ** n
    W = [int(x) for x in toks[1:1 + N]]
    pow3 = [3 ** k for k in range(n)]
    D = []
    for i in range(N):
        d = []; x = i
        for _ in range(n):
            d.append(x % 3); x //= 3
        D.append(d)

    pool = [i for i in range(N) if D[i][n - 1] != 2]
    pool.sort(key=lambda i: (-W[i], i))

    S = []; F = set()
    for p in pool:
        if p in F:
            continue
        dp = D[p]
        for a in S:
            da = D[a]; c = 0
            for k in range(n):
                c += ((3 - (dp[k] + da[k]) % 3) % 3) * pow3[k]
            F.add(c)
        S.append(p)

    sys.stdout.write(str(len(S)) + "\n")
    sys.stdout.write(" ".join(map(str, S)) + "\n")

if __name__ == "__main__":
    main()
