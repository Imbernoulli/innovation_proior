import sys, random, subprocess

SOL = "/tmp/cpv4-dp-1d-boundary_sol"

def gen(seed, nmax=11, vmax=9, kmax=50, slack=2):
    rng = random.Random(seed)
    n = rng.randint(0, nmax)
    L = rng.randint(1, max(1, n) + slack)
    R = rng.randint(L, max(L, n) + slack)
    K = rng.randint(0, kmax)
    vals = [rng.randint(-vmax, vmax) for _ in range(n)]
    return f"{n} {K} {L} {R}\n" + " ".join(map(str, vals)) + "\n"

def brute(data):
    it = iter(data.split())
    n = int(next(it)); K = int(next(it)); L = int(next(it)); R = int(next(it))
    v = [int(next(it)) for _ in range(n)]
    S = [0]*(n+1)
    for i in range(n):
        S[i+1] = S[i] + v[i]
    INF = float('inf')
    best = [INF]
    def rec(pos, acc):
        if acc >= best[0]:
            return
        if pos == n:
            best[0] = min(best[0], acc)
            return
        for length in range(L, R+1):
            np = pos + length
            if np > n:
                break
            seg = S[np] - S[pos]
            rec(np, acc + K + abs(seg))
    rec(0, 0)
    return "-1" if best[0] == INF else str(best[0])

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    base = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    fails = 0
    for s in range(base, base + N):
        data = gen(s)
        got = subprocess.run([SOL], input=data, capture_output=True, text=True).stdout.strip()
        exp = brute(data)
        if got != exp:
            fails += 1
            if fails <= 8:
                print("MISMATCH seed", s, "sol=[%s]" % got, "brute=[%s]" % exp)
                print(data)
    print("TOTAL", N, "FAILS", fails)

main()
