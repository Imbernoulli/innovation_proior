import sys
from collections import deque

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    dat = open(path).read().split()
    it = iter(dat)
    n = int(next(it)); m = int(next(it)); B = int(next(it)); W = int(next(it))
    theta = [0] * n; r = [0] * n
    for i in range(n):
        theta[i] = int(next(it)); r[i] = int(next(it))
    succ = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)); v = int(next(it))
        succ[u].append(v)
    return n, m, B, W, theta, r, succ

def cascade(n, W, theta, r, succ, x):
    indeg = [0] * n
    adopted = bytearray(n)
    q = deque()
    for i in range(n):
        if x[i] * r[i] >= theta[i]:
            adopted[i] = 1; q.append(i)
    while q:
        u = q.popleft()
        for v in succ[u]:
            if adopted[v]:
                continue
            indeg[v] += 1
            if x[v] * r[v] + W * indeg[v] >= theta[v]:
                adopted[v] = 1; q.append(v)
    return sum(adopted)

def main():
    n, m, B, W, theta, r, succ = read_instance(sys.argv[1])

    # ---- parse participant output: exactly n nonnegative integers ----
    toks = open(sys.argv[2]).read().split()
    if len(toks) != n:
        fail("expected %d rebate values, got %d" % (n, len(toks)))
    x = [0] * n
    tot = 0
    for i, tk in enumerate(toks):
        try:
            v = int(tk)                     # rejects nan/inf/floats/garbage
        except Exception:
            fail("non-integer rebate %r" % tk)
        if v < 0:
            fail("negative rebate")
        x[i] = v
        tot += v
    if tot > B:
        fail("budget exceeded: sum=%d > B=%d" % (tot, B))

    F = cascade(n, W, theta, r, succ, x)

    # ---- internal baseline: uniform equal subsidy floor(B/n) to every household ----
    u = B // n
    B0 = cascade(n, W, theta, r, succ, [u] * n)
    B0 = max(1, B0)

    sc = min(1000.0, 100.0 * F / max(1e-9, B0))
    print("F=%d B0=%d Ratio: %.6f" % (F, B0, sc / 1000.0))

if __name__ == "__main__":
    main()
