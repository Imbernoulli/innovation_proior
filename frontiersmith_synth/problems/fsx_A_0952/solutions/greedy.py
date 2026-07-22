# TIER: greedy
# The obvious first move: this LOOKS like op-count data from a divide-and-batch
# routine, so fit a smooth closed-form curve to it -- a straight line (least
# squares) capturing the dominant per-portion cost, plus (being a little extra
# careful) a mod-5 residual-averaging correction in case there is some cheap
# periodic pattern left over. Both are excellent fits on the training range and
# even just past it. What this NEVER considers is that T(n) is defined in
# terms of T at SMALLER arguments (a genuine recursion) -- so it silently
# drops every correction injected by the recursive substructure below the top
# level, and that dropped mass grows (like n^0.7-0.8, not O(1)) as n grows.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("BASE 1 0 0"); print("REC n"); return
    n_train = int(data[0])
    vals = data[2:]
    ns = []
    ts = []
    for i in range(n_train + 1):
        ns.append(int(vals[2 * i]))
        ts.append(int(vals[2 * i + 1]))

    # least-squares line T(n) ~= A*n + B
    N = len(ns)
    sx = sum(ns); sy = sum(ts)
    sxx = sum(x * x for x in ns); sxy = sum(x * y for x, y in zip(ns, ts))
    denom = N * sxx - sx * sx
    if denom == 0:
        A, B = 1.0, 0.0
    else:
        A = (N * sxy - sx * sy) / denom
        B = (sy - A * sx) / N
    A_r = int(round(A))
    B_r = int(round(B))

    # mod-5 residual-averaging correction (guessing a plausible short period)
    M = 5
    buckets = [[] for _ in range(M)]
    for x, y in zip(ns, ts):
        buckets[x % M].append(y - A_r * x - B_r)
    g = [int(round(sum(b) / len(b))) if b else 0 for b in buckets]

    t0 = ts[0] if ns and ns[0] == 0 else ts[0]
    t1 = ts[1] if len(ts) > 1 else t0

    print("BASE 1 %d %d" % (t0, t1))
    print("REC %d * n + %d + TAB ( MOD ( n , %d ) , %s )"
          % (A_r, B_r, M, " , ".join(str(v) for v in g)))


if __name__ == "__main__":
    main()
