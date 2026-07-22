import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    m = int(next(it)); n = int(next(it)); R = int(next(it))
    home = [0]*n; fp = [0]*n
    for i in range(n):
        home[i] = int(next(it)); fp[i] = int(next(it))
    a = [0]*m
    for g in range(m):
        a[g] = int(next(it))
    return m, n, R, home, fp, a

def next_empty_array(occupied, m):
    # nextEmpty[g] = smallest circular position >= g that is empty.
    # at least one empty exists.
    ne = [0]*m
    last = -1
    for i in range(2*m - 1, -1, -1):
        p = i % m
        if not occupied[p]:
            last = p
        if i < m:
            ne[p] = last
    return ne

def cost_of(berth, m, n, R, home, fp, a):
    # berth[i] = berth of boat i.  Returns total cost, or None if infeasible.
    occ = [False]*m
    for i in range(n):
        s = berth[i]
        if s < 0 or s >= m:
            return None
        d = (s - home[i]) % m
        if d > R:
            return None
        if occ[s]:
            return None
        occ[s] = True
    present = 0
    for i in range(n):
        d = (berth[i] - home[i]) % m
        present += fp[i] * (d + 1)
    ne = next_empty_array(occ, m)
    absent = 0
    for g in range(m):
        if a[g]:
            dist = (ne[g] - g) % m
            absent += a[g] * (dist + 1)
    return present + absent

def baseline_first_fit(m, n, R, home, fp, a):
    # first-fit by (home, index): earliest free berth in [h, h+R] (circular).
    occ = [False]*m
    berth = [-1]*n
    order = sorted(range(n), key=lambda i: (home[i], i))
    for i in order:
        h = home[i]
        placed = False
        for d in range(R + 1):
            s = (h + d) % m
            if not occ[s]:
                occ[s] = True
                berth[i] = s
                placed = True
                break
        if not placed:
            return None  # should not happen on generated instances
    return cost_of(berth, m, n, R, home, fp, a)

def main():
    m, n, R, home, fp, a = read_instance(sys.argv[1])

    out_toks = open(sys.argv[2]).read().split()
    if len(out_toks) != n:
        fail("expected %d berths, got %d" % (n, len(out_toks)))
    berth = [0]*n
    seen = [False]*m
    for i in range(n):
        t = out_toks[i]
        try:
            v = int(t)
        except Exception:
            fail("non-integer berth %r" % t)
        if not math.isfinite(v):
            fail("non-finite")
        if v < 0 or v >= m:
            fail("berth out of range")
        d = (v - home[i]) % m
        if d > R:
            fail("berth outside reach window for boat %d" % i)
        if seen[v]:
            fail("berth %d used twice" % v)
        seen[v] = True
        berth[i] = v

    F = cost_of(berth, m, n, R, home, fp, a)
    if F is None:
        fail("infeasible")
    F = max(1e-9, float(F))

    B = baseline_first_fit(m, n, R, home, fp, a)
    if B is None or B <= 0:
        B = F  # degenerate guard
    B = float(B)

    sc = min(1000.0, 100.0 * B / F)
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    print("cost=%d baseline=%d Ratio: %.6f" % (int(F), int(B), ratio))

if __name__ == "__main__":
    main()
