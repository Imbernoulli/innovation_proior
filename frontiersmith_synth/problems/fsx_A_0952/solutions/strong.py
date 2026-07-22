# TIER: strong
# The insight: this is NOT "a curve to fit" -- it is a two-branch recursion in
# disguise. Define the residual R(n) = T(n) - T(floor(n/A)) - T(ceil(n/B)).
# For the WRONG (A,B) this residual is a noisy mess. For the RIGHT (A,B) it
# collapses to an EXACT integer function of n alone: R(n) = C*n + g(n mod M).
# Search the small (A,B) grid; for each, solve C from two same-residue ledger
# rows (an exact division, not a fit) and verify the identity holds bit-for-
# bit across the WHOLE ledger. Because the ledger has thousands of rows, a
# wrong hypothesis is exposed instantly (it can't satisfy an EXACT identity by
# accident); the right one is confirmed by total agreement. Emit the recovered
# two-branch recursive program -- it reproduces the true op-count exactly at
# any extrapolated order size, because it computes it exactly the same way
# the kitchen's routine does.
import sys

PAIR_CANDIDATES = [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (2, 6), (3, 6), (4, 5)]
M_CANDIDATES = [4, 5, 6, 7, 8]


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("BASE 1 0 0"); print("REC n"); return
    n_train = int(data[0])
    vals = data[2:]
    ts = [0] * (n_train + 1)
    for i in range(n_train + 1):
        nn = int(vals[2 * i])
        vv = int(vals[2 * i + 1])
        ts[nn] = vv

    found = None
    for a, b in PAIR_CANDIDATES:
        # residual R(n) for n = 2..n_train
        R = {}
        ok = True
        for n in range(2, n_train + 1):
            lo = n // a
            hi = -(-n // b)
            if lo >= n or hi >= n or lo > n_train or hi > n_train:
                ok = False
                break
            R[n] = ts[n] - ts[lo] - ts[hi]
        if not ok:
            continue
        for m in M_CANDIDATES:
            groups = [[] for _ in range(m)]
            for n, r in R.items():
                groups[n % m].append((n, r))
            # need at least 2 points in some group to pin down C exactly
            c_val = None
            for grp in groups:
                if len(grp) >= 2:
                    (n1, r1), (n2, r2) = grp[0], grp[1]
                    if n1 == n2:
                        continue
                    dn = n1 - n2
                    dr = r1 - r2
                    if dn != 0 and dr % dn == 0:
                        c_val = dr // dn
                        break
            if c_val is None:
                continue
            # verify EXACT identity across every training row, and extract g[]
            g = [None] * m
            consistent = True
            for r_idx, grp in enumerate(groups):
                base_d = None
                for n, r in grp:
                    d = r - c_val * n
                    if base_d is None:
                        base_d = d
                    elif d != base_d:
                        consistent = False
                        break
                if not consistent:
                    break
                g[r_idx] = base_d if base_d is not None else 0
            if consistent and all(v is not None for v in g):
                found = (a, b, m, c_val, g)
                break
        if found:
            break

    t0, t1 = ts[0], ts[1] if n_train >= 1 else ts[0]

    if found is None:
        # fallback (should not trigger given the search grid matches the family)
        print("BASE 1 %d %d" % (t0, t1))
        print("REC n")
        return

    a, b, m, c_val, g = found
    print("BASE 1 %d %d" % (t0, t1))
    print("REC T ( FLOORDIV ( n , %d ) ) + T ( CEILDIV ( n , %d ) ) + TAB ( MOD ( n , %d ) , %s ) + %d * n"
          % (a, b, m, " , ".join(str(v) for v in g), c_val))


if __name__ == "__main__":
    main()
