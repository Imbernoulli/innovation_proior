#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  (ans ignored) -- Format D op-count scorer.

Verifies the submitted artifact is an exact permutation of the N experiment indices
(every required configuration visited exactly once), then counts the exact total
reconfiguration cost:

    F = w(0, cfg[p0]) + sum_k w(cfg[p_{k-1}], cfg[p_k]),   w(x,y) = sum_{j: x_j!=y_j} c_j

(integer weighted toggle count; no timing, no randomness). Internal baseline B = the
same cost of the input order (0,1,...,N-1), which `trivial` reproduces (~0.1).

    ratio = min(1.0, 0.1 * B / F)

ANY feasibility violation, non-integer token, wrong token count, duplicate or
out-of-range index -> Ratio 0.0.
"""
import sys


def die0(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inpath, outpath = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    try:
        with open(inpath, "rb") as f:
            toks = f.read().decode("utf-8", "strict").split()
    except Exception:
        die0("cannot read instance")
    try:
        it = iter(toks)
        N = int(next(it))
        M = int(next(it))
        if not (1 <= N <= 1000000) or not (1 <= M <= 60):
            die0("bad N/M header")
        C = []
        for _ in range(M):
            c = int(next(it))
            if not (1 <= c <= 10 ** 9):
                die0("bad line cost")
            C.append(c)
        cfg = []
        seen_cfg = set()
        for _ in range(N):
            s = next(it)
            if len(s) != M or any(ch not in "01" for ch in s):
                die0("bad configuration string")
            v = 0
            for j, ch in enumerate(s):
                if ch == '1':
                    v |= (1 << j)
            cfg.append(v)
            seen_cfg.add(v)
        if len(seen_cfg) != N:
            die0("duplicate configurations in instance")
    except StopIteration:
        die0("truncated instance")
    except ValueError:
        die0("non-integer token in instance")

    # ---- read submission ----
    try:
        with open(outpath, "rb") as f:
            raw = f.read(20_000_000)
    except Exception:
        die0("cannot read output")
    try:
        otoks = raw.decode("utf-8", "strict").split()
    except Exception:
        die0("output is not valid utf-8")
    if len(otoks) != N:
        die0("expected %d indices, got %d" % (N, len(otoks)))
    perm = []
    for t in otoks:
        try:
            v = int(t, 10)
        except ValueError:
            die0("non-integer token %r" % t[:24])
        if v < 0 or v >= N:
            die0("index %d out of range" % v)
        perm.append(v)
    if len(set(perm)) != N:
        die0("indices are not all distinct")

    # ---- exact toggle count ----
    def wdiff(x, y):
        d = x ^ y
        s = 0
        j = 0
        while d:
            if d & 1:
                s += C[j]
            d >>= 1
            j += 1
        return s

    def path_cost(order):
        total = wdiff(0, cfg[order[0]])
        prev = cfg[order[0]]
        for k in range(1, len(order)):
            cur = cfg[order[k]]
            total += wdiff(prev, cur)
            prev = cur
        return total

    F = path_cost(perm)
    B = path_cost(list(range(N)))
    if F <= 0 or B <= 0:
        die0("degenerate instance cost")

    sc = min(1000.0, 100.0 * B / float(F))
    print("F=%d B=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
