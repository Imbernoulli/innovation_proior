#!/usr/bin/env python3
# Independent brute force for Functional-Graph Cycle Arithmetic.
# PURE simulation: for each query (start, t), take t single steps and report
# where we land. This is the most obviously-correct possible oracle. To keep it
# usable as a differential oracle it relies on the small-case generator keeping
# t bounded (gen.py caps t). It still tolerates moderately large t via an
# optional fast-forward ONLY after a node repeats, which is a provably-equivalent
# shortcut (the orbit is eventually periodic). The shortcut is disabled when
# t is small so the common path is literally a step-by-step loop.
import sys

def solve(data):
    it = iter(data)
    n = int(next(it))
    f = [int(next(it)) for _ in range(n)]
    q = int(next(it))
    out = []
    for _ in range(q):
        s = int(next(it)); t = int(next(it))
        cur = s
        # Pure step-by-step simulation with an equivalence-preserving guard:
        # if t is huge we fall back to detecting the period; otherwise we just
        # iterate t times. Correctness of the period fallback: a functional
        # graph orbit is ultimately periodic, so once a node repeats at global
        # step `first`, every later position is order[first + (k-first) % len].
        if t <= 2_000_000:
            for _ in range(t):
                cur = f[cur]
            out.append(str(cur))
        else:
            seen = {}
            order = []
            step = 0
            while True:
                if cur in seen:
                    first = seen[cur]
                    clen = step - first
                    rem = (t - first) % clen
                    out.append(str(order[first + rem]))
                    break
                seen[cur] = step
                order.append(cur)
                cur = f[cur]
                step += 1
    return "\n".join(out)

def main():
    data = sys.stdin.read().split()
    res = solve(data)
    sys.stdout.write(res + ("\n" if res else ""))

if __name__ == "__main__":
    main()
