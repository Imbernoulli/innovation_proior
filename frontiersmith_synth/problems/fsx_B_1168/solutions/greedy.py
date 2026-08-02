# TIER: greedy
# The "obvious" recipe: never look at the redundancy graph at all. For each sensor, compare
# its raw trace against the network-wide per-timestep MEDIAN and flag whoever has the single
# most extreme spike (a natural first instinct: "find the outlier reading"). "Fix" it with a
# flat, whole-record mean offset. This has two blind spots by construction: (1) a brief, large
# genuine local event produces a bigger single-timestep spike than a modest but persistent
# instrument bias, so the event -- not the real fault -- wins the budget; (2) a flat-offset
# correction cannot represent a drifting bias at all, so even a correctly-flagged drift fault
# is only partly repaired.
import sys


def median(xs):
    s = sorted(xs)
    m = len(s)
    if m % 2 == 1:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it))
    n = int(next(it)); T = int(next(it)); F_max = int(next(it))
    m_edges = int(next(it))
    for _ in range(m_edges):
        next(it); next(it)                      # edges parsed but intentionally unused
    R = [[0.0] * T for _ in range(n)]
    for i in range(n):
        for t in range(T):
            R[i][t] = float(next(it))

    med = [median([R[i][t] for i in range(n)]) for t in range(T)]

    scored = []
    offs = {}
    for i in range(n):
        dev = [R[i][t] - med[t] for t in range(T)]
        peak = max(abs(x) for x in dev)
        offs[i] = sum(dev) / T                  # naive flat-offset "fix" (no drift term)
        scored.append((peak, i))
    scored.sort(reverse=True)

    thresh = 0.6
    chosen = [i for (p, i) in scored[:F_max] if p > thresh]

    out = [str(len(chosen))]
    for i in chosen:
        out.append(f"{i} {offs[i]:.6f} 0.0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
