# TIER: greedy
"""The obvious, standard approach: rank candidate mutations by their individually
reported stability delta, then stack the best ones while a NAIVE additive activity
budget holds. This never consults the epistasis table and never limits how many
selected mutations cluster in the active-site neighbourhood -- exactly the
recipe the innovation hook says is the trap."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); K = int(nxt()); C = int(nxt()); R = int(nxt())
    A0 = float(nxt()); ActMin = float(nxt()); alpha = float(nxt())
    dstab = [0.0] * n
    dact = [0.0] * n
    dist = [0] * n
    for i in range(n):
        dstab[i] = float(nxt())
        dact[i] = float(nxt())
        dist[i] = int(nxt())
    # (epistasis table and active-site crowding are read from the same stream by the
    # checker, but the "obvious" strategy never looks at them.)

    order = sorted(range(n), key=lambda i: -dstab[i])
    chosen = []
    naive_act = A0
    for i in order:
        if len(chosen) >= K:
            break
        if naive_act + dact[i] >= ActMin - 1e-9:
            chosen.append(i)
            naive_act += dact[i]

    print(len(chosen))
    print(*chosen)


if __name__ == "__main__":
    main()
