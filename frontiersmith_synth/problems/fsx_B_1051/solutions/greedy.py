# TIER: greedy
"""The obvious first attempt: this looks like the classical 'sequenceable
group' construction, so build the parade with a single deterministic greedy
pass that chases ONLY the drift term (P) -- at every step append the unused
clan, in fixed lexicographic scan order, that makes the running drift land on
a value never seen before. The banner-gap term (D) is never looked at, so it
is left to whatever falls out of the P-only choices.

This single pass has no restarts and no repair, so once it exhausts drift-new
candidates it just keeps taking the first remaining clan, cascading into runs
of repeated banner-gaps -- D collapses to roughly O(sqrt(n)) while P stays
high. On the larger / trap instances this leaves F = P + D far below what a
construction that also watches D can reach.
"""
import sys


def main():
    data = sys.stdin.read().split()
    n1, n2 = int(data[0]), int(data[1])
    n = n1 * n2

    elems = [(a, b) for a in range(n1) for b in range(n2)]
    used = [False] * n

    cur = (0, 0)
    sums_seen = {cur}
    seq = []

    for _ in range(n):
        pick = None
        for i, x in enumerate(elems):
            if used[i]:
                continue
            ns = ((cur[0] + x[0]) % n1, (cur[1] + x[1]) % n2)
            if ns not in sums_seen:
                pick = i
                break
        if pick is None:
            for i in range(n):
                if not used[i]:
                    pick = i
                    break
        used[pick] = True
        x = elems[pick]
        seq.append(x)
        cur = ((cur[0] + x[0]) % n1, (cur[1] + x[1]) % n2)
        sums_seen.add(cur)

    out = "\n".join(f"{a} {b}" for a, b in seq)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
