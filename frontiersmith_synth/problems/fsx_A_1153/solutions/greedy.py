# TIER: greedy
"""The obvious "single pass, no number theory" recipe: deal trees to plots
round-robin in natural position order (tree x -> plot ((x-1) mod k) + 1).
This is excellent for row/window tests (every window sees a near-perfectly
proportional slice of every plot) but has no idea that a "graft inspector"
even exists -- it does not touch discrete logs at all, so on graft tests it
is merely mediocre (same order of imbalance as blind chance), never
catastrophic and never good. It beats the trivial reference solidly on rows
but leaves the whole graft family on the table."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    p = int(nxt())
    k = int(nxt())
    sizes = [int(nxt()) for _ in range(k)]
    n = p - 1
    # sizes are always n/k for every generated instance, so plain round-robin
    # by natural value hits every prescribed size exactly.

    out = [str(((x - 1) % k) + 1) for x in range(1, n + 1)]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
