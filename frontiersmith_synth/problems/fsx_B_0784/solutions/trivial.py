# TIER: trivial
# Reproduces the checker's own internal baseline: head 0 alone visits every stroke
# in the given input order (ascending rail position), moving directly to each one,
# swapping color only when it changes, painting; the other heads never move.
# Since only one head ever moves, throttling never triggers -- this is the
# "do nothing clever" reference, so it should land at ~0.1.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    L = int(next(it)); H = int(next(it)); D = int(next(it)); S = int(next(it))
    K = int(next(it)); N = int(next(it))
    starts = [int(next(it)) for _ in range(H)]
    order = []
    for _ in range(N):
        p = int(next(it)); c = int(next(it))
        order.append((p, c))

    ops0 = []
    pos = starts[0]
    cur_color = 0
    for (p, c) in order:
        dx = 1 if p >= pos else -1
        while pos != p:
            ops0.append("M %d" % dx)
            pos += dx
        if cur_color != c:
            ops0.append("S %d" % c)
            cur_color = c
        ops0.append("P")

    out = [str(H)]
    out.append(str(len(ops0)))
    out.extend(ops0)
    for h in range(1, H):
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
