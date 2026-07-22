# TIER: trivial
# Literal, macro-free construction: DOT every cell of the orbit closure of the
# anchors directly. R == B exactly -> Ratio == 0.1 on every case.
import sys


def apply_t(t, x, y, N):
    if t == 0: return (x, y)
    if t == 1: return (N - 1 - y, x)
    if t == 2: return (N - 1 - x, N - 1 - y)
    if t == 3: return (y, N - 1 - x)
    if t == 4: return (N - 1 - x, y)
    if t == 5: return (x, N - 1 - y)
    if t == 6: return (y, x)
    if t == 7: return (N - 1 - y, N - 1 - x)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); A = int(next(it))
    anchors = [(int(next(it)), int(next(it))) for _ in range(A)]

    closure = set()
    for (x, y) in anchors:
        for t in range(8):
            closure.add(apply_t(t, x, y, N))

    out = []
    for (x, y) in sorted(closure):
        out.append("DOT %d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
